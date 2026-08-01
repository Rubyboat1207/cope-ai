// Runs the finetuned model entirely client-side via wllama (WASM + WebGPU),
// no server involved — the model is fetched directly from Hugging Face.
import { Wllama } from "https://cdn.jsdelivr.net/npm/@wllama/wllama@3.5.1/esm/index.js";

const WLLAMA_WASM_URL = "https://cdn.jsdelivr.net/npm/@wllama/wllama@3.5.1/esm/wasm/wllama.wasm";
const HF_BASE = "https://huggingface.co/Rubyboat/cope-ai-v3/resolve/main";

// Mobile browsers hit wllama's ArrayBuffer/WASM heap limits much harder
// than desktop (an Android phone crashed with "Invalid typed array length"
// on the larger Q4_K_M file), and Q3_K_M's ~1.16GB single-buffer download
// itself failed to even fetch on real-device mobile testing for reasons
// that didn't reproduce in any desktop/emulated test — so a model picker
// exists to let people try a different size/URL directly rather than
// requiring a code change + redeploy every time.
export const MODEL_OPTIONS = [
  { id: "q2_k", label: "Q2_K — last resort only, barely coherent (~1.2GB)", url: `${HF_BASE}/model-q2_k.gguf` },
  { id: "q3_k_m", label: "Q3_K_M — small (~1.6GB), default", url: `${HF_BASE}/model-q3_k_m.gguf` },
  { id: "q4_k_m", label: "Q4_K_M — better quality (~1.9GB), desktop recommended", url: `${HF_BASE}/model-q4_k_m.gguf` },
];
const DEFAULT_MODEL_ID = "q3_k_m";
const MODEL_CHOICE_KEY = "cope-ai-model-choice";

export function getSelectedModelId() {
  return localStorage.getItem(MODEL_CHOICE_KEY) || DEFAULT_MODEL_ID;
}

export function setSelectedModelId(id) {
  localStorage.setItem(MODEL_CHOICE_KEY, id);
}

function getSelectedModelUrl() {
  const id = getSelectedModelId();
  return (MODEL_OPTIONS.find((m) => m.id === id) || MODEL_OPTIONS[0]).url;
}

const TARGET_MARKER = "\n<|next|>\n";
const LINE_RE = /^\[([^\]]+)\]\s*([^:]+):\s*(.*)$/;
// how many recent messages to include as context; the model was trained on
// windows up to ~1024 tokens, and wllama no longer exposes a tokenizer to
// count exactly, so this is a message-count approximation of that budget.
const MAX_CONTEXT_MESSAGES = 40;

let wllama = null;
let loadingPromise = null;

function formatGap(seconds) {
  seconds = Math.max(0, Math.round(seconds));
  if (seconds < 60) return `+${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `+${minutes}m`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `+${hours}h`;
  const days = Math.floor(hours / 24);
  return `+${days}d`;
}

export function gapStringToSeconds(gap) {
  const m = gap.match(/\+(\d+)([smhd])/);
  if (!m) return 30;
  const value = parseInt(m[1], 10);
  const unit = { s: 1, m: 60, h: 3600, d: 86400 }[m[2]];
  return value * unit;
}

export function isModelReady() {
  return wllama !== null;
}

// wllama's "stop" option only truncates the final resolved response, not
// the individual streamed chunks — since we read text purely from onData
// (the resolved value carries nothing useful when stream:true), we have to
// enforce the newline stop ourselves or the model keeps rambling past the
// first line and we accumulate multiple garbled lines together.
function firstLine(s) {
  const idx = s.indexOf("\n");
  return idx === -1 ? s : s.substring(0, idx);
}

export async function loadModel(onProgress) {
  if (wllama) return;
  if (loadingPromise) return loadingPromise;

  loadingPromise = (async () => {
    const instance = new Wllama({ default: WLLAMA_WASM_URL });
    const useGpu = instance.isSupportWebGPU ? instance.isSupportWebGPU() : false;

    await instance.loadModelFromUrl(getSelectedModelUrl(), {
      n_gpu_layers: useGpu ? 99999 : 0,
      // matches the ~1024-token training context and halves KV-cache
      // memory vs. 2048 — every bit matters on memory-constrained mobile.
      n_ctx: 1024,
      progressCallback: ({ loaded, total }) => {
        const pct = total ? Math.round((loaded / total) * 100) : 0;
        onProgress?.({ stage: useGpu ? "downloading (GPU mode)" : "downloading (CPU mode)", pct });
      },
    });

    wllama = instance;
  })();

  return loadingPromise;
}

function buildPrompt(messages) {
  const lines = [];
  let prevTs = null;
  for (const m of messages) {
    const ts = new Date(m.timestamp);
    const gap = prevTs === null ? 0 : (ts - prevTs) / 1000;
    prevTs = ts;
    lines.push(`[${formatGap(gap)}] ${m.author_name}: ${m.text}`);
  }
  const kept = lines.slice(-MAX_CONTEXT_MESSAGES);
  return kept.join("\n") + TARGET_MARKER;
}

// Generates just the "[gap]" part of the next line (stopping before the
// colon), so a forced author can be spliced in without letting the model
// pick who talks — used by generateNextMessage when forcedAuthorName is set.
async function generateGapOnly(prompt) {
  const response = await wllama.createCompletion({
    prompt,
    max_tokens: 24,
    temperature: 0.9,
    top_p: 0.9,
    stop: [":"],
  });
  const partial = (response.choices[0].text || "").trim();
  const match = partial.match(/^\[([^\]]+)\]/);
  return match ? match[1] : "+30s";
}

const MAX_GENERATION_ATTEMPTS = 3;

// A generation only counts as usable if the format actually came out right
// and there's real content — an empty message, or free-choice generation
// that never produced a recognizable "[gap] author: text" line at all
// (author stayed "unknown"), means the model went off the rails and is
// worth resampling rather than showing to the user as-is.
function isValidGeneration(authorName, text, forced) {
  if (!text || !text.trim()) return false;
  if (!forced && authorName === "unknown") return false;
  return true;
}

async function attemptForced(prompt, forcedAuthorName, onProgress) {
  onProgress?.({ status: "picking timing...", partialText: "" });
  const gapStr = await generateGapOnly(prompt);

  // Splice in the forced author and resume completion for the message
  // content only, streaming so progress is visible live.
  const forcedPrefix = `[${gapStr}] ${forcedAuthorName}: `;
  let acc = "";
  // wllama's resolved promise doesn't carry the text when streaming —
  // only the onData chunks do — so accumulate it ourselves.
  await wllama.createCompletion({
    prompt: prompt + forcedPrefix,
    max_tokens: 128,
    temperature: 0.9,
    top_p: 0.9,
    repeat_penalty: 1.15,
    stop: ["\n"],
    stream: true,
    onData: (chunk) => {
      acc += chunk.choices[0].text || "";
      onProgress?.({ status: `${forcedAuthorName} is typing...`, partialText: firstLine(acc) });
    },
  });

  return { gapStr, authorName: forcedAuthorName, text: firstLine(acc).trim() };
}

async function attemptFree(prompt, onProgress) {
  let acc = "";
  onProgress?.({ status: "someone is typing...", partialText: "" });
  await wllama.createCompletion({
    prompt,
    max_tokens: 128,
    temperature: 0.9,
    top_p: 0.9,
    repeat_penalty: 1.15,
    stop: ["\n"],
    stream: true,
    onData: (chunk) => {
      acc += chunk.choices[0].text || "";
      const partial = firstLine(acc);
      const partialMatch = partial.match(/^\[([^\]]+)\]\s*([^:]+):\s*([\s\S]*)$/);
      if (partialMatch) {
        onProgress?.({ status: `${partialMatch[2].trim()} is typing...`, partialText: partialMatch[3] });
      } else {
        onProgress?.({ status: "someone is typing...", partialText: "" });
      }
    },
  });

  const line = firstLine(acc).trim();
  const match = line.match(LINE_RE);
  if (match) {
    const [, gapStr, authorNameRaw, textRaw] = match;
    return { gapStr, authorName: authorNameRaw.trim(), text: textRaw.trim() };
  }
  return { gapStr: "+30s", authorName: "unknown", text: line };
}

export async function generateNextMessage(messages, authors, forcedAuthorName = null, onProgress = null) {
  if (!wllama) throw new Error("model not loaded yet");

  const prompt = buildPrompt(messages);
  let result = null;

  for (let attempt = 1; attempt <= MAX_GENERATION_ATTEMPTS; attempt++) {
    if (attempt > 1) {
      onProgress?.({ status: `that came out blank/garbled, regenerating (attempt ${attempt}/${MAX_GENERATION_ATTEMPTS})...`, partialText: "" });
    }

    const candidate = forcedAuthorName
      ? await attemptForced(prompt, forcedAuthorName, onProgress)
      : await attemptFree(prompt, onProgress);

    if (isValidGeneration(candidate.authorName, candidate.text, !!forcedAuthorName)) {
      result = candidate;
      break;
    }
  }

  if (!result) {
    throw new Error(
      `Model produced an empty or malformed message ${MAX_GENERATION_ATTEMPTS} times in a row — try generating again.`
    );
  }

  const { gapStr, authorName, text } = result;

  let authorId = null;
  let avatar = null;
  for (const [id, info] of Object.entries(authors)) {
    if (info.name.toLowerCase() === authorName.toLowerCase()) {
      authorId = id;
      avatar = info.avatar;
      break;
    }
  }

  const gapSeconds = gapStringToSeconds(gapStr);
  const lastTs = messages.length ? new Date(messages[messages.length - 1].timestamp) : new Date();
  const newTs = new Date(lastTs.getTime() + gapSeconds * 1000);

  return {
    author_id: authorId,
    author_name: authorName,
    avatar,
    text,
    gap_seconds: gapSeconds,
    gap_display: gapStr,
    timestamp: newTs.toISOString(),
  };
}
