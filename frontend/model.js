// Runs the finetuned model entirely client-side via wllama (WASM + WebGPU),
// no server involved — the model is fetched directly from Hugging Face.
import { Wllama } from "https://cdn.jsdelivr.net/npm/@wllama/wllama@3.5.1/esm/index.js";

const WLLAMA_WASM_URL = "https://cdn.jsdelivr.net/npm/@wllama/wllama@3.5.1/esm/wasm/wllama.wasm";
const MODEL_URL = "https://huggingface.co/Rubyboat/cope-ai-v3/resolve/main/model-q4_k_m.gguf";

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

export async function loadModel(onProgress) {
  if (wllama) return;
  if (loadingPromise) return loadingPromise;

  loadingPromise = (async () => {
    const instance = new Wllama({ default: WLLAMA_WASM_URL });
    const useGpu = instance.isSupportWebGPU ? instance.isSupportWebGPU() : false;

    await instance.loadModelFromUrl(MODEL_URL, {
      n_gpu_layers: useGpu ? 99999 : 0,
      n_ctx: 2048,
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

export async function generateNextMessage(messages, authors) {
  if (!wllama) throw new Error("model not loaded yet");

  const prompt = buildPrompt(messages);
  const response = await wllama.createCompletion({
    prompt,
    max_tokens: 128,
    temperature: 0.9,
    top_p: 0.9,
    stop: ["\n"],
  });

  const line = (response.choices[0].text || "").trim();
  const match = line.match(LINE_RE);

  let gapStr, authorName, text;
  if (match) {
    [, gapStr, authorName, text] = match;
    authorName = authorName.trim();
    text = text.trim();
  } else {
    gapStr = "+30s";
    authorName = "unknown";
    text = line;
  }

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
