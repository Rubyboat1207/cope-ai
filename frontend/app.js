import { loadModel, generateNextMessage, isModelReady, MODEL_OPTIONS, getSelectedModelId, setSelectedModelId } from "./model.js";
import { initCompatGuide } from "./compat.js";

const messagesEl = document.getElementById("messages");
const loadBannerEl = document.getElementById("load-banner");
const errorBannerEl = document.getElementById("error-banner");
const personaSelect = document.getElementById("persona-select");
const respondAsSelect = document.getElementById("respond-as-select");
const autoContinueEl = document.getElementById("auto-continue");
const continueBtn = document.getElementById("continue-btn");
const textInput = document.getElementById("text-input");
const sendBtn = document.getElementById("send-btn");

const DEFAULT_AVATAR =
  "data:image/svg+xml;utf8," +
  encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40"><rect width="40" height="40" rx="20" fill="#5865f2"/></svg>'
  );

const clearBtn = document.getElementById("clear-btn");
const modelSelect = document.getElementById("model-select");

modelSelect.innerHTML = "";
MODEL_OPTIONS.forEach((m) => {
  const opt = document.createElement("option");
  opt.value = m.id;
  opt.textContent = m.label;
  modelSelect.appendChild(opt);
});
modelSelect.value = getSelectedModelId();
modelSelect.addEventListener("change", () => {
  setSelectedModelId(modelSelect.value);
  location.reload();
});

const sidebarEl = document.getElementById("sidebar");
const sidebarScrimEl = document.getElementById("sidebar-scrim");
const hamburgerBtn = document.getElementById("hamburger-btn");
const sidebarCloseBtn = document.getElementById("sidebar-close-btn");
const devicePillEl = document.getElementById("device-pill");
const devicePillTextEl = document.getElementById("device-pill-text");

function openSidebar() {
  sidebarEl.classList.add("open");
  sidebarScrimEl.classList.add("visible");
}
function closeSidebar() {
  sidebarEl.classList.remove("open");
  sidebarScrimEl.classList.remove("visible");
}
hamburgerBtn.addEventListener("click", openSidebar);
sidebarCloseBtn.addEventListener("click", closeSidebar);
sidebarScrimEl.addEventListener("click", closeSidebar);

function setDevicePill(text, { busy = false, offline = false } = {}) {
  devicePillTextEl.textContent = text;
  devicePillEl.classList.toggle("busy", busy);
  devicePillEl.classList.toggle("offline", offline);
}

function showError(message) {
  errorBannerEl.textContent = message;
  errorBannerEl.classList.remove("hidden");
}
function hideError() {
  errorBannerEl.classList.add("hidden");
}
errorBannerEl.addEventListener("click", hideError);

window.addEventListener("unhandledrejection", (event) => {
  showError(`Unexpected error: ${event.reason?.message || event.reason}`);
});
window.addEventListener("error", (event) => {
  showError(`Unexpected error: ${event.message}`);
});

// Made-up placeholder conversation shown on first load, just so there's
// something in the chat before you generate/type anything -- not real
// messages from anyone.
const DEFAULT_CHAT = [
  { author_id: "sample-alex", author_name: "Alex", avatar: null, text: "yo is anyone free this weekend", timestamp: "2026-01-01T18:00:00.000Z" },
  { author_id: "sample-jordan", author_name: "Jordan", avatar: null, text: "depends, what's the plan", timestamp: "2026-01-01T18:02:15.000Z" },
  { author_id: "sample-sam", author_name: "Sam", avatar: null, text: "movie night? we haven't hung out in forever", timestamp: "2026-01-01T18:05:40.000Z" },
  { author_id: "sample-alex", author_name: "Alex", avatar: null, text: "bet, i'm down", timestamp: "2026-01-01T18:06:02.000Z" },
];

let authors = {};
let history = DEFAULT_CHAT.slice(); // {author_id, author_name, avatar, text, timestamp}

async function loadAuthors() {
  const res = await fetch("authors.json");
  authors = await res.json();
  personaSelect.innerHTML = "";
  Object.entries(authors)
    .sort((a, b) => a[1].name.localeCompare(b[1].name))
    .forEach(([id, info]) => {
      const opt = document.createElement("option");
      opt.value = id;
      opt.textContent = info.name;
      personaSelect.appendChild(opt);

      const respondOpt = document.createElement("option");
      respondOpt.value = info.name;
      respondOpt.textContent = info.name;
      respondAsSelect.appendChild(respondOpt);
    });
}

function clearChat() {
  history = [];
  renderAll();
}

function formatTime(iso) {
  const d = new Date(iso);
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function renderAll() {
  messagesEl.innerHTML = "";
  history.forEach((m) => appendMessageEl(m));
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function appendMessageEl(m) {
  const row = document.createElement("div");
  row.className = "message-row";

  const img = document.createElement("img");
  img.className = "avatar";
  img.src = m.avatar || DEFAULT_AVATAR;
  img.onerror = () => (img.src = DEFAULT_AVATAR);

  const body = document.createElement("div");
  body.className = "msg-body";

  const header = document.createElement("div");
  header.className = "msg-header";
  const name = document.createElement("span");
  name.className = "author-name";
  name.textContent = m.author_name;
  const ts = document.createElement("span");
  ts.className = "timestamp";
  ts.textContent = formatTime(m.timestamp);
  header.appendChild(name);
  header.appendChild(ts);

  const text = document.createElement("div");
  text.className = "msg-text";
  text.textContent = m.text;

  body.appendChild(header);
  body.appendChild(text);
  row.appendChild(img);
  row.appendChild(body);
  messagesEl.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function showTyping(text) {
  const row = document.createElement("div");
  row.className = "typing-row";
  row.textContent = text;
  messagesEl.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return row;
}

async function generateNext() {
  if (!isModelReady()) {
    const row = showTyping("model still loading, hang on");
    setTimeout(() => row.remove(), 2000);
    return;
  }
  const contextMessages = history.slice(-60).map((m) => ({
    author_id: m.author_id,
    author_name: m.author_name,
    text: m.text,
    timestamp: m.timestamp,
  }));
  const forcedAuthorName = respondAsSelect.value || null;
  hideError();
  devicePillEl.classList.add("busy");

  const typingRow = showTyping("thinking...");
  let result;
  try {
    result = await generateNextMessage(contextMessages, authors, forcedAuthorName, ({ status, partialText }) => {
      typingRow.textContent = partialText ? `${status} ${partialText}` : status;
      messagesEl.scrollTop = messagesEl.scrollHeight;
    });
  } catch (e) {
    typingRow.remove();
    showError(`Generation failed: ${e.message}`);
    autoContinueEl.checked = false;
    return;
  } finally {
    devicePillEl.classList.remove("busy");
  }

  typingRow.remove();

  const msg = {
    author_id: result.author_id || "unknown",
    author_name: result.author_name,
    avatar: result.avatar,
    text: result.text,
    timestamp: result.timestamp,
  };
  history.push(msg);
  appendMessageEl(msg);
  respondAsSelect.value = "";
  return msg;
}

async function autoContinueLoop() {
  while (autoContinueEl.checked) {
    await generateNext();
    await new Promise((r) => setTimeout(r, 500));
  }
}

continueBtn.addEventListener("click", () => generateNext());

autoContinueEl.addEventListener("change", () => {
  if (autoContinueEl.checked) autoContinueLoop();
});

function sendAsPersona() {
  const text = textInput.value.trim();
  if (!text) return;
  const authorId = personaSelect.value;
  const info = authors[authorId] || { name: "unknown", avatar: null };

  const msg = {
    author_id: authorId,
    author_name: info.name,
    avatar: info.avatar,
    text,
    timestamp: new Date().toISOString(),
  };
  history.push(msg);
  appendMessageEl(msg);
  textInput.value = "";
}

sendBtn.addEventListener("click", sendAsPersona);
textInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendAsPersona();
});

async function initModel() {
  loadBannerEl.textContent = "Loading model in your browser (first load downloads 1-2GB, cached after)...";
  loadBannerEl.classList.remove("hidden", "error");
  setDevicePill("connecting");
  let usedGpu = true;
  try {
    await loadModel(({ stage, pct }) => {
      loadBannerEl.textContent = `${stage}: ${pct}%`;
      usedGpu = stage.includes("GPU");
      setDevicePill(`${usedGpu ? "GPU" : "CPU"} · ${pct}%`);
    });
    loadBannerEl.textContent = "Model ready — running entirely on your device.";
    setDevicePill(`${usedGpu ? "GPU" : "CPU"} · local`);
    setTimeout(() => loadBannerEl.classList.add("hidden"), 4000);
  } catch (e) {
    loadBannerEl.textContent = `Model failed to load: ${e.message}`;
    loadBannerEl.classList.add("error");
    setDevicePill("offline", { offline: true });
  }
}

clearBtn.addEventListener("click", () => {
  clearChat();
  closeSidebar();
});

(async function init() {
  const compat = initCompatGuide();
  await loadAuthors();
  renderAll();
  initModel();

  // Always show the compatibility guide on load -- it's useful onboarding
  // (explains what this even is / that it needs WebGPU) regardless of
  // whether this specific browser happens to support it.
  compat.open();
})();
