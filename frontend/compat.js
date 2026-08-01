// Detects OS/browser and renders per-platform guidance for getting WebGPU
// working, since that's what makes in-browser inference possible at all.

const PLATFORMS = [
  {
    id: "windows",
    label: "Windows",
    match: (ua) => /Windows/.test(ua),
    html: `
      <p><b>Chrome or Edge</b> (recommended) — WebGPU works out of the box
      on recent versions. If it still doesn't work:</p>
      <ol>
        <li>Go to <code>chrome://flags</code> (or <code>edge://flags</code>)</li>
        <li>Search for <b>"Unsafe WebGPU Support"</b> and set it to Enabled</li>
        <li>Relaunch the browser</li>
      </ol>
      <p><b>Firefox</b> — WebGPU support is still experimental. If Chrome/Edge
      aren't an option, try <code>about:config</code> → set
      <code>gfx.webgpu.enabled</code> to <code>true</code>, but expect it to
      be less reliable.</p>`,
  },
  {
    id: "macos",
    label: "macOS",
    match: (ua) => /Macintosh|Mac OS X/.test(ua) && !/Mobile/.test(ua),
    html: `
      <p><b>Chrome or Edge</b> (recommended) — WebGPU works out of the box.</p>
      <p><b>Safari</b> (17.4+) — needs a feature flag enabled:</p>
      <ol>
        <li>Safari menu → Settings → Advanced → turn on
        "Show features for web developers"</li>
        <li>A new "Feature Flags" tab appears in Settings — open it</li>
        <li>Find <b>WebGPU</b> and turn it on</li>
        <li>Reload this page</li>
      </ol>`,
  },
  {
    id: "linux",
    label: "Linux",
    match: (ua) => /Linux/.test(ua) && !/Android/.test(ua),
    html: `
      <p><b>Chrome or Chromium</b> — often needs flags enabled manually,
      especially with AMD GPUs:</p>
      <ol>
        <li>Go to <code>chrome://flags</code></li>
        <li>Search <b>"Vulkan"</b> — set to Enabled</li>
        <li>Search <b>"Unsafe WebGPU Support"</b> (may also show as "WebGPU
        Developer Features") — set to Enabled</li>
        <li>Relaunch the browser</li>
      </ol>
      <p><b>Firefox</b> — WebGPU support is experimental on Linux and
      frequently doesn't detect the GPU correctly; Chrome/Chromium is the
      more reliable option here.</p>`,
  },
  {
    id: "android",
    label: "Android",
    match: (ua) => /Android/.test(ua),
    html: `
      <p><b>Chrome</b> (recommended) — supported by default on most
      reasonably recent devices/Chrome versions. If it's not working:</p>
      <ol>
        <li>Open <code>chrome://flags</code> in the Chrome address bar</li>
        <li>Search <b>"WebGPU"</b> and set it to Enabled</li>
        <li>Restart Chrome</li>
      </ol>
      <p>Other Android browsers (including Firefox for Android) generally
      don't support WebGPU yet — use Chrome.</p>`,
  },
  {
    id: "ios",
    label: "iOS / iPadOS",
    match: (ua) => /iPhone|iPad|iPod/.test(ua),
    html: `
      <p>All browsers on iOS (Chrome, Firefox, etc.) use Apple's WebKit
      engine under the hood, so this is really a <b>Safari/iOS setting</b>,
      not a per-browser one:</p>
      <ol>
        <li>Update to iOS/iPadOS 17.4 or later if you can</li>
        <li>Open the <b>Settings</b> app → scroll to <b>Apps</b> → <b>Safari</b>
        → <b>Advanced</b> → <b>Feature Flags</b></li>
        <li>Find <b>WebGPU</b> and turn it on</li>
        <li>Reload this page (in any browser)</li>
      </ol>
      <p>On older iOS versions WebGPU may not be available at all yet.</p>`,
  },
];

function detectPlatformId() {
  const ua = navigator.userAgent;
  const match = PLATFORMS.find((p) => p.match(ua));
  return match ? match.id : null;
}

export function renderCompatSections() {
  const container = document.getElementById("compat-sections");
  const detected = detectPlatformId();
  container.innerHTML = "";

  for (const platform of PLATFORMS) {
    const section = document.createElement("div");
    section.className = "compat-section" + (platform.id === detected ? " detected" : "");

    const title = document.createElement("div");
    title.className = "compat-section-title";
    title.textContent = platform.label;
    if (platform.id === detected) {
      const badge = document.createElement("span");
      badge.className = "compat-badge";
      badge.textContent = "your device";
      title.appendChild(badge);
    }

    const body = document.createElement("div");
    body.className = "compat-section-body";
    body.innerHTML = platform.html;

    section.appendChild(title);
    section.appendChild(body);
    container.appendChild(section);
  }

  return detected;
}

export async function checkWebGPUSupport() {
  if (!("gpu" in navigator)) return false;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    return !!adapter;
  } catch (e) {
    return false;
  }
}

export function initCompatGuide() {
  const overlay = document.getElementById("compat-overlay");
  const openBtn = document.getElementById("compat-btn");
  const closeBtn = document.getElementById("compat-close-btn");

  renderCompatSections();

  function open() {
    overlay.classList.remove("hidden");
    const detected = document.querySelector(".compat-section.detected");
    detected?.scrollIntoView({ block: "nearest" });
  }
  function close() {
    overlay.classList.add("hidden");
  }

  openBtn.addEventListener("click", open);
  closeBtn.addEventListener("click", close);
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) close();
  });

  return { open, close };
}
