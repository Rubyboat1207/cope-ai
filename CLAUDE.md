# Cope AI v3

Finetunes an LLM on the user's own Discord group chat ("cope") to predict the
next message — who sends it, how long after the previous one, and what it
says — then serves that model as a **fully static** browser-based chat
simulator: no backend, no inference server, no ongoing hosting cost. The
model runs client-side via WebGPU (`wllama`), fetched directly from a public
Hugging Face model repo (https://huggingface.co/Rubyboat/cope-ai-v3). The
whole `frontend/` directory can be dropped onto GitHub Pages (or any static
host) as-is.

Public repo: https://github.com/Rubyboat1207/cope-ai — see **Repo & privacy**
below for exactly what is and isn't in it.

## Layout

```
pipeline/     one-time data-prep + training/merge scripts (below)
frontend/     the entire static site — HTML/CSS/JS + authors.json + avatars/.
              deployable as-is (GitHub Pages, any static host, or just open
              index.html locally / `python -m http.server` from inside it).
llama.cpp/    vendored checkout, used only for convert_hf_to_gguf.py + llama-quantize/llama-server
logs/         all *.log output (gitignored)
models/       base model + merged models (gitignored, large)
lora_out/     LoRA training checkpoints (gitignored, large)
gguf_out/     converted/quantized GGUF files (gitignored, large)
messages.jsonl, finetune_dataset.jsonl, me.json
              your group's raw chat data (gitignored — private, never published)
dashboard.py / dashboard.sh   control-panel TUI covering everything below except step 1
monitor_tui.py / monitor.sh   lighter-weight standalone training monitor (subset of dashboard.py's Dashboard tab)
```

There is intentionally no `server.py` — an earlier iteration had a FastAPI
dev server, but once inference moved fully client-side (via `wllama`) and
the date-range history-browsing feature was removed, there was nothing left
that a static file server couldn't do, so it was deleted.

All `pipeline/` scripts use plain relative paths (`Path("messages.jsonl")`
etc.), not `__file__`-relative ones — they must be run with the repo root as
the working directory, e.g. `python pipeline/build_dataset.py` from here,
not `cd pipeline && python build_dataset.py`. `dashboard.py` always invokes
them this way (`cwd=ROOT`).

## Pipeline (run in this order)

1. **`pipeline/download.py`** — logs into Discord as the user (via
   `discord-py-self` and a token in `.env`) and pulls the full message
   history of a channel into `messages.jsonl`. Resumable (skips
   already-downloaded IDs), supports `--limit N` for test runs.
2. **`pipeline/fetch_avatars.py`** — logs in again, downloads every unique
   author's profile picture into `frontend/avatars/{author_id}.png` (this
   lives under `frontend/` because it's a static site asset, not private
   data — see **Repo & privacy**), and writes `me.json` (the logged-in
   user's own id/name).
3. **`pipeline/build_authors.py`** — derives `frontend/authors.json` (id →
   `{name, avatar}`) from `messages.jsonl` + `frontend/avatars/`. Rerun after
   re-downloading history or fetching new avatars.
4. **`pipeline/build_dataset.py`** — turns `messages.jsonl` into
   sliding-window next-message-prediction training examples
   (`finetune_dataset.jsonl`). Windows snap to whole-message boundaries
   (never split mid-message) and are capped at ~1024 tokens using the real
   tokenizer. Every message (including one nobody replied to) is a potential
   prediction target, with its real timestamp gap included — no explicit
   "conversation" segmentation, the model learns gap-vs-continuation
   correlation directly.
5. **`pipeline/train_lora.py`** — LoRA finetunes `models/Qwen2.5-3B-Instruct`
   on that dataset. See **Training** below for the details that matter
   operationally. Steps 5 onward (finetune → merge → convert → quantize) all
   have a guided/point-and-click path through **`dashboard.py`** — see
   **The control dashboard** below; you don't need to run these by hand.
6. **`pipeline/merge_lora.py <checkpoint> --out <dir>`** — merges a LoRA
   checkpoint into the base weights, producing a standalone HF model
   directory.
7. **GGUF conversion** (via the vendored `llama.cpp/` checkout):
   ```
   python llama.cpp/convert_hf_to_gguf.py models/merged-X --outfile gguf_out/X-f16.gguf --outtype f16
   ./llama.cpp/build/bin/llama-quantize gguf_out/X-f16.gguf gguf_out/X-q4_k_m.gguf Q4_K_M
   ```
8. **`frontend/`** — the chat UI, already deployed and pointed at the live
   Hugging Face model. No backend step needed (see **Deployment**).

## Format

Every training example / generation prompt is a flat text format, one
message per line:

```
[+42s] wiry__: can you add Nicholas
[+3h] rubyboat: yeah adding now
<|next|>
[+12s] rubyboat: done
```

`[gap]` is bucketed (`+Ns` / `+Nm` / `+Nh` / `+Nd`), computed from real
consecutive-message timestamps. The model predicts the line after
`<|next|>` — author, gap, and text jointly. Parsed client-side by
`frontend/model.js`'s `LINE_RE` — keep it in sync if the format changes.

## Hardware / environment gotchas

This box has **2x AMD RX 9070/9070 XT (gfx1201, RDNA4)**, no NVIDIA, and
only **14GB system RAM** — both matter a lot:

- **PyTorch is a ROCm build** (`torch==2.9.1+rocm6.4`), not CUDA. Everything
  in this venv assumes that.
- **QLoRA (4-bit bitsandbytes) does not work here** — bitsandbytes' ROCm
  kernels aren't built for gfx1201 and crash on `quantize_4bit`. All training
  uses plain bf16 LoRA instead, which fits comfortably (3B model, ~6GB VRAM).
- **Multi-GPU DDP hangs indefinitely** — tried torchrun across both cards,
  RCCL process-group init deadlocks (no P2P/xGMI link between these
  consumer cards). Training is single-GPU only (pinned to `cuda:0` /
  physical GPU 0, the RX 9070 XT). The chat server is pinned to `cuda:1`
  (the RX 9070) so the two never contend for VRAM.
- **`torch.cuda.device_count()` reports a bogus extra device** — ROCm
  exposes the CPU/iGPU as an HSA agent alongside the real GPUs, which makes
  HF `Trainer` try to wrap the model in `DataParallel` across all of them
  and crash on NCCL. `pipeline/train_lora.py` hard-sets
  `HIP_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES` before importing torch to
  work around this — don't remove those lines.
- **System RAM is the real bottleneck**, not VRAM. Loading a 3B model for
  merging (`pipeline/merge_lora.py`) while training is also running pushes
  the system into heavy swap and can take several minutes just to
  `.to(cuda)`. This is expected, not a hang — check `free -h` before
  assuming something is broken.
- **Root partition (`/`) is chronically near-full** (~23GB free); everything
  large (models, datasets, venv, pip cache) lives under `/mnt/2tb_ssd`
  instead. `PIP_CACHE_DIR` is pinned there via `venv/bin/activate`.
- **Discord snowflake IDs exceed JS's safe-integer range** — this bit us
  when the old dev server sent `author_id`/message `id` as JSON *numbers*,
  which the browser's `JSON.parse` silently corrupted. Not an active
  concern anymore (the static site only ever handles these as object keys/
  strings, e.g. in `authors.json`), but if you ever add anything that
  serializes a raw snowflake as a JSON number for the frontend to consume,
  stringify it first.

## Training

- Effective batch size 16 (`per_device=2 × grad_accum=8`), lr `2e-4`, LoRA
  `r=16 alpha=32` on all attention + MLP projections, gradient checkpointing
  on. ~10s/step on the RX 9070 XT; a full 3-epoch run over the full dataset
  is on the order of 2+ days, hence everything below.
- **Checkpoints every 500 steps** (`save_steps=500`, `save_total_limit=5`),
  not the HF default of once-per-epoch (~18h) — this was deliberately
  tightened so the chat server always has something recent to load.
- **Graceful shutdown**: SIGTERM/SIGINT are caught (`GracefulShutdown` +
  `SaveOnShutdownCallback`) and trigger a real checkpoint-and-stop instead of
  losing the in-flight step. Always stop training with `kill -TERM <pid>`,
  never `-9`, unless you're fine losing up to 500 steps.
- **Crash safety**: an uncaught exception during `trainer.train()` triggers
  a best-effort emergency save to `lora_out/crashed/`.
- **Auto-resume**: `train_lora.py` looks for the latest `lora_out/checkpoint-*`
  on startup and resumes from it automatically — just rerun the script after
  any stop.
- Loss logging needed a `LossPrinterCallback` because HF's default console
  logger doesn't reliably print in this transformers version — don't remove
  it or `monitor_tui.py`'s loss parsing goes dark.

## The control dashboard

`./dashboard.sh` (wraps `dashboard.py`, a Textual TUI) is the point-and-click
way to do everything from step 5 onward: it has tabs for **Dashboard**
(status overview — training progress, GPU/disk, checkpoint/model/gguf
listings), **Train** (start/stop with a live log — stop is graceful,
identical to sending `train_lora.py` a SIGTERM by hand), **Merge** (pick any
checkpoint, merge it), **Quantize** (pick any merged model + quant level,
convert + quantize), **Guided** (one button: latest checkpoint → merge →
convert → quantize to whatever level you pick, skipping steps whose output
already exists), and **Tutorial** (plain-language explainer of the whole
pipeline, written for a non-ML-background reader).

It shells out to the real `pipeline/` scripts and the `llama.cpp/` binaries
directly (`asyncio.create_subprocess_exec`, no intermediate shell) rather
than reimplementing their logic, so behavior always matches running them by
hand — including `train_lora.py`'s own SIGTERM handling, which the dashboard
relies on for its "Stop (graceful)" button.

Guided/Quantize-tab output naming: `gguf_out/{merged_model_name}-{f16,or,quant_level}.gguf`,
derived from the merged model directory's name with `merged-` stripped. The
very first GGUF export (done manually, before the dashboard existed) used a
different fixed naming scheme — `gguf_out/model-f16.gguf` /
`gguf_out/model-q4_k_m.gguf` — and the latter is the one actually uploaded
to Hugging Face and referenced by `frontend/model.js`'s `MODEL_URL` right
now. If you quantize a newer checkpoint and want to ship it, upload the new
file to the HF repo and update `MODEL_URL` (see **Deployment**) — don't
just delete/overwrite the old local file without doing both.

## Monitoring

`./monitor.sh` (wraps `monitor_tui.py`) is a lighter standalone alternative
to the dashboard's Dashboard tab — just training progress/ETA, recent loss,
per-GPU VRAM/util, disk usage, and checkpoints, nothing interactive. Useful
when you don't need the merge/quantize/train controls. Both parse
`logs/train.log`'s tqdm output directly (handling the `\r`
carriage-return-separated format), so they work without touching the
trainer — note this only picks up output from a manually-run,
manually-redirected `train_lora.py` (`... > logs/train.log 2>&1`); training
started from the dashboard's Train tab streams straight into its own log
widget instead and never touches `logs/train.log`.

## The chat app (`frontend/`, fully static)

- **Inference**: `frontend/model.js` loads `wllama` from a CDN (jsdelivr —
  **use the direct `esm/index.js` and `esm/wasm/wllama.wasm` paths, not the
  `+esm` shortcut or the bare `wasm/wllama.wasm` path from wllama's own
  README** — both 404/503 for this package version; verified working paths
  are already wired up in `model.js`), detects WebGPU via
  `isSupportWebGPU()`, and runs `createCompletion()` (raw completion, *not*
  `createChatCompletion` — the model was trained on a custom flat format,
  not a chat template) entirely in-browser. `MODEL_URL` points at the live
  HF resolve URL — there is no local/dev fallback path anymore, it's the
  same URL used everywhere.
- `wllama` v3+ removed the `tokenize`/`detokenize` API, so there's no exact
  client-side token counting. Context trimming in `model.js` is a
  message-count heuristic (`MAX_CONTEXT_MESSAGES = 40`), not an exact
  token-budget match to training — good enough in practice, not identical
  to `build_dataset.py`'s trimming.
- Frontend starts with an **empty chat** — you seed it by typing a message
  as one of the personas in the "Chat as" dropdown (which can be *any*
  known author, not just the account owner), then generating from there.
  There used to be a date-range history browser backed by a dev server;
  it was removed along with `server.py`.
- A **browser/device compatibility guide** (`frontend/compat.js`) detects
  OS + browser via `navigator.userAgent` and shows per-platform WebGPU
  setup instructions (Windows/macOS/Linux/Android/iOS) — accessible via the
  sidebar link, and auto-opens if `navigator.gpu.requestAdapter()` fails.
- The device-status pill (`GPU · LOCAL` / `CPU · LOCAL`) in the top bar
  reflects real `wllama` state (loading %, ready, generating) — it's the
  whole point of this build (zero server cost), keep it honest if you touch
  the loading code.

## Deployment

Already done — this is the reference for redoing it (e.g. after further
training):

1. `pipeline/merge_lora.py` the chosen checkpoint, convert + quantize to GGUF
   (or just use the dashboard's **Guided** tab for this whole step) —
   `q4_k_m` is what's live (~1.8GB for the 3B model, good quality/size
   tradeoff; regular `git`/GitHub rejects files this large outright, and
   Git LFS's free tier is too small for repeated downloads).
2. Upload the GGUF to the **public Hugging Face model repo**
   (https://huggingface.co/Rubyboat/cope-ai-v3), e.g.:
   ```
   hf upload Rubyboat/cope-ai-v3 gguf_out/<file>.gguf <file>.gguf
   ```
3. Update `MODEL_URL` in `frontend/model.js` to the new file's
   `resolve/main/...` URL if the filename changed.
4. `frontend/` is the entire deployable site — point GitHub Pages (or any
   static host) at it directly. No backend, no build step.

## Repo & privacy

Public repo: https://github.com/Rubyboat1207/cope-ai. What's in it vs. not,
and why:

- **Published (intentionally, with explicit confirmation)**: all code,
  `frontend/authors.json` and `frontend/avatars/*.png` — real Discord
  usernames and profile pictures of the group's members. These are static
  site assets the deployed chat app needs to render at all, and publishing
  them was a deliberate choice, not an oversight.
- **Never published**: `messages.jsonl` / `finetune_dataset.jsonl` (the
  full raw chat history/training data) and `me.json` — these stay local
  only, excluded via `.gitignore`, along with `.env` (the Discord token),
  `models/`/`lora_out/`/`gguf_out/`/`llama.cpp/` (large, regenerable), and
  `logs/`.
- If you ever change this tradeoff (e.g. decide the avatars/usernames
  shouldn't be public after all), you'd need to both remove them from
  `frontend/` **and** scrub git history (`git filter-repo` or similar) —
  deleting the files in a new commit alone leaves them recoverable from
  earlier commits in a public repo.

## Known rough edges

- `llama-cli`'s newer versions always apply the model's chat template to
  `-p` prompts (even with `--no-conversation`) — there's no raw-completion
  mode via the CLI anymore. Use `llama-server`'s `/completion` HTTP endpoint
  to test raw-format generation from the command line instead.
- The vendored `llama.cpp/` checkout was built CPU-only
  (`-DGGML_NATIVE=OFF`, no GPU backend) — it exists purely for
  `convert_hf_to_gguf.py` and `llama-quantize`/`llama-server`, not for
  serving production inference.
