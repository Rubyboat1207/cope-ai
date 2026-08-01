"""
Cope AI v3 control dashboard — a Textual TUI covering every step of the
pipeline except scraping the group chat itself (see download.py/
fetch_avatars.py/build_authors.py/build_dataset.py for that, run those
first): finetuning, merging LoRA checkpoints, converting to GGUF,
quantizing, and a guided one-click flow for "take the latest checkpoint,
merge it, quantize it to whatever level I want."

Run with ./dashboard.sh (or: source venv/bin/activate && python dashboard.py)
"""
import asyncio
import re
import signal
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    Footer,
    Header,
    Label,
    Markdown,
    RichLog,
    Select,
    Static,
    TabbedContent,
    TabPane,
)

ROOT = Path(__file__).parent
LORA_DIR = ROOT / "lora_out"
MODELS_DIR = ROOT / "models"
GGUF_DIR = ROOT / "gguf_out"
LLAMA_CPP = ROOT / "llama.cpp"
PIPELINE_DIR = ROOT / "pipeline"
TRAIN_LOG = ROOT / "logs" / "train.log"

QUANT_LEVELS = [
    ("Q4_0 — smallest, lowest quality (~1.7GB for 3B)", "Q4_0"),
    ("Q4_K_M — recommended default, best size/quality tradeoff (~1.8GB)", "Q4_K_M"),
    ("Q5_K_M — noticeably better quality, bigger (~2.4GB)", "Q5_K_M"),
    ("Q6_K — close to full quality, bigger still (~2.8GB)", "Q6_K"),
    ("Q8_0 — near-lossless, largest practical option (~3.6GB)", "Q8_0"),
    ("F16 — no quantization, full precision (~6GB)", "F16"),
]

PROGRESS_RE = re.compile(
    r"(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[([\d:]+)<([\d:]+),\s*([\d.]+)s/it\]"
)


def checkpoints():
    if not LORA_DIR.exists():
        return []
    return sorted(
        LORA_DIR.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1]),
    )


def latest_checkpoint():
    ckpts = checkpoints()
    return ckpts[-1] if ckpts else None


def merged_models():
    if not MODELS_DIR.exists():
        return []
    return sorted(p for p in MODELS_DIR.glob("merged-*") if p.is_dir())


def gguf_files():
    if not GGUF_DIR.exists():
        return []
    return sorted(GGUF_DIR.glob("*.gguf"))


def human_size(path: Path) -> str:
    try:
        n = path.stat().st_size
    except FileNotFoundError:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024
    return f"{n:.1f}PB"


def read_tail(path: Path, chunk_size: int = 65536) -> str:
    if not path.exists():
        return ""
    size = path.stat().st_size
    with path.open("rb") as f:
        f.seek(max(0, size - chunk_size))
        data = f.read()
    return data.decode("utf-8", errors="ignore")


def latest_progress():
    text = read_tail(TRAIN_LOG)
    matches = PROGRESS_RE.findall(text)
    if not matches:
        return None
    pct, step, total, elapsed, remaining, s_per_it = matches[-1]
    return {
        "pct": int(pct),
        "step": int(step),
        "total": int(total),
        "elapsed": elapsed,
        "remaining": remaining,
        "s_per_it": float(s_per_it),
    }


def gpu_stats():
    try:
        out = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--showuse"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except Exception:
        return {}
    stats = {}
    for line in out.splitlines():
        m = re.match(r"GPU\[(\d+)\]\s*:\s*VRAM Total Memory \(B\): (\d+)", line)
        if m:
            stats.setdefault(int(m.group(1)), {})["total"] = int(m.group(2))
            continue
        m = re.match(r"GPU\[(\d+)\]\s*:\s*VRAM Total Used Memory \(B\): (\d+)", line)
        if m:
            stats.setdefault(int(m.group(1)), {})["used"] = int(m.group(2))
            continue
        m = re.match(r"GPU\[(\d+)\]\s*:\s*GPU use \(%\): (\d+)", line)
        if m:
            stats.setdefault(int(m.group(1)), {})["util"] = int(m.group(2))
    return stats


async def stream_subprocess(cmd, cwd, log: RichLog, env=None):
    """Run a subprocess directly (no shell) and stream its output into a
    RichLog line by line. Returns the process return code."""
    process = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )
    while True:
        line = await process.stdout.readline()
        if not line:
            break
        text = line.decode("utf-8", errors="ignore").rstrip()
        for part in text.split("\r"):
            part = part.strip()
            if part:
                log.write(part)
    return await process.wait()


class DashboardTab(Vertical):
    def compose(self) -> ComposeResult:
        yield Static("", id="dash-status")
        yield Static("", id="dash-gpu")
        yield Static("", id="dash-files")

    def on_mount(self) -> None:
        self.refresh_status()
        self.set_interval(3, self.refresh_status)

    def refresh_status(self) -> None:
        progress = latest_progress()
        if progress:
            step, total = progress["step"], progress["total"]
            eta = timedelta(seconds=int((total - step) * progress["s_per_it"]))
            status = (
                f"[b]Training[/b] — step {step:,}/{total:,} ({progress['pct']}%), "
                f"{progress['s_per_it']:.2f}s/step, elapsed {progress['elapsed']}, "
                f"ETA ~{eta}"
            )
        else:
            status = "[b]Training[/b] — no progress data (not running, or just started)"
        self.query_one("#dash-status", Static).update(status)

        gpu = gpu_stats()
        gpu_lines = ["[b]GPU[/b]"]
        labels = {0: "training (RX 9070 XT)", 1: "chat inference (RX 9070)"}
        for gid in sorted(gpu.keys()):
            s = gpu[gid]
            used = s.get("used", 0) / 1e9
            total = s.get("total", 0) / 1e9
            util = s.get("util", "?")
            gpu_lines.append(
                f"  GPU {gid} ({labels.get(gid, '-')}): {used:.1f}/{total:.1f}GB, {util}% util"
            )
        self.query_one("#dash-gpu", Static).update("\n".join(gpu_lines))

        ckpts = checkpoints()
        merged = merged_models()
        ggufs = gguf_files()
        file_lines = ["[b]Checkpoints[/b] (lora_out/)"]
        file_lines.append("  " + (", ".join(p.name for p in ckpts[-5:]) or "none yet"))
        file_lines.append("[b]Merged models[/b] (models/)")
        file_lines.append("  " + (", ".join(p.name for p in merged) or "none yet"))
        file_lines.append("[b]GGUF files[/b] (gguf_out/)")
        file_lines.append(
            "  " + (", ".join(f"{p.name} ({human_size(p)})" for p in ggufs) or "none yet")
        )
        self.query_one("#dash-files", Static).update("\n".join(file_lines))


class TrainTab(Vertical):
    process: asyncio.subprocess.Process | None = None

    def compose(self) -> ComposeResult:
        with Horizontal(id="train-controls"):
            yield Button("Start training", id="train-start", variant="success")
            yield Button("Stop (graceful)", id="train-stop", variant="warning")
        yield Static("Not running", id="train-state")
        yield RichLog(id="train-log", wrap=True, highlight=False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "train-start":
            self.start_training()
        elif event.button.id == "train-stop":
            self.stop_training()

    @work(exclusive=True)
    async def start_training(self) -> None:
        if self.process is not None:
            self.query_one("#train-log", RichLog).write("[already running]")
            return
        log = self.query_one("#train-log", RichLog)
        log.clear()
        log.write("Starting pipeline/train_lora.py ...")
        self.query_one("#train-state", Static).update("[b yellow]running[/b yellow]")

        self.process = await asyncio.create_subprocess_exec(
            sys.executable, str(PIPELINE_DIR / "train_lora.py"),
            cwd=str(ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        while True:
            line = await self.process.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="ignore").rstrip()
            for part in text.split("\r"):
                part = part.strip()
                if part:
                    log.write(part)
        code = await self.process.wait()
        log.write(f"[process exited, code {code}]")
        self.process = None
        self.query_one("#train-state", Static).update("[b]not running[/b]")

    def stop_training(self) -> None:
        log = self.query_one("#train-log", RichLog)
        if self.process is None:
            log.write("[nothing to stop]")
            return
        log.write("Sending SIGTERM (will checkpoint before stopping)...")
        self.process.send_signal(signal.SIGTERM)
        self.query_one("#train-state", Static).update("[b yellow]stopping...[/b yellow]")


class MergeTab(Vertical):
    def compose(self) -> ComposeResult:
        yield Label("Checkpoint to merge:")
        yield Select([], id="merge-select", allow_blank=True)
        yield Button("Refresh checkpoint list", id="merge-refresh")
        yield Button("Merge selected checkpoint", id="merge-run", variant="success")
        yield RichLog(id="merge-log", wrap=True, highlight=False)

    def on_mount(self) -> None:
        self.refresh_checkpoints()

    def refresh_checkpoints(self) -> None:
        options = [(p.name, str(p)) for p in reversed(checkpoints())]
        select = self.query_one("#merge-select", Select)
        select.set_options(options)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "merge-refresh":
            self.refresh_checkpoints()
        elif event.button.id == "merge-run":
            self.run_merge()

    @work(exclusive=True)
    async def run_merge(self) -> None:
        log = self.query_one("#merge-log", RichLog)
        select = self.query_one("#merge-select", Select)
        if select.value is Select.BLANK or select.value is None:
            log.write("[pick a checkpoint first]")
            return

        ckpt_path = Path(select.value)
        out_dir = MODELS_DIR / f"merged-{ckpt_path.name}"
        log.clear()
        log.write(f"Merging {ckpt_path.name} -> {out_dir} ...")

        code = await stream_subprocess(
            [sys.executable, str(PIPELINE_DIR / "merge_lora.py"), str(ckpt_path), "--out", str(out_dir)],
            ROOT, log,
        )
        log.write(f"[done, exit code {code}]" if code == 0 else f"[FAILED, exit code {code}]")


QUANT_HELP = {
    "Q4_0": "Simplest 4-bit scheme, smallest file, most quality loss.",
    "Q4_K_M": "Mixed 4/6-bit ('K-quant'), the standard recommended default.",
    "Q5_K_M": "Mixed 5/6-bit, clearly better than Q4_K_M, ~30% bigger.",
    "Q6_K": "Mostly 6-bit, very close to full precision.",
    "Q8_0": "8-bit, essentially lossless, still much smaller than F16.",
    "F16": "No quantization at all — the raw converted model.",
}


class QuantizeTab(Vertical):
    def compose(self) -> ComposeResult:
        yield Label("Merged model to convert:")
        yield Select([], id="quant-model-select", allow_blank=True)
        yield Label("Quantization level:")
        yield Select(QUANT_LEVELS, id="quant-level-select", value="Q4_K_M")
        yield Static("", id="quant-help")
        yield Horizontal(
            Button("Refresh model list", id="quant-refresh"),
            Button("Convert & quantize", id="quant-run", variant="success"),
        )
        yield RichLog(id="quant-log", wrap=True, highlight=False)

    def on_mount(self) -> None:
        self.refresh_models()
        self.update_help()

    def refresh_models(self) -> None:
        options = [(p.name, str(p)) for p in merged_models()]
        self.query_one("#quant-model-select", Select).set_options(options)

    def update_help(self) -> None:
        level = self.query_one("#quant-level-select", Select).value
        self.query_one("#quant-help", Static).update(QUANT_HELP.get(level, ""))

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "quant-level-select":
            self.update_help()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "quant-refresh":
            self.refresh_models()
        elif event.button.id == "quant-run":
            self.run_quantize()

    @work(exclusive=True)
    async def run_quantize(self) -> None:
        log = self.query_one("#quant-log", RichLog)
        model_select = self.query_one("#quant-model-select", Select)
        level_select = self.query_one("#quant-level-select", Select)

        if model_select.value is Select.BLANK or model_select.value is None:
            log.write("[pick a merged model first]")
            return

        model_dir = Path(model_select.value)
        level = level_select.value
        log.clear()

        await convert_and_quantize(model_dir, level, log)


async def convert_and_quantize(model_dir: Path, level: str, log: RichLog) -> Path | None:
    """Shared logic used by both the Quantize tab and the Guided tab.
    Skips steps whose output already exists. Returns the final gguf path."""
    GGUF_DIR.mkdir(exist_ok=True)
    name = model_dir.name.removeprefix("merged-")
    f16_path = GGUF_DIR / f"{name}-f16.gguf"
    final_path = GGUF_DIR / f"{name}-{level.lower()}.gguf"

    if final_path.exists():
        log.write(f"[{final_path.name} already exists, skipping]")
        return final_path

    if f16_path.exists():
        log.write(f"[{f16_path.name} already exists, skipping conversion step]")
    else:
        log.write(f"Converting {model_dir.name} -> {f16_path.name} ...")
        code = await stream_subprocess(
            [
                sys.executable, str(LLAMA_CPP / "convert_hf_to_gguf.py"),
                str(model_dir), "--outfile", str(f16_path), "--outtype", "f16",
            ],
            ROOT, log,
        )
        if code != 0:
            log.write(f"[conversion FAILED, exit code {code}]")
            return None

    if level == "F16":
        log.write("[F16 selected — no quantization step needed]")
        return f16_path

    log.write(f"Quantizing -> {final_path.name} ...")
    quantize_bin = LLAMA_CPP / "build" / "bin" / "llama-quantize"
    code = await stream_subprocess(
        [str(quantize_bin), str(f16_path), str(final_path), level],
        ROOT, log,
    )
    if code != 0:
        log.write(f"[quantization FAILED, exit code {code}]")
        return None

    log.write(f"[done — {final_path.name} ({human_size(final_path)})]")
    return final_path


class GuidedTab(Vertical):
    def compose(self) -> ComposeResult:
        latest = latest_checkpoint()
        latest_text = f"Latest checkpoint: [b]{latest.name}[/b]" if latest else "[b red]no checkpoints found[/b red]"
        yield Static(latest_text, id="guided-latest")
        yield Label("Quantization level:")
        yield Select(QUANT_LEVELS, id="guided-level-select", value="Q4_K_M")
        yield Button("Run: merge + convert + quantize latest checkpoint", id="guided-run", variant="success")
        yield RichLog(id="guided-log", wrap=True, highlight=False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "guided-run":
            self.run_guided()

    @work(exclusive=True)
    async def run_guided(self) -> None:
        log = self.query_one("#guided-log", RichLog)
        log.clear()

        ckpt = latest_checkpoint()
        if ckpt is None:
            log.write("[no checkpoints in lora_out/ — train first]")
            return

        level = self.query_one("#guided-level-select", Select).value
        out_dir = MODELS_DIR / f"merged-{ckpt.name}"

        log.write(f"=== Step 1/3: merge {ckpt.name} ===")
        if out_dir.exists():
            log.write(f"[{out_dir.name} already exists, skipping merge]")
        else:
            code = await stream_subprocess(
                [sys.executable, str(PIPELINE_DIR / "merge_lora.py"), str(ckpt), "--out", str(out_dir)],
                ROOT, log,
            )
            if code != 0:
                log.write(f"[merge FAILED, exit code {code}]")
                return

        log.write("=== Step 2/3 & 3/3: convert + quantize ===")
        result = await convert_and_quantize(out_dir, level, log)
        if result:
            log.write(f"\n[b]All done.[/b] Final file: {result} ({human_size(result)})")
            log.write("Next: upload this file to your Hugging Face model repo, then update")
            log.write("MODEL_URL in frontend/model.js to point at it.")


TUTORIAL_TEXT = """\
# How this pipeline works

This dashboard covers everything **after** you already have `messages.jsonl`
(from `download.py`), `avatars/` + `authors.json` (from `fetch_avatars.py` +
`build_authors.py`), and `finetune_dataset.jsonl` (from `build_dataset.py`).
Those are one-time data-prep steps you run yourself first.

## 1. Finetuning (the **Train** tab)

The base model (Qwen2.5-3B-Instruct) is finetuned with **LoRA** — instead of
updating all 3 billion parameters, LoRA freezes the base model and trains a
small set of extra "adapter" weights (~30M parameters here) that get added on
top. Much faster and cheaper than full finetuning, and the result is a tiny
adapter file rather than a whole new copy of the model.

Training saves a **checkpoint** every 500 steps into `lora_out/checkpoint-N/`.
You can stop training at any time with the Stop button — it sends a
graceful signal that finishes the current step and saves a checkpoint before
exiting, so you never lose more than a few minutes of progress. Restarting
training automatically resumes from the latest checkpoint.

## 2. Merging (the **Merge** tab)

A LoRA checkpoint by itself isn't a complete model — it's just the small
adapter. **Merging** bakes the adapter's changes into the base model's actual
weights, producing one standalone model folder. This step is required before
converting to GGUF, since the conversion tool doesn't understand LoRA
adapters directly.

## 3. Converting to GGUF

GGUF is the file format used by `llama.cpp` (and, importantly, by `wllama`
— the library that runs the model directly in a web browser via WebGPU).
The merged model gets converted from HuggingFace's format into GGUF, still
at full precision (F16) at this stage.

## 4. Quantizing (the **Quantize** tab)

Quantization shrinks the model by storing its numbers less precisely.
The tradeoff is size/speed vs. quality:

| Level    | Size (3B model) | Quality                          |
|----------|-----------------|-----------------------------------|
| Q4_0     | ~1.7GB          | Smallest, most quality loss       |
| Q4_K_M   | ~1.8GB          | **Recommended default**           |
| Q5_K_M   | ~2.4GB          | Clearly better, bigger            |
| Q6_K     | ~2.8GB          | Very close to full quality        |
| Q8_0     | ~3.6GB          | Essentially lossless              |
| F16      | ~6GB            | No quantization                   |

For a site your friends will load in their browser, **Q4_K_M** is the
sensible default — the ~2GB download is the main cost your friends pay, and
this level rarely produces noticeably worse output than bigger ones for a
casual chat model.

## 5. The Guided flow (the **Guided** tab)

If you just want to take whatever the latest checkpoint is, merge it, and
quantize it to a level of your choosing — the Guided tab does exactly that
in one click. It automatically skips steps whose output already exists, so
re-running it after picking a different quantization level won't redo the
merge or the F16 conversion, just the final quantize step.

## After quantizing

The resulting `.gguf` file in `gguf_out/` is what you upload to a public
Hugging Face model repo, and point `MODEL_URL` in `frontend/model.js` at.
See `CLAUDE.md`'s Deployment section for the full picture.
"""


class TutorialTab(VerticalScroll):
    def compose(self) -> ComposeResult:
        yield Markdown(TUTORIAL_TEXT)


class CopeDashboard(App):
    CSS = """
    Screen {
        background: #15171c;
    }
    RichLog {
        height: 1fr;
        border: solid #272a33;
        background: #1b1e25;
    }
    Select {
        width: 100%;
    }
    #train-controls {
        height: auto;
    }
    """
    TITLE = "Cope AI v3 — control dashboard"
    BINDINGS = [("q", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("Dashboard", id="tab-dashboard"):
                yield DashboardTab()
            with TabPane("Train", id="tab-train"):
                yield TrainTab()
            with TabPane("Merge", id="tab-merge"):
                yield MergeTab()
            with TabPane("Quantize", id="tab-quantize"):
                yield QuantizeTab()
            with TabPane("Guided", id="tab-guided"):
                yield GuidedTab()
            with TabPane("Tutorial", id="tab-tutorial"):
                yield TutorialTab()
        yield Footer()


if __name__ == "__main__":
    CopeDashboard().run()
