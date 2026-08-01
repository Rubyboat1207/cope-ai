"""
Terminal dashboard for monitoring the LoRA training run: progress/ETA
parsed from train.log's tqdm output, plus live GPU and disk stats.
"""
import re
import subprocess
import time
from datetime import timedelta
from pathlib import Path

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

TRAIN_LOG = Path("logs/train.log")
LORA_DIR = Path("lora_out")

PROGRESS_RE = re.compile(
    r"(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[([\d:]+)<([\d:]+),\s*([\d.]+)s/it\]"
)
LOSS_RE = re.compile(r"\{'loss':\s*'?([\d.]+)'?.*?'epoch':\s*'?([\d.]+)'?\}")


def read_tail(path: Path, chunk_size: int = 65536) -> str:
    if not path.exists():
        return ""
    size = path.stat().st_size
    with path.open("rb") as f:
        f.seek(max(0, size - chunk_size))
        data = f.read()
    return data.decode("utf-8", errors="ignore")


def latest_progress(text: str):
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


def latest_losses(text: str, n: int = 8):
    matches = LOSS_RE.findall(text)
    return matches[-n:]


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


def disk_stats():
    out = subprocess.run(["df", "-h", "/", "/mnt/2tb_ssd"], capture_output=True, text=True).stdout
    return out.strip().splitlines()[1:]


def checkpoint_list():
    if not LORA_DIR.exists():
        return []
    return sorted(
        (p.name for p in LORA_DIR.glob("checkpoint-*")),
        key=lambda n: int(n.split("-")[1]),
    )


def pct_color(pct: float) -> str:
    if pct < 60:
        return "green"
    if pct < 85:
        return "yellow"
    return "red"


def build_dashboard():
    text = read_tail(TRAIN_LOG)
    progress = latest_progress(text)
    losses = latest_losses(text)

    progress_group = []
    if progress:
        step, total = progress["step"], progress["total"]
        pct = progress["pct"]

        bar = ProgressBar(total=total, completed=step, width=50, complete_style="bright_magenta", finished_style="bright_green")
        header = Text(f"  {pct}%  ", style="bold bright_magenta")
        header.append(f"({step:,} / {total:,} steps)", style="dim")

        info = Table.grid(padding=(0, 2))
        info.add_row(Text("Speed", style="bold cyan"), Text(f"{progress['s_per_it']:.2f}s/step", style="white"))
        info.add_row(Text("Elapsed", style="bold cyan"), Text(progress["elapsed"], style="white"))
        eta_seconds = (total - step) * progress["s_per_it"]
        eta = str(timedelta(seconds=int(eta_seconds)))
        info.add_row(Text("Remaining", style="bold cyan"), Text(f"{progress['remaining']}  (~{eta})", style="bright_yellow"))

        progress_group = [header, bar, Text(""), info]
    else:
        progress_group = [Text("no progress data yet — training may still be starting up", style="dim italic")]

    progress_panel = Panel(
        Group(*progress_group),
        title="[bold]Training Progress[/bold]",
        border_style="bright_magenta",
    )

    loss_table = Table(title="[bold]Recent loss[/bold]", expand=True, border_style="cyan", header_style="bold cyan")
    loss_table.add_column("Epoch")
    loss_table.add_column("Loss")
    if losses:
        loss_values = [float(l) for l, _ in losses]
        min_loss = min(loss_values)
        for loss_str, epoch in losses:
            style = "bold bright_green" if float(loss_str) <= min_loss else "white"
            loss_table.add_row(epoch, Text(loss_str, style=style))
    else:
        loss_table.add_row("-", Text("not logged for this run (see pipeline/train_lora.py note)", style="dim italic"))

    gpu = gpu_stats()
    gpu_table = Table(title="[bold]GPU[/bold]", expand=True, border_style="green", header_style="bold green")
    gpu_table.add_column("GPU")
    gpu_table.add_column("VRAM used")
    gpu_table.add_column("Util")
    gpu_table.add_column("Role")
    labels = {0: "RX 9070 XT (training)", 1: "RX 9070 (chat inference)"}
    for gid in sorted(gpu.keys()):
        s = gpu[gid]
        used_gb = s.get("used", 0) / 1e9
        total_gb = s.get("total", 0) / 1e9
        used_pct = (used_gb / total_gb * 100) if total_gb else 0
        util = s.get("util")

        vram_text = Text(f"{used_gb:.1f} / {total_gb:.1f} GB", style=pct_color(used_pct))
        util_text = Text(f"{util}%" if util is not None else "-", style=pct_color(util) if util is not None else "dim")

        gpu_table.add_row(
            Text(f"GPU {gid}", style="bold"),
            vram_text,
            util_text,
            labels.get(gid, "-"),
        )

    disk_table = Table(title="[bold]Disk[/bold]", expand=True, border_style="blue", header_style="bold blue")
    disk_table.add_column("Mount")
    disk_table.add_column("Used")
    disk_table.add_column("Avail")
    disk_table.add_column("Use%")
    for line in disk_stats():
        parts = line.split()
        if len(parts) >= 6:
            use_pct = int(parts[4].rstrip("%"))
            disk_table.add_row(parts[5], parts[2], parts[3], Text(parts[4], style=pct_color(use_pct)))

    checkpoints = checkpoint_list()
    ckpt_text = (
        Text(", ".join(checkpoints), style="bright_green")
        if checkpoints
        else Text("none saved yet", style="dim italic")
    )
    ckpt_panel = Panel(ckpt_text, title="[bold]Checkpoints (lora_out/)[/bold]", border_style="yellow")

    return Group(progress_panel, loss_table, gpu_table, disk_table, ckpt_panel)


def main():
    with Live(build_dashboard(), refresh_per_second=1, screen=True) as live:
        while True:
            time.sleep(2)
            live.update(build_dashboard())


if __name__ == "__main__":
    main()
