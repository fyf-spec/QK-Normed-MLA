#!/usr/bin/env python3
"""Render the XTuner QK-Normed MLA handoff figures from accepted results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks" / "qknorm" / "results"
OUTPUT = ROOT / "docs" / "assets" / "qknorm-mla-decode"
FONT_ROOT = Path(
    "/mnt/shared-storage-gpfs2/plm-gpfs/yffeng/.codex/skills/"
    "lumia-plot-style/assets/fonts/xcharter"
)

TEAL = "#007C83"
BLACK = "#111827"
RED = "#C23B55"
BLUE = "#3B5CCC"
GRID = "#E8EDF2"


def _configure_style() -> None:
    family = "serif"
    roman = FONT_ROOT / "XCharter-Roman.otf"
    bold = FONT_ROOT / "XCharter-Bold.otf"
    if roman.exists() and bold.exists():
        font_manager.fontManager.addfont(roman)
        font_manager.fontManager.addfont(bold)
        family = "XCharter"
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 9.5,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "stix",
        }
    )


def _finish(fig: plt.Figure, ax: plt.Axes, stem: str) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3, width=0.6)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.45, alpha=0.6)
    ax.xaxis.grid(False)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(
            OUTPUT / f"{stem}.{suffix}",
            dpi=240,
            bbox_inches="tight",
            pad_inches=0.03,
        )
    plt.close(fig)


def plot_frontend_fusion() -> None:
    # Accepted Qwen/Megatron B=16, one-MLA-layer microbenchmark.
    labels = ["Unfused", "Fused Triton"]
    values = [145.812, 31.661]
    fig, ax = plt.subplots(figsize=(5.8, 4.15))
    bars = ax.bar(labels, values, width=0.56, color=[BLACK, TEAL])
    ax.set_ylabel("Latency per MLA layer (μs)")
    ax.set_ylim(0, 165)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 4, f"{value:.3f}", ha="center")
    ax.text(0.5, 155, "4.61× faster", color="#273240", ha="center", fontweight="bold")
    _finish(fig, ax, "frontend-qknorm-rope-fusion")


def plot_kernel_evolution() -> None:
    # Accepted summaries in benchmarks/qknorm/RESULTS.md.
    labels = ["Naive T×24", "Rank-1 T×1", "+ prefetch/dedup"]
    values = [16.34, 3.90, 4.00]
    fig, ax = plt.subplots(figsize=(5.8, 4.15))
    bars = ax.bar(labels, values, width=0.58, color=[RED, TEAL, BLUE])
    ax.set_ylabel("Mean overhead vs. FlashMLA core (%)")
    ax.set_ylim(0, 18.5)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.45, f"{value:.2f}%", ha="center")
    ax.text(0.5, 17.4, "4.2× lower overhead", color="#273240", ha="center", fontweight="bold")
    _finish(fig, ax, "attention-core-optimization")


def plot_full_layer_latency() -> None:
    data = json.loads((RESULTS / "fulllayer_manifest.json").read_text())
    rows = data["rows"]
    contexts = [row["ctx"] for row in rows]
    series = [
        ("torch.compile", [row["torch_ms"] for row in rows], BLACK, "o"),
        ("FlashMLA", [row["flash_ms"] for row in rows], BLUE, "s"),
        ("QK-Norm FlashMLA", [row["qk_ms"] for row in rows], TEAL, "^"),
    ]
    fig, ax = plt.subplots(figsize=(5.8, 4.15))
    for label, values, color, marker in series:
        ax.plot(contexts, values, label=label, color=color, marker=marker, linewidth=0.9, markersize=3.8)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Context length")
    ax.set_ylabel("Full MLA layer latency (ms)")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value / 1024)}K"))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False, fontsize=9.5)
    _finish(fig, ax, "full-layer-latency")


def main() -> None:
    _configure_style()
    plot_frontend_fusion()
    plot_kernel_evolution()
    plot_full_layer_latency()


if __name__ == "__main__":
    main()
