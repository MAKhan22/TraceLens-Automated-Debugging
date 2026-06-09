#!/usr/bin/env python3
"""Generate v1/v2 architecture figures for the paper."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BOX = dict(boxstyle="round,pad=0.35,rounding_size=0.06", linewidth=1.0, edgecolor="#333333")
RED, BLUE, GREEN, GRAY, WARN = "#f8d7da", "#cfe2ff", "#d1e7dd", "#f0f0f0", "#fff3cd"


def box(ax, cx, cy, w, h, text, color=GRAY, fs=8.5):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, facecolor=color, **BOX))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, linespacing=1.2)
    return cy - h / 2, cy + h / 2


def arrow_v(ax, x, y_bottom, y_top):
    ax.add_patch(
        FancyArrowPatch(
            (x, y_bottom),
            (x, y_top),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.0,
            color="#444",
            shrinkA=0,
            shrinkB=0,
        )
    )


def fig_v2():
    fig, ax = plt.subplots(figsize=(6.5, 9.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis("off")
    ax.set_title("TraceLens v2: Unified Fusion Pipeline", fontsize=11, fontweight="bold", pad=10)

    h = 0.72
    cx = 5.0

    b_pass_bot, b_pass_top = box(ax, 3.0, 18.6, 2.0, h, "Pass trace", BLUE)
    b_fail_bot, b_fail_top = box(ax, 7.0, 18.6, 2.0, h, "Fail trace", BLUE)
    b_parse_bot, b_parse_top = box(ax, cx, 17.0, 4.2, h, "Parse and align traces", GRAY)
    b_sig_bot, b_sig_top = box(ax, cx, 15.4, 4.6, h, "Relative signal extraction", GRAY)
    b_chan_bot, b_chan_top = box(ax, cx, 13.8, 6.8, h, "Fusion channels\n(text, nav, causal, observer, pixel)", BLUE, fs=7.5)

    y_branch = 12.0
    b_llm_bot, b_llm_top = box(ax, 2.0, y_branch, 2.6, h, "LLM rerank\n(optional)", GRAY, fs=7.5)
    b_vc_bot, b_vc_top = box(ax, cx, y_branch, 2.6, h, "Visual-causal\nscan", GRAY, fs=7.5)
    b_vlm_bot, b_vlm_top = box(ax, 8.0, y_branch, 2.6, h, "VLM pairs\n(optional)", GRAY, fs=7.5)

    b_fus_bot, b_fus_top = box(ax, cx, 10.2, 5.4, h, "Weighted fusion + structural policies", GREEN)
    b_sort_bot, b_sort_top = box(ax, cx, 8.6, 3.8, h, "Single sort to top-k", GREEN)

    y_out = 7.0
    b_diag_bot, _ = box(ax, 2.0, y_out, 2.4, h, "Read-only\ndiagnosis", GRAY, fs=7.5)
    b_eval_bot, _ = box(ax, cx, y_out, 2.4, h, "Evaluation\n(Hit@k)", GRAY, fs=7.5)
    b_rep_bot, _ = box(ax, 8.0, y_out, 2.4, h, "Stakeholder\nreport", GRAY, fs=7.5)
    _, b_out_top = b_diag_bot, y_out + h / 2

    arrow_v(ax, 3.0, b_pass_bot, b_parse_top)
    arrow_v(ax, 7.0, b_fail_bot, b_parse_top)
    arrow_v(ax, cx, b_parse_bot, b_sig_top)
    arrow_v(ax, cx, b_sig_bot, b_chan_top)
    arrow_v(ax, cx, b_chan_bot, b_llm_top)
    arrow_v(ax, cx, b_chan_bot, b_vc_top)
    arrow_v(ax, cx, b_chan_bot, b_vlm_top)
    arrow_v(ax, 2.0, b_llm_bot, b_fus_top)
    arrow_v(ax, cx, b_vc_bot, b_fus_top)
    arrow_v(ax, 8.0, b_vlm_bot, b_fus_top)
    arrow_v(ax, cx, b_fus_bot, b_sort_top)
    arrow_v(ax, cx, b_sort_bot, b_out_top)
    arrow_v(ax, 2.0, b_sort_bot, b_out_top)
    arrow_v(ax, 8.0, b_sort_bot, b_out_top)

    fig.savefig(OUT / "arch_v2_pipeline.png", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_v1():
    fig, ax = plt.subplots(figsize=(6.0, 9.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis("off")
    ax.set_title("TraceLens v1: Stacked Overrides (superseded)", fontsize=11, fontweight="bold", pad=10)

    cx, w, h, gap = 5.0, 4.8, 0.72, 1.15
    stages = [
        ("Pass/fail parse and align", GRAY),
        ("Heuristic sort", GRAY),
        ("LLM hard rerank", GRAY),
        ("Hit@k guard", RED),
        ("Deterministic promote", RED),
        ("Diagnosis promote", RED),
        ("VLM ensemble", GRAY),
        ("Post-blend guard", RED),
        ("Final rank and report", WARN),
    ]
    y = 18.0
    edges = []
    for label, color in stages:
        bot, top = box(ax, cx, y, w, h, label, color, fs=8)
        edges.append((bot, top))
        y -= gap

    for i in range(len(edges) - 1):
        arrow_v(ax, cx, edges[i][0], edges[i + 1][1])

    ax.text(
        cx,
        8.0,
        "Hybrid mode runs both override chains;\ntrace-specific guards can conflict.",
        ha="center",
        va="center",
        fontsize=8,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#fff8e1", edgecolor="#bbb"),
    )

    legend_y = 6.5
    ax.add_patch(FancyBboxPatch((1.2, legend_y - 0.35), 1.0, 0.5, facecolor=RED, **BOX))
    ax.text(2.9, legend_y - 0.1, "Can override rank", fontsize=7.5, va="center")
    ax.add_patch(FancyBboxPatch((5.5, legend_y - 0.35), 1.0, 0.5, facecolor=GRAY, **BOX))
    ax.text(7.2, legend_y - 0.1, "Processing stage", fontsize=7.5, va="center")

    fig.savefig(OUT / "arch_v1_pipeline.png", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    fig_v2()
    fig_v1()
    print(f"Wrote {OUT / 'arch_v2_pipeline.png'}")
    print(f"Wrote {OUT / 'arch_v1_pipeline.png'}")
