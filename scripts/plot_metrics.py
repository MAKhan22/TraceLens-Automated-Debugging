#!/usr/bin/env python3
"""
plot_metrics.py — publication figures for TraceLens evaluation.

Reads scripts/metrics_manifest.yaml and writes to outputs/figures/:

  fig1_aggregate_hitk.png   Main result: Hit@1/3/5, v1 vs v2 × 5 configs
  fig2_v2_by_source.png     v2 Hit@1 broken down by trace source set
  fig3_v2_trace_heatmap.png Per-trace Hit@1 (v2): traces × configs
  fig4_v2_rank_distance.png Per-trace rank distance strip chart (v2)
  fig5_radar_combined.png   Multi-metric radar: v1 | v2 side-by-side
  fig6_llm_vs_hybrid_tradeoff.png  v2: LLM vs Hybrid on Hit@1/3 + rank quality

Usage:
  python scripts/plot_metrics.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]

# ── palette (Okabe–Ito, colorblind-safe) ─────────────────────────────────
CFG_COLORS = {
    "heuristic": "#0072B2",
    "heuristic_no_pixel": "#009E73",
    "llm": "#D55E00",
    "vlm": "#CC79A7",
    "hybrid": "#E69F00",
}

CONFIGS: list[tuple[str, str]] = [
    ("heuristic", "Heuristic"),
    ("heuristic_no_pixel", "No pixel"),
    ("llm", "LLM"),
    ("vlm", "VLM"),
    ("hybrid", "Hybrid"),
]

SOURCE_META = {
    "efe_irem": {"label": "efe/irem", "color": "#0072B2", "short": "efe"},
    "areeb_salem": {"label": "areeb/salem", "color": "#D55E00", "short": "areeb"},
    "ersel": {"label": "ersel (test set)", "color": "#009E73", "short": "ersel"},
}


def _source_label(src: str) -> str:
    return SOURCE_META[src]["label"]

HIT_METRICS = [
    ("hit@1", "Hit@1"),
    ("hit@3", "Hit@3"),
    ("hit@5", "Hit@5"),
]

RADAR_METRICS = [
    ("hit@1", "Hit@1", True),
    ("hit@3", "Hit@3", True),
    ("hit@5", "Hit@5", True),
    ("mean_rank_distance", "Rank dist ↓", False),
    ("mean_mad@5", "MAD@5 ↓", False),
    ("found_in_top5_rate", "Top-5", True),
]

V1_COLOR = "#6B7B8C"
V2_COLOR = "#C44E52"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _resolve_paths(entry: str | list[str] | None, base: Path) -> list[Path]:
    if entry is None:
        return []
    entries = [entry] if isinstance(entry, str) else list(entry)
    return [(base / p).resolve() for p in entries]


def load_run(paths: list[Path]) -> dict[str, Any] | None:
    if not paths:
        return None
    merged: dict[str, dict] = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        for row in _load_json(path).get("per_trace") or []:
            if not row.get("exclude_from_aggregate"):
                merged[row["trace_id"]] = row
    if not merged:
        return None
    traces = list(merged.values())
    return {"per_trace": traces, "aggregate": compute_aggregate(traces)}


def compute_aggregate(traces: list[dict]) -> dict[str, float | int]:
    n = len(traces)
    if n == 0:
        return {"n_traces": 0}

    def mean(key: str) -> float:
        vals = [float(t[key]) for t in traces if t.get(key) is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    h1 = sum(int(t.get("hit@1") or 0) for t in traces)
    h3 = sum(int(t.get("hit@3") or 0) for t in traces)
    h5 = sum(int(t.get("hit@5") or 0) for t in traces)
    return {
        "n_traces": n,
        "hit@1": h1 / n, "hit@1_count": h1,
        "hit@3": h3 / n, "hit@3_count": h3,
        "hit@5": h5 / n, "hit@5_count": h5,
        "mean_rank_distance": mean("rank_distance"),
        "mean_mad@5": mean("mad@5"),
        "found_in_top5_rate": h5 / n,
    }


def trace_source(tid: str) -> str:
    return tid.split("/")[0] if "/" in tid else "unknown"


def short_trace_name(tid: str) -> str:
    return tid.split("/", 1)[-1].replace("_", " ")[:18]


def load_manifest(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    out: dict[str, dict[str, list[Path]]] = {}
    for ver in ("v1", "v2"):
        out[ver] = {
            k: _resolve_paths(raw.get(ver, {}).get(k), ROOT) for k, _ in CONFIGS
        }
    return out


def build_results(manifest: dict) -> tuple[dict, dict]:
    results, runs = {}, {}
    for ver in ("v1", "v2"):
        results[ver], runs[ver] = {}, {}
        for cfg_key, _ in CONFIGS:
            run = load_run(manifest[ver].get(cfg_key) or [])
            runs[ver][cfg_key] = run
            results[ver][cfg_key] = {"all": compute_aggregate((run or {}).get("per_trace") or [])}
            for src in SOURCE_META:
                traces = [t for t in (run or {}).get("per_trace") or [] if t["trace_id"].startswith(f"{src}/")]
                results[ver][cfg_key][src] = compute_aggregate(traces)
    return results, runs


def _canonical_trace_order(runs: dict) -> list[str]:
    seen: dict[str, list[str]] = {s: [] for s in SOURCE_META}
    for cfg_key, _ in CONFIGS:
        run = runs.get("v2", {}).get(cfg_key)
        if not run:
            continue
        for row in run["per_trace"]:
            src = trace_source(row["trace_id"])
            if src in seen and row["trace_id"] not in seen[src]:
                seen[src].append(row["trace_id"])
    out: list[str] = []
    for src in SOURCE_META:
        out.extend(sorted(seen[src]))
    return out


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", facecolor="white")
    plt.close(fig)


def _bar_label(ax, bar, text: str, y_pad: float = 0.02) -> None:
    if not text:
        return
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + y_pad,
        text, ha="center", va="bottom", fontsize=7.5, color="#222222",
    )


# ── Figure 1: main Hit@k comparison ──────────────────────────────────────

def fig1_aggregate_hitk(results: dict, out_dir: Path) -> None:
    """Hit@1/3/5 — grouped v1 (hatched) vs v2 (solid) for each config."""
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)
    fig.suptitle(
        "Aggregate ranking accuracy — Hit@k (primary); all evaluated traces",
        fontsize=12, y=1.02,
    )

    n = len(CONFIGS)
    x = np.arange(n)
    w = 0.36
    labels = [c[1] for c in CONFIGS]

    for ax, (key, title) in zip(axes, HIT_METRICS):
        for i, (ver, color, hatch) in enumerate([("v1", V1_COLOR, "///"), ("v2", V2_COLOR, "")]):
            vals, notes = [], []
            for cfg_key, _ in CONFIGS:
                agg = results[ver][cfg_key]["all"]
                nt = agg.get("n_traces", 0)
                if nt == 0:
                    vals.append(0.0)
                    notes.append("")
                else:
                    vals.append(float(agg[key]))
                    notes.append(f"{int(agg[f'{key}_count'])}/{nt}")
            off = (i - 0.5) * w
            bars = ax.bar(
                x + off, vals, w, label=f"Pipeline {ver}",
                color=color, edgecolor="#333333", linewidth=0.6,
                hatch=hatch, alpha=0.92 if ver == "v2" else 0.75,
            )
            if key == "hit@1":
                for bar, note in zip(bars, notes):
                    _bar_label(ax, bar, note)

        ax.set_title(f"{title}  (↑ better)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylim(0, 1.12)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.yaxis.grid(True, linestyle="-", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc="lower left", framealpha=0.95, edgecolor="#CCCCCC")
    fig.tight_layout()
    _save(fig, out_dir, "fig1_aggregate_hitk")


# ── Figure 2: v2 by source set ───────────────────────────────────────────

def fig2_v2_by_source(results: dict, out_dir: Path) -> None:
    """v2 Hit@1 for each trace collection × config."""
    fig, ax = plt.subplots(figsize=(8, 3.8))
    sources = list(SOURCE_META.keys())
    n_cfg = len(CONFIGS)
    group_x = np.arange(len(sources))
    w = 0.15
    offsets = np.linspace(-(n_cfg - 1) / 2, (n_cfg - 1) / 2, n_cfg) * w

    for j, (cfg_key, cfg_label) in enumerate(CONFIGS):
        vals, notes = [], []
        for src in sources:
            agg = results["v2"][cfg_key].get(src, {"n_traces": 0})
            nt = agg.get("n_traces", 0)
            if nt == 0:
                vals.append(0.0)
                notes.append("")
            else:
                vals.append(float(agg["hit@1"]))
                notes.append(f"{int(agg['hit@1_count'])}/{nt}")
        bars = ax.bar(
            group_x + offsets[j], vals, w * 0.92,
            label=cfg_label, color=CFG_COLORS[cfg_key],
            edgecolor="#333333", linewidth=0.5,
        )
        for bar, note in zip(bars, notes):
            _bar_label(ax, bar, note, y_pad=0.015)

    ax.set_xticks(group_x)
    ax.set_xticklabels([_source_label(s) for s in sources])
    ax.set_ylabel("Hit@1 accuracy")
    ax.set_ylim(0, 1.15)
    ax.set_title("v2 Hit@1 by trace source set")
    ax.yaxis.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18), framealpha=0.95)
    fig.tight_layout()
    _save(fig, out_dir, "fig2_v2_by_source")


# ── Figure 3: per-trace heatmap ──────────────────────────────────────────

def fig3_v2_trace_heatmap(runs: dict, out_dir: Path) -> None:
    """v2: rows = traces, cols = configs, cell = Hit@1 (green/red)."""
    trace_order = _canonical_trace_order(runs)
    if not trace_order:
        return

    n_t, n_c = len(trace_order), len(CONFIGS)
    mat = np.full((n_t, n_c), np.nan)
    for j, (cfg_key, _) in enumerate(CONFIGS):
        by_id = {r["trace_id"]: r for r in (runs["v2"].get(cfg_key) or {}).get("per_trace") or []}
        for i, tid in enumerate(trace_order):
            row = by_id.get(tid)
            if row is not None:
                mat[i, j] = int(row.get("hit@1") or 0)

    cmap = ListedColormap(["#D73027", "#1A9850"])
    fig_h = max(5.5, n_t * 0.26)
    fig = plt.figure(figsize=(7.8, fig_h))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.55, 6], wspace=0.06)
    ax_band = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1])

    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")

    ax.set_xticks(range(n_c))
    ax.set_xticklabels([c[1] for c in CONFIGS], rotation=35, ha="right")
    ax.set_yticks(range(n_t))
    ax.set_yticklabels([short_trace_name(t) for t in trace_order], fontsize=7)
    ax.set_xlabel("Configuration")

    # Left gutter: colored bands + source labels (no overlap with trace names)
    ax_band.set_xlim(0, 1)
    ax_band.set_ylim(n_t - 0.5, -0.5)
    ax_band.axis("off")

    pos = 0
    for src in SOURCE_META:
        count = sum(1 for t in trace_order if trace_source(t) == src)
        if count == 0:
            continue
        y0 = pos - 0.5
        ax_band.add_patch(mpatches.Rectangle(
            (0.05, y0), 0.22, count,
            facecolor=SOURCE_META[src]["color"], alpha=0.35,
            edgecolor=SOURCE_META[src]["color"], linewidth=1.2,
        ))
        mid = pos + count / 2 - 0.5
        ax_band.text(
            0.62, mid, _source_label(src),
            va="center", ha="left", fontsize=7.5, fontweight="bold",
            color=SOURCE_META[src]["color"],
        )
        if pos > 0:
            ax.axhline(pos - 0.5, color="#BBBBBB", linewidth=0.8)
        pos += count

    for i in range(n_t):
        for j in range(n_c):
            if not math.isnan(mat[i, j]):
                sym = "✓" if mat[i, j] == 1 else "✗"
                ax.text(j, i, sym, ha="center", va="center", fontsize=7,
                        color="white", fontweight="bold")

    ax.set_title("v2 per-trace Hit@1 (22 traces × 5 configs)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, ticks=[0.25, 0.75])
    cbar.ax.set_yticklabels(["Miss", "Hit"])

    legend_patches = [
        mpatches.Patch(color=SOURCE_META[s]["color"], label=_source_label(s))
        for s in SOURCE_META
    ]
    fig.legend(handles=legend_patches, loc="lower center", bbox_to_anchor=(0.55, -0.06),
               ncol=3, frameon=False, fontsize=8)
    fig.subplots_adjust(bottom=0.14)
    _save(fig, out_dir, "fig3_v2_trace_heatmap")


# ── Figure 4: rank distance strip (replaces scatter) ─────────────────────

def fig4_v2_rank_distance(runs: dict, out_dir: Path) -> None:
    """
    v2: one row per config, x = trace index, y = rank distance.
    Dots colored by source; y=0 dashed = perfect rank.
    """
    trace_order = _canonical_trace_order(runs)
    if not trace_order:
        return
    n_t = len(trace_order)
    x = np.arange(1, n_t + 1)

    fig, axes = plt.subplots(len(CONFIGS), 1, figsize=(10, 7), sharex=True)
    fig.suptitle("v2 per-trace rank distance (0 = correct top rank)", fontsize=12, y=0.995)

    band_ends = []
    pos = 0
    for src in SOURCE_META:
        c = sum(1 for t in trace_order if trace_source(t) == src)
        pos += c
        band_ends.append(pos)

    for ax, (cfg_key, cfg_label) in zip(axes, CONFIGS):
        by_id = {r["trace_id"]: r for r in (runs["v2"].get(cfg_key) or {}).get("per_trace") or []}
        for i, tid in enumerate(trace_order):
            row = by_id.get(tid)
            if row is None:
                continue
            src = trace_source(tid)
            rd = float(row.get("rank_distance") or 0)
            ax.scatter(
                x[i], rd, s=36, c=SOURCE_META[src]["color"],
                edgecolors="white", linewidths=0.6, zorder=3,
            )
        ax.axhline(0, color="#1A9850", linestyle="--", linewidth=0.9, alpha=0.7, zorder=1)
        for be in band_ends[:-1]:
            ax.axvline(be + 0.5, color="#DDDDDD", linestyle="-", linewidth=0.8, zorder=0)
        ax.set_ylabel(cfg_label, fontsize=9, rotation=0, ha="right", va="center")
        ax.set_ylim(bottom=-0.2)
        ax.yaxis.grid(True, alpha=0.2)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=8)

    axes[-1].set_xlabel("Test trace index (efe → areeb → ersel (test set))")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([str(i) for i in x], fontsize=6, rotation=90)

    legend_patches = [
        mpatches.Patch(color=SOURCE_META[s]["color"], label=_source_label(s))
        for s in SOURCE_META
    ]
    fig.legend(handles=legend_patches, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=3, frameon=False, fontsize=9)
    fig.subplots_adjust(hspace=0.08, bottom=0.12)
    _save(fig, out_dir, "fig4_v2_rank_distance")


# ── Figure 5: combined radar ─────────────────────────────────────────────

def _radar_score(val: float | None, key: str, higher_better: bool) -> float:
    if val is None or math.isnan(val):
        return 0.0
    if key.startswith("hit@") or key.endswith("_rate"):
        s = float(val)
    elif key == "mean_rank_distance":
        s = 1.0 - min(float(val) / 3.0, 1.0)
    elif key == "mean_mad@5":
        s = 1.0 - min(float(val) / 12.0, 1.0)
    else:
        s = float(val)
    return max(0.0, min(1.0, s))


def fig5_radar_combined(results: dict, out_dir: Path) -> None:
    labels = [m[1] for m in RADAR_METRICS]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles_c = angles + angles[:1]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), subplot_kw={"projection": "polar"})
    fig.suptitle("Multi-metric profile by configuration", fontsize=12, y=1.05)

    for ax, ver in zip(axes, ("v1", "v2")):
        for cfg_key, cfg_label in CONFIGS:
            agg = results[ver][cfg_key]["all"]
            if agg.get("n_traces", 0) == 0:
                continue
            vals = [
                _radar_score(agg.get(k), k, hb)
                for k, _, hb in RADAR_METRICS
            ]
            vals_c = vals + vals[:1]
            ax.plot(angles_c, vals_c, color=CFG_COLORS[cfg_key], linewidth=1.8,
                    label=f"{cfg_label} (n={agg['n_traces']})")
            ax.fill(angles_c, vals_c, color=CFG_COLORS[cfg_key], alpha=0.06)
        ax.set_xticks(angles)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["", "0.5", "", "1.0"], fontsize=7, color="#AAAAAA")
        ax.grid(color="#CCCCCC", linewidth=0.6)
        ax.set_title(f"Pipeline {ver}", fontsize=11, pad=16)

    handles, lbls = axes[1].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", bbox_to_anchor=(0.5, -0.05),
               ncol=3, frameon=True, fontsize=8, edgecolor="#CCCCCC")
    fig.tight_layout()
    _save(fig, out_dir, "fig5_radar_combined")


# ── Figure 6: LLM vs Hybrid tradeoff (v2) ─────────────────────────────────

TRADEOFF_CFGS = [("llm", "LLM"), ("hybrid", "Hybrid")]

TRADEOFF_METRICS = [
    ("hit@1", "Hit@1", True, True),
    ("hit@3", "Hit@3", True, True),
    ("mean_rank_distance", "Mean rank distance", False, False),
    ("mean_mad@5", "Mean MAD@5", False, False),
]


def fig6_llm_vs_hybrid_tradeoff(results: dict, out_dir: Path) -> None:
    """
    v2 only: LLM vs Hybrid on headline (Hit@k) and rank-quality metrics.
    Shows complementary strengths — not a single winner on every axis.
    """
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    fig.suptitle(
        "v2 — LLM vs Hybrid: complementary metrics (22 traces)",
        fontsize=12, fontweight="bold", y=1.02,
    )

    x = np.arange(len(TRADEOFF_CFGS))
    w = 0.55
    mode_labels = [label for _, label in TRADEOFF_CFGS]

    for ax, (key, title, higher_better, show_count) in zip(axes.flatten(), TRADEOFF_METRICS):
        vals, colors, notes = [], [], []
        for cfg_key, cfg_label in TRADEOFF_CFGS:
            agg = results["v2"][cfg_key]["all"]
            nt = agg.get("n_traces", 0)
            colors.append(CFG_COLORS[cfg_key])
            if nt == 0:
                vals.append(0.0)
                notes.append("")
            else:
                vals.append(float(agg[key]))
                if show_count and key.startswith("hit@"):
                    notes.append(f"{int(agg[f'{key}_count'])}/{nt}")
                else:
                    notes.append(f"{vals[-1]:.2f}")

        bars = ax.bar(
            x, vals, w, color=colors,
            edgecolor="#333333", linewidth=0.7,
        )
        for bar, note in zip(bars, notes):
            _bar_label(ax, bar, note, y_pad=0.03 if key.startswith("hit@") else 0.08)

        arrow = "↑ better" if higher_better else "↓ better"
        ax.set_title(f"{title}  ({arrow})")
        ax.set_xticks(x)
        ax.set_xticklabels(mode_labels)
        ax.yaxis.grid(True, alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)

        if key.startswith("hit@"):
            ax.set_ylim(0, 1.12)
            ax.set_ylabel("Rate")
        else:
            ymax = max(vals) * 1.25 if vals else 1.0
            ax.set_ylim(0, ymax)
            ax.set_ylabel("Steps")

    fig.text(
        0.5, -0.02,
        "LLM wins Hit@1 on amazon; Hybrid wins Hit@3 on github and has lower rank distance / MAD@5.",
        ha="center", fontsize=8.5, color="#444444", style="italic",
    )
    fig.tight_layout()
    _save(fig, out_dir, "fig6_llm_vs_hybrid_tradeoff")


def _cleanup_stale(out_dir: Path) -> None:
    keep_prefixes = ("fig1_", "fig2_", "fig3_", "fig4_", "fig5_", "fig6_")
    for p in list(out_dir.glob("*")):
        if p.suffix == ".pdf":
            p.unlink()
        elif p.suffix == ".png" and not p.name.startswith(keep_prefixes):
            p.unlink()


def print_summary(results: dict) -> None:
    print("\n=== Hit@1 (all traces) ===")
    print(f"{'Config':<14} {'v1':>10} {'v2':>10}")
    print("-" * 36)
    for cfg_key, cfg_label in CONFIGS:
        a1, a2 = results["v1"][cfg_key]["all"], results["v2"][cfg_key]["all"]
        def fmt(a):
            return f"{int(a['hit@1_count'])}/{a['n_traces']}" if a.get("n_traces") else "n/a"
        print(f"{cfg_label:<14} {fmt(a1):>10} {fmt(a2):>10}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="TraceLens paper figures")
    parser.add_argument("--manifest", type=Path, default=ROOT / "scripts/metrics_manifest.yaml")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/figures")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest.resolve())
    results, runs = build_results(manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_stale(args.out_dir)

    print_summary(results)
    print(f"Writing → {args.out_dir.resolve()}\n")

    fig1_aggregate_hitk(results, args.out_dir)
    fig2_v2_by_source(results, args.out_dir)
    fig3_v2_trace_heatmap(runs, args.out_dir)
    fig4_v2_rank_distance(runs, args.out_dir)
    fig5_radar_combined(results, args.out_dir)
    fig6_llm_vs_hybrid_tradeoff(results, args.out_dir)

    print("Done:")
    for p in sorted(args.out_dir.glob("fig*")):
        print(f"  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
