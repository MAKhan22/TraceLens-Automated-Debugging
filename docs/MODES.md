# TraceLens — Pipeline Modes & Signal Layers

This document explains **what runs under each CLI flag**, how the three visual layers relate (pixel boost, visual causal, VLM), and what happens when screenshots are missing.

For VLM setup, models, and tuning see [`VLM.md`](VLM.md). For config fields see [`CONFIG.md`](CONFIG.md).

---

## CLI flags at a glance

| Command | Final ranking | API calls |
|---------|---------------|-----------|
| `python main.py` | Heuristic (+ pixel boost by default) | None |
| `python main.py --no-pixel` | Heuristic (text only) | None |
| `python main.py --llm` | LLM re-rank + diagnosis | LLM only |
| `python main.py --vlm` | VLM (+ visual pipeline) | VLM only |
| `python main.py --llm --vlm` | Hybrid ensemble (LLM then VLM) | LLM + VLM |

**Rule of thumb:** each flag turns on its own layer. Nothing cross-contaminates.

- `--llm` → text reasoning only (no screenshot scan, no visual causal, no VLM).
- `--vlm` → full local visual pipeline + VLM (no LLM unless both flags are set).
- Both flags → LLM first on text, then visual pipeline + VLM ensemble.

---

## Pipeline by mode

### Heuristic only (default)

```
Parse & align traces
       │
       ▼
Text signal scoring (network, console, action, intent)
       │
       ▼
Heuristic rank (top-5)
       │
       ▼
Pixel boost on top-5 (optional, default on)   ← local PNG diff, no API
       │
       ▼
Report + evaluation
```

### `--llm`

```
… same text heuristic + pixel boost for report tables …
       │
       ▼
LLM re-rank (text candidates only)
       │
       ▼
LLM diagnosis + optional promotion to #1
       │
       ▼
Report + evaluation
```

The LLM never receives screenshots or pixel/visual-causal scores. Pixel boost only populates an extra **baseline table** in the report for comparison.

### `--vlm`

```
… text heuristic …
       │
       ▼
Full-trace screenshot scan + visual causal attribution   ← local, no API
       │
       ▼
Heuristic re-rank (text + visual_causal where found)
       │
       ▼
Pixel boost table (report only — does not dilute VLM ranking input)
       │
       ▼
VLM analysis on top-K + divergent-window steps
       │
       ▼
VLM-only ensemble (visual causal rules + VLM scores)
       │
       ▼
Report + evaluation
```

### `--llm --vlm`

Same as `--vlm`, but LLM runs **before** the visual pipeline:

```
Text heuristic → LLM re-rank + diagnosis → visual pipeline → VLM → hybrid ensemble
```

Final ranking blends LLM position scores with VLM visual scores (default 60% LLM / 40% VLM via `ensemble_vlm_weight` in `config.yaml`).

---

## Three visual layers (don't confuse them)

All three can use pass/fail PNG files from the trace data, but they differ in scope, method, and which modes run them.

| Layer | Uses PNGs? | Uses VLM API? | Scope | Purpose |
|-------|------------|---------------|-------|---------|
| **Pixel boost** | Yes — heuristic top-5 only | No | 5 steps | Simple blend: `65% text + 35% pixel` (default weights) |
| **Visual causal** | Yes — **every step** | No | Full trace | Find earliest *persistent* divergence; attribute root cause with rules |
| **VLM** | Yes — top-K + injected window | **Yes** | ~5–7 steps | Model interprets *what* changed in the UI |

### Pixel boost

- Loads pass/fail after-screenshots for the current heuristic top-5.
- Computes a local pixel diff score (`src/visual_diff.py`) — center-crop mean difference, 0 = identical.
- **Heuristic-only:** mutates the final ranking.
- **`--llm` / `--vlm`:** builds a separate **report table** only; does not change LLM input or VLM ensemble input (under `--vlm`, the ranking path keeps the full visual-causal score undiluted).

Disable with `--no-pixel`.

### Visual causal

- Runs only when `--vlm` is set (including `--llm --vlm`).
- Scans all steps, computes `global`, `localized`, and `effective` pixel diffs.
- Finds the first divergence that **persists** (skips transient flicker).
- Applies deterministic rules to map visible divergence → silent cause step (e.g. missing zip field on step 27, visible break on step 28).
- Produces `visual_causal_score`, `visual_divergence_step`, `visual_causal_reason` on the attributed root step.
- **Does not call any model.** It is pixel math + rules (`src/visual_signals.py`).

Visual causal also **feeds** the VLM stage: injects divergent-window steps into VLM candidates, backfills VLM scores from symptom steps onto the cause, and caps ranking so downstream steps don't beat the root.

### VLM

- Sends pass/fail screenshot pairs to a vision-language model.
- Returns per-step `visual_score` and a `visual_root_cause_step_id`.
- Required for `--vlm` mode to complete — see [Missing screenshots](#missing-screenshots) below.

---

## Report sections by mode

| Report section | Heuristic | `--llm` | `--vlm` | `--llm --vlm` |
|----------------|-----------|---------|---------|---------------|
| HEURISTIC TOP 5 (text signals only) | If pixel ran | Yes | Yes | Yes |
| HEURISTIC TOP 5 (text + visual causal) | — | — | If signal found | If signal found |
| HEURISTIC TOP 5 (text + pixel) | If pixel ran | If pixel ran | If pixel ran | If pixel ran |
| LLM TOP 5 | — | Yes | — | Yes |
| VLM TOP 5 / VLM + LLM TOP 5 | — | — | Yes | Yes |
| SCREENSHOT ANALYSIS | — | — | Yes | Yes |
| TECHNICAL ROOT CAUSE (LLM) | — | Yes | — | Yes |
| VISUAL ANALYSIS (VLM) | — | — | Yes | Yes |

---

## Missing screenshots

Nothing crashes globally. Each mode degrades gracefully except where the VLM API **requires** at least one image pair.

### Heuristic / `--no-pixel`

| Situation | Behavior |
|-----------|----------|
| No screenshots at all | Text heuristic runs normally |
| Some steps missing PNGs | Pixel boost skipped if **no** top-5 step has both pass + fail files; otherwise missing steps get `pixel=0` |

### `--llm`

| Situation | Behavior |
|-----------|----------|
| No screenshots | LLM runs on text only; no pixel table |
| Partial screenshots | Pixel table may appear; LLM still text-only |

LLM never needs screenshots.

### `--vlm` and `--llm --vlm`

| Situation | Behavior |
|-----------|----------|
| No screenshots anywhere | Text heuristic runs; visual causal finds nothing; **trace is skipped** at VLM with `error: no screenshots` — no report saved for that trace |
| Partial screenshots | Visual causal uses whatever exists (missing steps → zero diff); VLM runs on steps that have both PNGs |
| VLM API failure (credits, rate limit) | Trace skipped; no report |

Other traces in the same batch continue normally.

---

## Design rationale: why visual causal is VLM-gated

Visual causal is kept behind `--vlm` (and `--llm --vlm`) by design:

1. **Clear mode semantics** — heuristic = fast text baseline; LLM = text reasoning; VLM = visual stack.
2. **Evaluation separation** — heuristic metrics measure text (+ optional simple pixel), not the full visual pipeline.
3. **Report clarity** — avoids four overlapping ranking tables in default mode.
4. **Coupling to VLM** — visual causal exists to inject candidates, backfill scores, and enforce cause-over-symptom ranking for VLM.

Pixel boost remains available in heuristic and LLM modes as a **lightweight, top-5-only** visual hint without a full trace scan.

If you need offline screenshot-fault detection without calling the VLM (e.g. saucedemo-style silent input faults), a future opt-in such as `ranking.visual_causal_in_heuristic: true` or a `--visual-causal` flag would be the explicit way to add it without changing default baselines.

---

## Examples

```bash
# Fast baseline — text + pixel on top-5
python main.py

# Text-only baseline
python main.py --no-pixel

# LLM on text; report still shows heuristic + pixel tables
python main.py --llm

# Full visual pipeline + VLM (no LLM)
python main.py --vlm

# LLM first, then visual pipeline + hybrid ranking
python main.py --llm --vlm

# Single trace, skip evaluation
python main.py --source efe_irem --trace saucedemo_2 --vlm --no-eval
```

---

## Related files

| File | Role |
|------|------|
| `main.py` | Flag gating and pipeline order |
| `src/visual_diff.py` | Local pixel diff (pixel boost + visual causal input) |
| `src/visual_signals.py` | Visual causal scan, attribution, VLM inject |
| `src/vlm_reasoner.py` | VLM API + ensemble ranking |
| `src/ranker.py` | Heuristic rank, pixel boost, VLM candidates, backfill |
| `src/report_generator.py` | Multi-table report formatting |
| `src/screenshot_resolver.py` | Pass/fail PNG path resolution per source |
