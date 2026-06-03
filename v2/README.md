# TraceLens v2

Unified fusion ranking: **weighted channels → one sort → no post-hoc promotes**.  
LLM/VLM provide **soft priors**; diagnosis is **read-only**.

## Run

From the project root (same flags as `main.py`):

```bash
python main2.py --source areeb_salem --llm --vlm
python main2.py --trace github --llm --vlm
python main2.py --trace amazon,hackernews,imdb,npm --llm --vlm
python main2.py                              # heuristic-only fusion
python main2.py --no-pixel --no-eval
```

## vs v1 (`main.py`)

| | v1 | v2 |
|---|----|----|
| Final rank | Multiple guards + promoters + blend overrides | Single fusion sort |
| LLM rerank | Hard reorder + anchor/Hit@k guards | `llm_prior` channel |
| VLM | Ensemble + post Hit@k guards | `vlm_prior` channel |
| Diagnosis | Can promote #1 | Explanatory only |
| Hybrid | Special second arbitration pass | Same scorer as LLM/VLM modes |
| Outputs | `outputs/reports/` | `outputs/v2/reports/` |

## Architecture

```
Parse → Align → Text scores → Visual scan (if --llm/--vlm)
    → LLM rerank → llm_prior (optional)
    → VLM analyze → vlm_prior (optional)
    → Fusion channels (v2/features.py)
    → Weighted score (v2/fusion.py)
    → Sort once → top-k
    → LLM diagnosis (read-only)
    → Report + metrics
```

## Fusion channels

Configured in `v2/config.yaml` (merged over root `config.yaml`):

- `text` — heuristic combined score
- `navigation` — wrong-page signals
- `causal_action` — action step that caused errors on next verify
- `observer_symptom` — pixel/VLM on verify steps
- `visual_causal` — earliest screenshot divergence root
- `pixel` — local screenshot diff
- `llm_prior` — LLM rerank position (when `--llm`)
- `vlm_prior` — VLM per-step score (when `--vlm`)

Each ranked step includes `fusion_score` and `fusion_channels` in the JSON report metadata.

## Files

```
v2/
  config.yaml    # v2 weights + output paths overlay
  features.py    # per-step channel extraction
  fusion.py      # weighted fusion + single sort
  runner.py      # per-trace pipeline
  main.py        # CLI + aggregate metrics
main2.py         # convenience entry point
```

Reuses v1 infrastructure: parsers, detector, LLM/VLM API clients, reports, evaluation.

Shared signal improvements (both v1 and v2):

- **Placeholder network noise** — `example.com` distractor URLs filtered in heuristic scoring (`src/anomaly_detector.py` + `src/causal_signals.py`).
- **Consecutive-click visual attribution** — TOC-style faults stay on the diverging click; type→click chains still walk back (`src/visual_signals.py`).

Tests: `tests/test_visual_and_noise.py`.

Reports include **TECHNICAL ROOT CAUSE** (LLM `diagnose`) and **PLAIN LANGUAGE SUMMARY** (LLM `stakeholder_summary`). Both require `--llm`.

## Tuning

Edit fusion weights in `v2/config.yaml`:

```yaml
v2:
  fusion:
    causal_action: 0.20
    observer_symptom: 0.20
    llm_prior: 0.24
    vlm_prior: 0.24
```

Metrics land in `outputs/v2/metrics/runs/v2_*_run_<timestamp>.json`.

LLM calls use OpenRouter **prompt caching** when `model.prompt_cache: true` in root `config.yaml`.

**Reproducibility:** `model.temperature` and `vlm.temperature` are **`0.0`** in root `config.yaml` (greedy sampling). Rerank order is still not guaranteed on OpenRouter free-tier if the backend endpoint changes between runs.

## Observed results (representative runs)

| Mode | Scope | Hit@1 |
|------|-------|-------|
| v2 `--llm --vlm` | areeb_salem (10) | ~80% |
| v2 `--llm --vlm` | efe_irem (8) | ~75% |

Reference runs: `v2_llm+vlm_run_20260601_114558.json` (areeb), `v2_llm+vlm_run_20260601_145850.json` (efe).
