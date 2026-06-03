# TraceLens v2

Unified fusion ranking: **one score → one sort → no post-hoc promotes**.  
LLM/VLM provide **soft priors**; diagnosis is **read-only**.

## Run

From the project root (same flags as `main.py`):

```bash
python main2.py --source areeb_salem --llm --vlm
python main2.py --trace github --llm --vlm
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

LLM calls use the shared `LlmReasoner` with OpenRouter **prompt caching** when `model.prompt_cache: true` in root `config.yaml` (see main README §5).

## Observed results (representative runs)

| Mode | Scope | Hit@1 |
|------|-------|-------|
| v2 heuristic | 22 traces | ~64% |
| v2 `--llm` | areeb_salem (10) | ~70% |
| v2 `--vlm` | areeb_salem (10) | ~50% (before/after `vlm_prior` fix — re-run to confirm) |
| v2 `--llm --vlm` | areeb_salem (10) | ~80% |
| v2 `--llm --vlm` | efe_irem (8) | ~75% |

Heuristic-only fusion is often **stronger than LLM-only** on areeb because `--llm` enables screenshot scan + `visual_causal`, which can overrank scroll/symptom steps vs action GT (e.g. amazon). Hybrid (`--llm --vlm`) is the best v2 mode so far.

## Recommended next improvements (generalizable)

These stay inside “one fusion sort” — no v1-style guard stack.

1. **Re-run after `vlm_prior` fix** — VLM scores were mapped from the wrong JSON field; hybrid/VLM-only numbers need a fresh batch.
2. **Conditional `llm_prior` sharpen** — #1 → 1.0, #2+ → ~0.4 when LLM #1 has `action_changed` / `action_divergence` (helps amazon-type; gate so bbc early-nav mistakes are not amplified).
3. **VC vs action conflict** — When LLM #1 is an action step with no pixel and fusion leader is an earlier scroll with `visual_causal` only, discount `visual_causal` on that scroll (symptom vs action GT).
4. **`page_load_noise` channel** — Downweight high `text` on steps flagged `page_load_noise_only` when a later step has strong visual_causal or VLM ≥ 0.5 (wikipedia step 0/2 problem).
5. **Symptom step parity** — Already inject `visual_causal_next_step` into the fusion pool and boost `observer_symptom`; extend to promote `vlm_prior` on symptom step when VLM scores it ≥ root (efe wikipedia GT = visible step).
6. **VLM prior TOP 5 report table** — Same as LLM rerank table, for debugging (not a rank change).
7. **Optional `response_format` + OpenRouter response-healing** on rerank only — fewer malformed `ranked_step_ids` JSON failures.
8. **Weight presets in `v2/config.yaml`** — e.g. `llm_only` lowers `visual_causal` / `pixel` when you want text+LLM without visual stack dominating.

Do **not** blindly raise global `llm_prior` / `vlm_prior` weights — that helps some traces and hurts others (bbc, npm verify-vs-action).
