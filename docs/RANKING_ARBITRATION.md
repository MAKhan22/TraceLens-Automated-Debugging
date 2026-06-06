# Ranking arbitration (Hit@1 policy)

TraceLens can promote a step to **rank #1** in several places: post-LLM guards, post-VLM guards, deterministic code rules, and LLM diagnosis. Without a single policy, LLM and VLM paths could diverge, and a blunt “never override the guard” rule can **regress** traces where the ground-truth fault is labeled on the **action** step while a **verify** step has higher pixel score.

This document describes the shared module **`src/ranking_arbitrator.py`**, how it is wired into the pipeline, and how to read decisions in reports.

For CLI modes and when pixel / visual-causal / VLM run, see [`MODES.md`](MODES.md). For VLM scoring and ensemble weights, see [`VLM.md`](VLM.md).

---

## Why this exists

Typical failure patterns on the evaluation set:

| Pattern | What happens | Desired #1 |
|--------|----------------|------------|
| **Action fault** (hackernews, npm, pypi) | Verify/wait step has high pixel; GT is the click/navigate step | Action step — often fixed by `text_causal` deterministic promote after guard |
| **Symptom fault** (github, youtube) | Verify step has very strong pixel; GT is the wait/verify step | Verify step — block `text_causal` from promoting the earlier action |
| **Spurious visual-causal** (amazon) | Scroll step gets VC score; action step is true fault | `text_action_anchor` or causal-over-observer in guard |
| **Spurious wrong_nav** (bbc) | Step 0 flagged as wrong navigation | `wrong_navigation` deterministic still allowed when appropriate |

Previously, **LLM Hit@k guard** lived in `ranker.py` and **VLM Hit@k guard** in `vlm_reasoner.py` as near-duplicate logic. A failed experiment blocked **all** deterministic promotes whenever any visual lock was set, which dropped Hit@1 on action-labeled traces (~40% on areeb_salem vs ~70% baseline).

The arbitrator unifies guard precedence and adds a **narrow** block: only `text_causal` deterministic promote is skipped when #1 is locked by a **strong observer pixel** (verify step, pixel ≥ 0.95).

---

## Module overview

| Symbol | Role |
|--------|------|
| `arbitrate_hit_at_k()` | Shared guard: may promote a step to #1 after LLM rerank or VLM ensemble |
| `should_apply_deterministic_promote()` | Gate for `find_deterministic_root` → `promote_step` on the LLM path |
| `is_strong_observer_pixel_lock()` | True when lock is `pixel_leader` on an observer step with pixel ≥ `STRONG_OBSERVER_PIXEL` (0.95) |
| `ArbitrationResult` | `ranked`, `notes`, `lock_reason` returned to callers |

**Callers**

- `Ranker.guard_llm_hit_at_k()` → `arbitrate_hit_at_k(..., path_label="LLM Hit@k guard")`
- `VlmReasoner._guard_hit_at_k()` → same with `visual_map` and `vlm_output`
- `main.py` (LLM path) → `should_apply_deterministic_promote()` before deterministic promote; `is_strong_observer_pixel_lock()` when deciding whether diagnosis may override #1

VLM-only helpers (e.g. `_guard_pre_vlm_anchor`) remain in `vlm_reasoner.py`; they run **before** arbitration.

---

## Guard policy (`arbitrate_hit_at_k`)

Rules are evaluated **in order**; the first match promotes that step to #1 and sets `lock_reason`.

| Order | Rule | `lock_reason` | LLM | VLM |
|-------|------|---------------|-----|-----|
| 1 | Causal **action** over **observer** pixel leader (errors on next verify step). **VLM / hybrid path** (`vlm_path=True`): always when chain matches. **LLM path** (`vlm_path=False`): blocked at strong observer pixel unless text-only #1 is the same interactive causal step. | `causal_over_observer` | LLM gated | VLM always | Hybrid always |
| 2 | Text-only #1 over spurious **visual-causal** leader (needs `action_score` or causal root) or over observer+VLM leader (network/console/action signal) | `text_action_anchor` | ✓ | ✓ |
| 3 | Visible **symptom** step after visual-causal root | `visible_symptom` | pixel ≥ 0.35 | pixel ≥ 0.35 **and** VLM ≥ 0.5 |
| 4 | Pixel / visual-causal leader | `pixel_leader` or `visual_causal_leader` | pixel ≥ 0.65 or VC > 0 | same + VLM ≥ 0.5 |
| 5 | Pre-heuristic / pre-VLM visual-causal #1 | `visual_causal_heuristic` | VC > 0 | VC > 0 + VLM ≥ 0.5 |
| 6 | Raw VLM root (not downstream of VC root) | `vlm_root` | — | ✓ |

**Strong observer pixel exception (rule 1):** If the pixel-boost #1 observer step has pixel ≥ **0.95**, causal-over-observer is **not** applied. That keeps github/youtube-style verify steps at #1 when the screenshot diff is overwhelming.

---

## After the guard: deterministic promote (LLM only)

On `--llm`, `find_deterministic_root()` may return a step for hard promotion (`wrong_navigation`, `text_causal`, etc.). Visual-causal roots are **not** hard-promoted on the LLM path (they are handled via rerank + guards).

`should_apply_deterministic_promote()`:

| `det_reason` | Behavior |
|--------------|----------|
| `wrong_navigation` | Always promote if not already #1 |
| `text_causal` | Promote **unless** `is_strong_observer_pixel_lock(hit1_lock, top1_step)` |
| Other | Promote if not already #1 |

When blocked, the report records e.g. `Deterministic promote: step N (text_causal) skipped — text_causal blocked — strong observer pixel lock on step M.`

When allowed, ranking mode is `llm+deterministic`; when skipped, ranking stays `llm`.

---

## Diagnosis promote (LLM only)

If there is no deterministic root, the pipeline may promote `diagnosis.root_cause_step_id` to #1 (`llm+diagnosis`), **unless** a visual guard locked #1:

- Lock reasons: `visible_symptom`, `pixel_leader`, `text_action_anchor`, `causal_over_observer`
- Or `is_strong_observer_pixel_lock()` (strong observer pixel on #1 even if `lock_reason` is only `pixel_leader`)

This prevents the diagnosis call from undoing a confident visual/symptom ranking.

---

## End-to-end flow (LLM path)

```
Heuristic rank + inject pool
        │
        ▼
LLM rerank + anchor guard
        │
        ▼
arbitrate_hit_at_k  ──►  lock_reason?, notes in RANKING DECISIONS
        │
        ▼
LLM diagnosis
        │
        ├── find_deterministic_root?
        │       ├── should_apply_deterministic_promote? ──► promote or skip
        │       └── else diagnosis root ──► promote unless visual lock
        ▼
Final top-5 + evaluation
```

VLM path: local pixel/VC scan → (optional LLM) → VLM scores → ensemble → `_guard_pre_vlm_anchor` → `arbitrate_hit_at_k` → report.

Hybrid path: LLM rerank + LLM guards + diagnosis → VLM scores → weighted blend → post-blend `arbitrate_hit_at_k` (same VLM-path policy as `--vlm`). LLM-stage guards and deterministic promote still apply before the blend.

---

## Reading reports

Look at **`ranking_decisions`** in `outputs/reports/{source}/{trace}.json`:

```json
"ranking_decisions": [
  "LLM Hit@k guard: causal action step 12 over observer pixel leader step 13 — promoted to #1 (was step 13).",
  "Deterministic promote: step 12 (text_causal) skipped — text_causal blocked — strong observer pixel lock on step 13."
]
```

Also check **`ranking_mode`**: `llm`, `llm+deterministic`, or `llm+diagnosis`.

---

## Ground-truth labeling caveat

Some traces label the fault on the **action** step (click/navigate); others on the **verify/wait** step where the UI breaks. Arbitration is tuned for Hit@1 on the evaluation JSON, but disagreements between “causal action” and “visible symptom” are expected when labels mix types. Document this when comparing LLM-only vs VLM runs.

---

## Files

| File | Change |
|------|--------|
| `src/ranking_arbitrator.py` | Shared policy (new) |
| `src/ranker.py` | `guard_llm_hit_at_k` delegates to arbitrator |
| `src/vlm_reasoner.py` | `_guard_hit_at_k` delegates to arbitrator |
| `main.py` | Deterministic + diagnosis gates use arbitrator helpers |

---

## VLM candidate pool

`merge_vlm_inject_ids` and the VLM ensemble pool include text-only top-5, causal root, and steps with `console_score ≥ 0.5`, so high text-signal actions (e.g. wishlist click) can reach arbitration even when pixel/VC favors an earlier scroll step.

`promote_to_top` may insert a step from `step_map` when it was not yet in the ranked top-k.

---

## Tuning

| Constant | Default | Effect |
|----------|---------|--------|
| `STRONG_OBSERVER_PIXEL` | `0.95` | Threshold for blocking `text_causal` and skipping causal-over-observer |
| Pixel leader (guard) | `0.65` | Minimum pixel to promote pixel leader (LLM path without VLM confirm) |
| Visible symptom pixel | `0.35` | Minimum pixel on symptom step |
| VLM confirm (VLM path) | `0.5` | Minimum `visual_map` score for pixel/symptom/VC guard tiers |

Adjust in `ranking_arbitrator.py` and re-run targeted traces before full benchmark refreshes.

---

## Related docs

- [`README.md`](../README.md) — pipeline stages 4–6
- [`MODES.md`](MODES.md) — `--llm` / `--vlm` / pixel boost
- [`VLM.md`](VLM.md) — VLM API and ensemble scoring
