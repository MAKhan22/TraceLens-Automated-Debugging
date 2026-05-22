# TraceLens — VLM Visual Analysis Guide (Phase 2)

This document explains how TraceLens uses **Vision-Language Models (VLMs)** to compare pass/fail screenshots, how that integrates with the text-based LLM pipeline, and how to run and tune it.

---

## Why VLM exists

Some faults are **invisible to text-only analysis**:

| Fault type | Text signal | Visual signal |
|-----------|-------------|---------------|
| Wrong page loaded (e.g. CAPTCHA, login wall) | Often none | Clear — different page content |
| UI element missing / not rendered | Often none | Clear — button or section absent |
| Scroll position unchanged | None | Clear — same viewport, wrong section visible |
| Stale CDN cache (old UI version) | None | Clear — outdated layout vs fresh pass run |
| Empty form field when pass had text | Weak action diff | Clear — field visibly blank |

The LLM reads network logs, console logs, and action text. The VLM reads **what the browser actually showed** at each step. Together they cover both signal types.

All 22 evaluation traces include screenshots — one image per step for both pass and fail runs.

---

## How LLM + VLM work together

TraceLens does **not** run VLM and LLM in parallel on the same input. They run **sequentially**, each contributing a different kind of evidence:

```
Heuristic ranking
       │
       ▼
LLM re-rank + diagnosis          ← text: actions, network diffs, console diffs
       │
       ▼
VLM visual analysis              ← images: pass vs fail screenshot pairs
       │
       ▼
Ensemble merge                   ← weighted combination → final top-5
       │
       ▼
Report + evaluation
```

### Step-by-step

**1. Heuristic + LLM (unchanged from Phase 1)**

The LLM re-ranks all steps using slimmed text data and diagnoses a root cause. The diagnosed step is promoted to rank #1 if needed (`llm+diagnosis` mode).

**2. Screenshot resolution**

For the current top-K ranked steps (default: top 5), `ScreenshotResolver` finds the matching pass and fail screenshot paths. Each data source uses a different folder layout — the resolver handles all three automatically (see [Screenshot paths](#screenshot-paths-by-source) below).

**3. VLM visual analysis (1 API call per trace)**

The VLM receives all K screenshot pairs in a **single API call**. For each step it sees:

```
--- Step 7: Click "Add to cart" ---
Pass screenshot: [image]
Fail screenshot: [image]
```

The prompt (`prompts/vlm_visual_rank.txt`) instructs it to look for wrong pages, missing elements, error screens, CAPTCHA walls, scroll issues, etc., while ignoring dynamic noise (timestamps, ads).

It returns:

```json
{
  "visual_scores": [
    {"step_id": 7, "visual_score": 0.9, "visual_note": "Fail shows login wall, pass shows product page"}
  ],
  "visual_root_cause_step_id": 7,
  "visual_summary": "Step 7 navigated to a login page instead of the product page."
}
```

**4. Ensemble merge**

`VlmReasoner.ensemble_rankings()` combines the LLM ranking with VLM visual scores:

```
combined_score = (1 - w) × llm_rank_score + w × visual_score
```

Where:
- `w` = `ensemble_vlm_weight` (default **0.4** → 60% LLM, 40% VLM)
- `llm_rank_score` = position-based: rank 1 → 1.0, rank 5 → 0.2
- `visual_score` = VLM's 0–1 anomaly rating for that step

Steps are re-sorted by `combined_score`.

**5. Visual promotion (optional override)**

If the VLM identifies a `visual_root_cause_step_id` with `visual_score ≥ 0.7` and that step is in the top-5 but not at rank #1, it is **promoted to #1**. The report's `ranking_mode` becomes `llm+vlm+visual_promoted`.

This is analogous to the LLM's diagnosis promotion — the model that has the strongest direct evidence for the root cause wins rank #1.

---

## Execution modes

| Command | Mode | API calls per trace | Best for |
|---------|------|---------------------|----------|
| `python main.py` | LLM only | 3 (re-rank, diagnose, summary) | Text-visible faults, fastest |
| `python main.py --vlm` | **LLM + VLM** | 4 (+ 1 VLM call) | **Recommended** — best overall accuracy |
| `python main.py --no-llm --vlm` | VLM only | 1 | Debugging visual-only faults |
| `python main.py --no-llm` | Heuristic only | 0 | No API needed |

Run reports are saved with mode-prefixed filenames:

```
outputs/metrics/runs/
  llm_run_TIMESTAMP.json
  llm+vlm_run_TIMESTAMP.json
  vlm_run_TIMESTAMP.json
  heuristic_run_TIMESTAMP.json
```

---

## Configuration

All VLM settings are in the `vlm:` section of `config.yaml`:

```yaml
vlm:
  base_url:    "https://openrouter.ai/api/v1"
  vlm_model:   "nvidia/nemotron-nano-12b-v2-vl:free"
  api_key_env: "OPENROUTER_API_KEY"   # same key as LLM
  temperature: 0.1

  ensemble_vlm_weight: 0.4   # 0.0 = ignore VLM scores, 1.0 = VLM only
  top_k_for_vlm: 5           # how many top-ranked steps get screenshot pairs
```

| Field | Default | Effect |
|-------|---------|--------|
| `vlm_model` | `nvidia/nemotron-nano-12b-v2-vl:free` | OpenRouter free VLM. Must support image input. |
| `ensemble_vlm_weight` | `0.4` | How much VLM influences the final ranking. Increase toward `0.6` if visual faults dominate your dataset. |
| `top_k_for_vlm` | `5` | Screenshot pairs sent per trace. Higher = broader visual coverage but more tokens/images per call. |

### Tuning tips

- **Visual faults still missed**: Increase `top_k_for_vlm` to `8` or `10` so the VLM sees more steps beyond the LLM's top-5.
- **VLM overriding good LLM rankings**: Lower `ensemble_vlm_weight` to `0.25`–`0.3`.
- **Screenshot-only traces** (`saucedemo_2`, `pypi`, `bbc`): Run with `--vlm`; text-only LLM will not find these.

---

## Screenshot paths by source

Screenshots live in the original raw data folder (`data/raw_base`), not in `data/processed`. `ScreenshotResolver` maps step IDs to paths per source:

### `efe_irem` — two layouts

**Layout A** (saucedemo_1/2, gutenberg, elinguistics):
```
{trace}/screenshots/correct/step_008_after.png    ← pass, 1-indexed, 3-digit
{trace}/screenshots/incorrect/step_008_after.png  ← fail
```

**Layout B** (dictionary, webmd, wikipedia, wolfram):
```
{trace}/correct screenshots/step_004_after.png    ← pass
{trace}/incorrect screenshots/step_004_after.png  ← fail
```

### `areeb_salem`
```
{trace}/pass/screenshots/step_05.png   ← pass, 0-indexed, 2-digit
{trace}/fail/screenshots/step_05.png   ← fail
```

### `ersel`
```
{trace}/passing/step_5_post.png   ← pass, 0-indexed, no padding
{trace}/failing/step_5_post.png   ← fail
```

If a screenshot file is missing for a step, that step is skipped silently — the VLM runs on whatever pairs are available.

---

## What appears in the report

When VLM runs, the JSON and text reports include a `visual_analysis` block:

```json
"visual_analysis": {
  "visual_root_cause_step_id": 7,
  "visual_summary": "Fail shows login wall instead of product page.",
  "visual_scores": [
    {"step_id": 7, "visual_score": 0.9, "visual_note": "Login wall in fail, product page in pass"}
  ],
  "steps_with_screenshots": 5
}
```

The text report adds a **VISUAL ANALYSIS (VLM)** section with per-step visual scores and notes, below the technical root cause block.

---

## Expected impact on accuracy

From [`docs/FAILURE_ANALYSIS.md`](FAILURE_ANALYSIS.md), several traces that scored Hit@5=0 with heuristic-only or text-only LLM are **VLM-dependent**:

| Trace | Why text fails | What VLM should see |
|-------|---------------|---------------------|
| `efe_irem/saucedemo_2` | No network/console signal | Wrong page or missing UI element |
| `areeb_salem/pypi` | No text signal | Visual layout difference |
| `areeb_salem/bbc` | Stale CDN — no HTTP error | Old UI version vs fresh pass |
| `efe_irem/wikipedia` | Zero text signal for fault | Visual scroll/render difference |

Running `python main.py --vlm` on the full 22-trace set is the intended final evaluation mode for the project.

---

## Rate limits and cost

- VLM calls send **base64-encoded PNG images** — much heavier than text-only calls.
- One VLM call per trace (all top-K pairs in one request) keeps total calls at 22 for a full run vs potentially 110 if done per-step.
- The same exponential backoff used for LLM applies to VLM (429 → wait → retry indefinitely).
- Use `--source` / `--trace` / `--skip` / `--from` to split a full run across sessions if rate limits are hit.

```bash
# Run visual-heavy traces only
python main.py --source efe_irem --trace saucedemo_2 --vlm
python main.py --source areeb_salem --trace pypi --vlm
python main.py --source areeb_salem --trace bbc --vlm
```
