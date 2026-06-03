# TraceLens — Automated Regression Failure Diagnosis

TraceLens is a multimodal automated debugging system that takes a **passing trace** and a **failing trace** of the same test case, identifies the most suspicious steps, ranks them, and produces a natural-language explanation of the root cause for both engineers and non-technical stakeholders.

---

## Table of Contents

1. [Background & Problem](#1-background--problem)
2. [Architecture Overview](#2-architecture-overview)
3. [Pipeline Stages](#3-pipeline-stages)
4. [Scoring System](#4-scoring-system)
5. [LLM Re-ranking & Diagnosis](#5-llm-re-ranking--diagnosis)
6. [VLM Visual Analysis (Phase 2)](#6-vlm-visual-analysis-phase-2)
7. [Evaluation Metrics](#7-evaluation-metrics)
8. [Data Sources](#8-data-sources)
9. [Project Structure](#9-project-structure)
10. [Setup & Running](#10-setup--running)
11. [Configuration](#11-configuration)
12. [Output Files](#12-output-files)

> **Additional guides**: [`docs/CONFIG.md`](docs/CONFIG.md) — full configuration reference. [`docs/MODES.md`](docs/MODES.md) — pipeline modes, flag behavior, pixel vs visual causal vs VLM. [`docs/VLM.md`](docs/VLM.md) — VLM + LLM integration, screenshot paths, tuning. [`docs/RANKING_ARBITRATION.md`](docs/RANKING_ARBITRATION.md) — shared Hit@1 guard policy, deterministic/diagnosis gates. [`docs/TRACE_SELECTION.md`](docs/TRACE_SELECTION.md) — trace selection rationale. [`docs/FAILURE_ANALYSIS.md`](docs/FAILURE_ANALYSIS.md) — traces that need VLM vs text-only fixes.

---

## 1. Background & Problem

Automated UI test suites produce execution traces — step-by-step records of browser actions, network requests, and console logs. When a test that passed yesterday suddenly fails today, engineers need to:

1. Identify **which step** in the trace first went wrong (root cause), not just the last step that errored out.
2. Distinguish **root cause** (e.g., a wrong password attempt) from **downstream symptoms** (e.g., a verification step that failed because of the wrong password).
3. Communicate the failure in **plain language** to non-technical stakeholders.

TraceLens solves this by diffing the pass and fail traces step-by-step, scoring each step's anomalousness, and using an LLM to reason about causality.

---

## 2. Architecture Overview

```
 Pass trace ─┐
              ├─► TraceParser ─► TraceAligner ─► AnomalyDetector ─► Ranker
 Fail trace ─┘                                        │                │
                                                       │         heuristic top-15 + inject
                                                       │                │
                                                       └──────────► LlmReasoner
                                                                   (re-rank + diagnose)
                                                                         │
                                                    ┌────────────────────┘
                                                    │  (--llm / --vlm flags)
                                                    ▼
                                          ScreenshotResolver ──► VlmReasoner
                                          (pass/fail paths)     (visual scores)
                                                    │
                                                    ▼
                                          Ensemble merge ──► ReportGenerator
                                                    │              │
                                               Evaluation    JSON + text report
```

There are four execution paths (two independent flags):

| Mode | Command | What runs |
|------|---------|-----------|
| **Heuristic only (default)** | `python main.py` | Score + rank by heuristics. Optional pixel-diff boost. No API needed. |
| **LLM only** | `python main.py --llm` | Local screenshot scan + LLM re-rank → guards → diagnosis. No VLM API. |
| **VLM only** | `python main.py --vlm` | Full visual pipeline + VLM API. Falls back to heuristic report if VLM fails. |
| **LLM + VLM** | `python main.py --llm --vlm` | LLM first, then VLM; unified visual scoring + Hit@k guards for final rank. |

Neither flag = heuristic only. Pass `--llm` and/or `--vlm` to enable each stage.

**v2 (fusion experiments)** — `python main2.py` with the same flags writes to `outputs/v2/` and uses a single weighted fusion rank (no post-hoc guards). See [`v2/README.md`](v2/README.md).

---

## 3. Pipeline Stages

### Stage 1 — Trace Parsing (`src/trace_parser.py`)

Converts each raw trace format into a unified list of step dictionaries:

```python
{
  "step_id":      int,         # 0-indexed position in the trace
  "action":       str,         # human-readable action description
  "action_type":  str,         # e.g. "click", "navigate", "type", "verify"
  "intent":       str | None,  # ersel only: "Verification passed/failed: ..."
  "network_logs": [ {"url": ..., "status": ..., "error": ...}, ... ],
  "console_logs": [ {"type": "error"|"warning"|"info", "text": ...}, ... ],
  "screenshot":   str | None,  # relative path; resolved at runtime by ScreenshotResolver
}
```

Three source formats are supported — see [Data Sources](#8-data-sources).

### Stage 2 — Trace Alignment (`src/trace_aligner.py`)

Pairs up steps from the pass and fail traces so they can be compared directly.

- **Same-length traces**: aligned 1:1 by index (`align_by_index`).
- **Different-length traces**: aligned by action text similarity (`align_by_similarity`), which handles inserted/deleted steps caused by the fault.

### Stage 3 — Anomaly Scoring (`src/anomaly_detector.py`)

Each aligned step pair receives four component scores (all in `[0, 1]`) and one `combined_score`. See [Scoring System](#4-scoring-system).

### Stage 4 — Heuristic Ranking (`src/ranker.py`)

Steps are sorted descending by `combined_score`. The top-5 become the **heuristic ranking** (always shown in reports). When `--llm` or `--vlm` is enabled, a **local screenshot scan** (pixel diff + visual-causal attribution) also runs — no VLM API needed for `--llm`. The rerank pool is **heuristic top-15 plus injected strong-signal steps** (`wrong_navigation`, `action_changed`, `errors_observed_on_next_step`, visual-causal root, visible symptom step, pixel leader ≥ 0.65).

### Stage 5 — LLM Re-ranking & Diagnosis (`src/llm_reasoner.py`, requires `--llm`)

See [LLM Re-ranking & Diagnosis](#5-llm-re-ranking--diagnosis).

### Stage 6 — VLM Visual Analysis (`src/vlm_reasoner.py`, requires `--vlm`)

When `--vlm` is passed, the pipeline compares pass/fail screenshot pairs for the top-K ranked steps and merges visual evidence with the current ranking (LLM ranking if `--llm` was also passed, otherwise heuristic). See [VLM Visual Analysis](#6-vlm-visual-analysis-phase-2).

### Stage 7 — Evaluation (`src/evaluation.py`)

Compares predicted rankings against ground truth (`data/evaluation/ground_truth.json`). See [Evaluation Metrics](#7-evaluation-metrics).

### Stage 8 — Report Generation (`src/report_generator.py`)

Assembles and saves the final structured report (JSON + plain text).

---

## 4. Scoring System

Each aligned step pair `(pass_step, fail_step)` is scored by four signals:

### 4.1 Network Score

Compares the network requests made during the pass step vs the fail step.

| Condition | Score |
|-----------|-------|
| New 5xx status (server error) | **1.0** |
| New 404 / 403 / 405 / 429 status | **0.9** |
| Status changed from 2xx to any error | **0.85** |
| New `net::ERR_*` / `FAILED` error string | **0.85** |
| Any other status change | **0.5** |
| URL present in pass but missing in fail (≥50% of requests gone) | **0.7** |
| URL present in pass but missing in fail (<50%) | **0.3** |
| Pass had requests, fail has none at all | **0.7** |

The final `network_score` is the **max** of all individual anomaly scores for that step (worst-case signal wins).

### 4.2 Console Score

Identifies new console log entries in the fail step that were absent in the pass step.

| New entry type | Score contribution |
|----------------|-------------------|
| `error` / `severe` | **+0.9** per entry |
| `warning` | **+0.4** per entry |
| `info` / other | **+0.1** per entry |
| Pass had console activity, fail has none ("suspicious silence") | max with **0.5** |

The final `console_score` is the **sum** of contributions, capped at 1.0.

### 4.3 Action Score

Compares the action text and type between the pass and fail steps.

| Condition | Score |
|-----------|-------|
| Identical action text | **0.0** |
| Same action type, different content, similarity < 50% | **0.5** |
| Same action type, different content, similarity ≥ 50% | **0.3** |
| Different action type entirely | **0.6** |
| Action present in pass but absent in fail (or vice versa) | **0.8** |

### 4.4 Intent Score *(ersel traces only)*

Some traces include a natural-language `intent` field describing what the step attempted and whether it succeeded. This is a direct fault signal.

| Intent text contains | Score |
|----------------------|-------|
| `"Verification failed"` | **1.0** |
| `"not found"` / `"not interactable"` / `"Type failed"` | **0.9** |
| `"Verification passed"` | **0.0** |
| Field absent or empty | **0.0** |

> **Why intent is only in ersel traces**: The three data sources come from separate prior projects (provided by the course instructor), each collected with different tooling and instrumentation. The ersel dataset explicitly logged verification outcomes in the `intent` field; the efe_irem and areeb_salem datasets did not include this field. This is a schema inconsistency inherited from the raw data, not a design choice of this project.
>
> **Consequence & fix**: If no step in a trace has an intent field, the 0.25 weight allocated to intent would permanently go to waste, artificially capping all combined scores at 0.75. TraceLens detects this automatically and **redistributes the intent weight proportionally** to the other three signals so the combined score still spans `[0, 1]`.

### 4.5 Combined Score

```
combined = w_net × network_score
         + w_con × console_score
         + w_act × action_score
         + w_int × intent_score
```

Default weights (when intent data is present):

```
network: 0.35  |  console: 0.25  |  action: 0.15  |  intent: 0.25
```

When no intent data exists (efe_irem / areeb_salem), weights are auto-renormalized to:

```
network: ≈0.467  |  console: ≈0.333  |  action: ≈0.200  |  intent: 0.000
```

Weights can be overridden in `config.yaml`.

> **Important**: Scores drive **heuristic ranking** (stage 4). The LLM reranker does not receive raw numeric scores in compact mode — it sees action text, filtered network/console diffs, causal flags, `heuristic_rank`, and `page_load_noise_only`. An **anchor guard** after reranking prevents strong heuristic top-5 steps from being demoted without cause.

---

## 5. LLM Re-ranking & Diagnosis

The LLM pipeline runs three sequential calls via an OpenAI-compatible API (currently [OpenRouter](https://openrouter.ai) free tier). The active model and provider are set in `config.yaml` — see [`docs/CONFIG.md`](docs/CONFIG.md) for all options.

### Call 1 — Re-ranking

**Input**: Heuristic **top-15** steps (configurable via `pre_llm_k`) **plus injected strong-signal steps** from anywhere in the trace (`wrong_navigation`, `action_changed`, `errors_observed_on_next_step`, causal root, **visual-causal root**, **visible symptom step**, **pixel leader ≥ 0.65**). Each step includes action text, filtered network/console diffs, causal flags, `heuristic_rank`, optional **`visual_causal_score` / `pixel_score` / `screenshots_available`**, and optionally `page_load_noise_only` for noisy homepage loads.

**Prompt** (`prompts/rerank_steps.txt`): Instructs the LLM to deprioritize page-load telemetry, respect heuristic priors, and rank visual-causal type/click steps highly.

**Output**: Re-ordered list of step IDs → top-5 after **anchor guard** and **LLM Hit@k guard** (pixel / visual-causal promotion when VLM is not used).

### Call 2 — Root Cause Diagnosis

**Input**: Top-5 re-ranked steps (same content format, full slim payload).

**Prompt** (`prompts/root_cause.txt`): Asks for the primary fault step, a technical summary, the failure chain, and downstream affected steps.

**Output**: Structured JSON with `root_cause_step_id`, `root_cause_summary`, `failure_chain`, `downstream_steps`.

### Call 3 — Stakeholder Summary

**Input**: The technical diagnosis from Call 2.

**Prompt** (`prompts/stakeholder_summary.txt`): Translates the technical diagnosis into plain language for a non-technical audience.

**Output**: A short paragraph.

### Ranking consistency fix

Because the re-ranking call and the diagnosis call are independent, they can disagree (e.g., re-ranking leaves a downstream symptom at #1, but diagnosis correctly identifies a different step as the root cause). TraceLens may promote the diagnosed `root_cause_step_id` to rank #1 **unless** a Hit@k guard already locked a different step (see [`docs/RANKING_ARBITRATION.md`](docs/RANKING_ARBITRATION.md)). When promotion applies, the ranking mode is `llm+diagnosis`.

### LLM safeguards (noise + anchors + Hit@k)

| Safeguard | Purpose |
|-----------|---------|
| **Telemetry noise filter** | Strips ad/iframe/manifest/WebSocket/console noise before scoring and before sending to the LLM |
| **`page_load_noise_only`** | Flags step 0 homepage loads with only third-party churn; errors stripped from compact payload |
| **`heuristic_rank` in rerank prompt** | LLM sees where the automated ranker placed each step |
| **Local screenshot scan (`--llm`)** | Full-trace pixel diff + visual-causal attribution (same local pipeline as `--vlm`, no VLM API) |
| **`screenshots_available` flag** | Steps without pass/fail PNGs skip pixel/visual rules; marked `[no screenshots]` in reports |
| **Hybrid rerank pool (`pre_llm_k: 15` + inject)** | Top-15 by score plus any strong-signal step the heuristic buried |
| **Anchor guard (post-rerank)** | Steps in heuristic top-5 with `action_changed`, `wrong_navigation`, causal links, pixel ≥ 0.65, or visual-causal root stay in final top-5 |
| **Hit@k guard (shared)** | `src/ranking_arbitrator.py` — used by LLM and VLM: causal action over observer pixel, text-action anchor, visible symptom, pixel/VC leader, pre-heuristic anchor; VLM adds VLM-confirm tiers |
| **Deterministic promote (`--llm`)** | `wrong_navigation` or `text_causal` after guard; `text_causal` skipped when verify step has strong observer pixel (≥ 0.95) — see [`docs/RANKING_ARBITRATION.md`](docs/RANKING_ARBITRATION.md) |

All guard and gate decisions appear in the report under **RANKING DECISIONS (rules applied)**.

Set `pre_llm_k: 0` in `config.yaml` to send **all steps** to the reranker (max recall, higher token cost, more step-0 reshuffle risk).

### OpenRouter prompt caching

When `base_url` points at OpenRouter and `model.prompt_cache` is `true` (default), each LLM call splits the prompt into:

1. **Static** — full text of `prompts/rerank_steps.txt`, `root_cause.txt`, or `stakeholder_summary.txt` (marked with `cache_control: {type: ephemeral}`).
2. **Dynamic** — per-trace JSON (`steps_json` / `diagnosis_json`), uncached.

Across a batch run, calls 2 and 3 on the same trace (and rerank on later traces that share the same prompt file) can reuse cached prefix tokens. OpenRouter reports hits in `usage.prompt_tokens_details.cached_tokens`; TraceLens prints `[prompt cache] N cached prompt tokens` when present.

Per-trace **`session_id`** (`source/trace_id`) is sent on rerank, diagnosis, and summary so OpenRouter can use [sticky routing](https://openrouter.ai/docs/guides/best-practices/prompt-caching#provider-sticky-routing) toward the same provider endpoint.

| Setting | Effect |
|---------|--------|
| `prompt_cache: true` | Enabled automatically on `openrouter.ai` URLs; can force on/off explicitly |
| `prompt_cache: false` | Single plain user message (use for Groq/Gemini or debugging) |
| Free `:free` models | Caching depends on the upstream provider; many support implicit prefix cache above ~1k tokens |

Caching applies to **LLM text calls only** (three per trace with `--llm`). VLM per-step image calls do not share this prefix.

### Rate limiting

Free-tier providers enforce request and token rate limits. TraceLens handles this with:
- **Indefinite exponential backoff** on 429 errors and empty responses: 10s → 20s → 40s → 80s → 120s (capped), retrying until the call succeeds
- Empty-response (200 OK but no `choices`) treated identically to a 429 — retried, not crashed
- 2-second pause between the three calls within a trace
- 5-second inter-trace pause when running all traces
- Only non-retriable errors (auth failure, bad request) cause fallback to heuristic mode

---

## 6. VLM Visual Analysis (Phase 2)

The LLM reads **text signals** (actions, network errors, console logs). The VLM reads **visual signals** (what the browser actually displayed). Some faults — wrong page loaded, missing UI element, stale CDN cache, scroll position unchanged — produce no text signal at all. These are only detectable by comparing screenshots.

All 22 evaluation traces include one screenshot per step for both pass and fail runs. Screenshots are resolved at runtime by `src/screenshot_resolver.py`, which handles three different folder layouts across the data sources.

### How LLM + VLM combine

The two models run **sequentially**:

1. **Local screenshot scan** (pixel + visual-causal) runs for both `--llm` and `--vlm`.
2. **LLM** re-ranks from text + visual-causal fields in the slim payload, then diagnosis.
3. **VLM** (if `--vlm`) scores screenshot pairs for top-K candidates.
4. **Unified ensemble** (`_vlm_step_score` + Hit@k guards) produces the final ranking:
   - **`--vlm` only**: VLM + pixel + visual-causal (+ downstream symptom penalty).
   - **`--llm --vlm`**: same visual score blended with LLM rank position via `ensemble_vlm_weight` (default **0.4** → 40% visual formula / 60% LLM position prior).

Hit@k guard tiers are defined in **`src/ranking_arbitrator.py`** (same order for LLM-only and VLM; VLM path requires VLM score ≥ 0.5 on visual tiers). Full precedence table: [`docs/RANKING_ARBITRATION.md`](docs/RANKING_ARBITRATION.md).

### VLM failure handling

| Mode | VLM fails | Report | Aggregate Hit@k |
|------|-----------|--------|-----------------|
| `--vlm` only | yes | Saved with **VLM STATUS: FAILED** + heuristic fallback | Trace **excluded** (`excluded_traces` in run JSON) |
| `--llm --vlm` | yes | Saved with LLM fallback ranking | **Included** (degraded run) |

### When to use each mode

| Scenario | Recommended command |
|----------|---------------------|
| Quick/offline testing, no API | `python main.py` |
| Text-visible faults | `python main.py --llm` |
| Screenshot-only faults (`saucedemo_2`, `pypi`, `bbc`) | `python main.py --llm --vlm` |
| Full 22-trace evaluation (paper/benchmark) | `python main.py --llm --vlm` |
| Debug VLM in isolation | `python main.py --vlm` |

```bash
# Recommended final evaluation run
python main.py --llm --vlm

# Single visual-heavy trace
python main.py --source efe_irem --trace saucedemo_2 --llm --vlm
```

For screenshot path layouts, tuning `ensemble_vlm_weight`, and rate-limit guidance, see [`docs/VLM.md`](docs/VLM.md). For which pipeline stages run under each CLI flag, see [`docs/MODES.md`](docs/MODES.md).

---

## 7. Evaluation Metrics

Ground truth is stored in `data/evaluation/ground_truth.json` as a 0-indexed `fault_step` per trace.

| Metric | Definition |
|--------|-----------|
| **Hit@k** | 1 if the actual fault step appears within our top-k predicted steps, else 0. Reported for k=1, 3, 5. |
| **Rank position** | The 1-based position of the actual fault step in our ranked list. `-1` (displayed as "not in top-5") if it does not appear in the top-5. |
| **Rank distance** | `\|predicted rank − actual rank\|` — how many ranks away from #1 the actual fault landed. 0 = perfect. |
| **Step distance (top-1)** | `\|predicted_rank1_step_id − actual_fault_step_id\|` — how many steps in the trace our #1 prediction missed by. |
| **MAD@5** | Mean absolute step distance over the top-5 predictions. Lower = top-5 are collectively closer to the fault in the trace. |

### Per-rank breakdown table

Every report includes a table showing both metrics for each of the top-5 predictions:

```
  Pred.Rank    Step   True Rank   rank_dist   step_dist
  (our rank)          (1=fault)  |pred-true|  |step-actual|
  ----------  -----  ----------  ----------  ----------
  1               2         N/A         N/A           5
  2               7           1           1           0  ← ACTUAL FAULT
  3               1         N/A         N/A           6
  4               4         N/A         N/A           3
  5               0         N/A         N/A           7
```

- **True Rank** is `1` only for the actual fault step; all other steps have unknown true rank.
- **rank_dist** is therefore only meaningful (non-N/A) on the row where the actual fault appears.
- **step_dist** is meaningful for every row.

---

## 8. Data Sources

Three data formats are supported, sourced from three prior projects provided by the course instructor (each collected with different tooling):

| Source key | Format | Pass/Fail files | Logs location |
|------------|--------|-----------------|---------------|
| `efe_irem` | `correct.json` / `incorrect.json` | One JSON per trace, steps embedded | Per-step `console_logs` / `network_logs` arrays inside JSON |
| `areeb_salem` | `trace_pass.json` / `trace_fail.json` | JSON steps + external `.txt` log files | Separate `*_console_*.txt` and `*_network_*.txt` files per step |
| `ersel` | `steps.json` (fail only, pass inferred) | Single `steps.json` with `intent` field | Global `global_console_logs.json` / `global_network_logs.json` (session-level, not per-step) |

**Selected traces**: 22 traces across all three sources are used for evaluation, selected to cover a range of fault types. Full selection rationale (including excluded traces and reasons) is in [`data/evaluation/TRACE_SELECTION.md`](data/evaluation/TRACE_SELECTION.md).

| Fault type | Example |
|-----------|---------|
| HTTP 4xx error (404, 403, 405) | `efe_irem/gutenberg`, `efe_irem/webmd` |
| Authentication failure (wrong password / 401) | `efe_irem/saucedemo_1` |
| Network disconnection | `efe_irem/dictionary` |
| UI rendering crash | `areeb_salem/github` |
| Auth-gated action | `areeb_salem/amazon` |
| Invalid search input | `ersel/opencart_purchase_40` |
| Screenshot-only faults | `areeb_salem/bbc` (stale CDN cache) |

---

## 9. Project Structure

```
TraceLens-Automated-Debugging/
├── main.py                      # Entrypoint — runs the full pipeline
├── config.yaml                  # All paths, model settings, trace list, weights
├── requirements.txt
├── .env                         # API keys (not committed)
│
├── src/
│   ├── trace_parser.py          # Raw → unified step schema
│   ├── trace_aligner.py         # Align pass/fail steps
│   ├── anomaly_detector.py      # Per-step scoring (4 signals)
│   ├── ranker.py                # Heuristic ranking + LLM re-ranking apply
│   ├── ranking_arbitrator.py    # Shared Hit@1 guard + deterministic/diagnosis gates
│   ├── llm_reasoner.py          # LLM API calls (re-rank, diagnose, summarise)
│   ├── vlm_reasoner.py          # VLM screenshot comparison + ensemble merge
│   ├── screenshot_resolver.py   # Pass/fail screenshot path resolution per source
│   ├── evaluation.py            # Metrics computation
│   └── report_generator.py      # Build + save JSON/text reports
│
├── prompts/
│   ├── rerank_steps.txt         # LLM prompt: re-rank candidates
│   ├── root_cause.txt           # LLM prompt: technical diagnosis
│   ├── stakeholder_summary.txt  # LLM prompt: plain-language summary
│   └── vlm_visual_rank.txt      # VLM prompt: screenshot pair comparison
│
├── data/
│   ├── efe_irem/                # Raw traces (correct.json / incorrect.json)
│   ├── areeb_salem/             # Raw traces (trace_pass/fail.json + .txt logs)
│   ├── ersel/                   # Raw traces (steps.json + global logs)
│   └── evaluation/
│       └── ground_truth.json    # 0-indexed actual fault step per trace
│
├── outputs/
│   ├── reports/                 # Per-trace .json and .txt diagnosis reports
│   ├── rankings/                # Per-trace ranked step lists
│   └── metrics/                 # aggregate.json + per_trace.json
│
└── notebooks/
    └── experiments.ipynb        # Interactive pipeline exploration
```

---

## 10. Setup & Running

### Install dependencies

```bash
pip install -r requirements.txt
```

### Set up API key

Create a `.env` file in the project root:

```
OPENROUTER_API_KEY=your_key_here
```

Get a free key at [openrouter.ai/keys](https://openrouter.ai/keys). The active provider and which env variable to read are set via `api_key_env` in `config.yaml`. See [`docs/CONFIG.md`](docs/CONFIG.md) for switching to Groq or Gemini.

### Run a single trace (with LLM)

```bash
python main.py --source efe_irem --trace gutenberg --llm
```

### Run a single trace (heuristic only, no API needed)

```bash
python main.py --source efe_irem --trace gutenberg
```

### Run all 22 traces (LLM + VLM — recommended)

```bash
python main.py --llm --vlm
```

### Run all 22 traces (LLM only)

```bash
python main.py --llm
```

> **Note on rate limits**: `--llm` makes ~66 API calls (3 per trace). `--llm --vlm` adds ~5 VLM calls per trace (~176 total). VLM calls are image-heavy and slower. TraceLens retries indefinitely with exponential backoff on 429. Split long runs with `--skip` or `--from` if needed. See [`docs/VLM.md`](docs/VLM.md) for details.

### Run all traces for one source

```bash
python main.py --source ersel
```

### Skip evaluation (no ground truth needed)

```bash
python main.py --source efe_irem --trace gutenberg --no-eval
```

---

## 11. Configuration

`config.yaml` controls all pipeline parameters. For a full explanation of every field, valid values, and provider-switching instructions, see [`docs/CONFIG.md`](docs/CONFIG.md).

Key fields:

```yaml
model:
  base_url:    "https://openrouter.ai/api/v1"   # provider endpoint
  llm_model:   "openai/gpt-oss-120b:free"       # model ID
  api_key_env: "OPENROUTER_API_KEY"             # env var name to read from .env
  temperature: 0.1
  prompt_cache: true                             # OpenRouter static prompt caching (see §5)

vlm:                              # only used with --vlm flag
  vlm_model: "google/gemma-4-31b-it:free"
  ensemble_vlm_weight: 0.4        # used when both --llm and --vlm (60% LLM + 40% VLM)
  top_k_for_vlm: 5                # screenshot pairs sent to VLM per trace
  per_step: true                   # one VLM API call per screenshot pair

ranking:
  top_k:     5    # final ranked steps shown
  pre_llm_k: 15   # heuristic top-K for LLM rerank (+ strong-signal injects; 0 = all steps)
  heuristic_pixel_fallback: true  # pixel-diff boost in heuristic-only mode

weights:
  network: 0.35
  console: 0.25
  action:  0.15
  intent:  0.25   # auto-redistributed when no intent data present
```

---

## 12. Output Files

After running, the following files are created:

| Path | Contents |
|------|----------|
| `outputs/reports/{source}/{trace_id}.json` | Full structured report including scores, diagnosis, evaluation, per-rank table |
| `outputs/reports/{source}/{trace_id}.txt` | Human-readable text version of the same report |
| `outputs/rankings/{source}/{trace_id}.json` | Top-5 ranked steps only |
| `outputs/metrics/aggregate.json` | Mean metrics across all evaluated traces |
| `outputs/metrics/per_trace.json` | Per-trace metrics for all evaluated traces |
| `outputs/metrics/runs/llm_run_TIMESTAMP.json` | Full run report (`--llm`) |
| `outputs/metrics/runs/llm+vlm_run_TIMESTAMP.json` | Full run report (`--llm --vlm`) |
| `outputs/metrics/runs/vlm_run_TIMESTAMP.json` | Full run report (`--vlm` only) |
| `outputs/metrics/runs/heuristic_run_TIMESTAMP.json` | Full run report (no flags) |

### Report JSON structure

```json
{
  "trace_id": "gutenberg",
  "generated": "2026-05-21T...",
  "metadata": {
    "source": "efe_irem",
    "fault_type": "http_403",
    "ranking_decisions": ["LLM Hit@k guard: pixel leader step 7 ..."],
    "ran_screenshot_scan": true,
    "screenshots_available_count": 32
  },
  "ranking_mode": "llm+diagnosis",
  "ranked_suspicious_steps": [ ... ],
  "heuristic_steps": [ ... ],
  "llm_ranked_steps": [ ... ],
  "technical_diagnosis": {
    "root_cause_step_id": 8,
    "root_cause_summary": "403 Forbidden on ebook resource...",
    "failure_chain": "...",
    "downstream_steps": []
  },
  "stakeholder_summary": "When our system tried to access a specific book...",
  "visual_analysis": {
    "visual_root_cause_step_id": 8,
    "visual_summary": "Fail screenshot shows 403 error page instead of book content.",
    "visual_scores": [
      { "step_id": 8, "visual_score": 0.9, "visual_note": "Error page visible in fail, book page in pass" }
    ],
    "steps_with_screenshots": 5
  },
  "evaluation": {
    "actual_fault_step": 8,
    "hit@1": 1, "hit@3": 1, "hit@5": 1,
    "rank_position": 1,
    "rank_distance": 0,
    "top1_step_distance": 0,
    "mad@5": 5.2,
    "step_distances": [
      { "rank": 1, "step_id": 8, "step_distance": 0 },
      ...
    ]
  }
}
```
