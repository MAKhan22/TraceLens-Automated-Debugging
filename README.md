# TraceLens — Automated Regression Failure Diagnosis

TraceLens is a two-stage automated debugging system that takes a **passing trace** and a **failing trace** of the same test case, identifies the most suspicious steps, ranks them, and produces a natural-language explanation of the root cause for both engineers and non-technical stakeholders.

---

## Table of Contents

1. [Background & Problem](#1-background--problem)
2. [Architecture Overview](#2-architecture-overview)
3. [Pipeline Stages](#3-pipeline-stages)
4. [Scoring System](#4-scoring-system)
5. [LLM Re-ranking & Diagnosis](#5-llm-re-ranking--diagnosis)
6. [Evaluation Metrics](#6-evaluation-metrics)
7. [Data Sources](#7-data-sources)
8. [Project Structure](#8-project-structure)
9. [Setup & Running](#9-setup--running)
10. [Configuration](#10-configuration)
11. [Output Files](#11-output-files)

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
                                                       │           heuristic top-10
                                                       │                │
                                                       └──────────► LlmReasoner ──► ReportGenerator
                                                                   (re-rank + diagnose)      │
                                                                         │              ┌────┴────┐
                                                                    Evaluation     JSON report  Text report
```

There are two execution paths:

| Mode | What runs |
|------|-----------|
| `--no-llm` | Heuristic scoring + ranking only. Fast, no API needed. |
| Default (with LLM) | Heuristic scoring → LLM re-ranking → LLM diagnosis + stakeholder summary. |

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
  "screenshot":   str | None,  # relative path (Phase 2 / VLM, not yet active)
}
```

Three source formats are supported — see [Data Sources](#7-data-sources).

### Stage 2 — Trace Alignment (`src/trace_aligner.py`)

Pairs up steps from the pass and fail traces so they can be compared directly.

- **Same-length traces**: aligned 1:1 by index (`align_by_index`).
- **Different-length traces**: aligned by action text similarity (`align_by_similarity`), which handles inserted/deleted steps caused by the fault.

### Stage 3 — Anomaly Scoring (`src/anomaly_detector.py`)

Each aligned step pair receives four component scores (all in `[0, 1]`) and one `combined_score`. See [Scoring System](#4-scoring-system).

### Stage 4 — Heuristic Ranking (`src/ranker.py`)

Steps are sorted descending by `combined_score`. The top-10 are passed to the LLM as **candidates**; the top-5 are kept as the **heuristic ranking** (fallback if LLM is disabled).

### Stage 5 — LLM Re-ranking & Diagnosis (`src/llm_reasoner.py`)

See [LLM Re-ranking & Diagnosis](#5-llm-re-ranking--diagnosis).

### Stage 6 — Evaluation (`src/evaluation.py`)

Compares predicted rankings against ground truth (`data/evaluation/ground_truth.json`). See [Evaluation Metrics](#6-evaluation-metrics).

### Stage 7 — Report Generation (`src/report_generator.py`)

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

> **Important**: Scores are **only used for heuristic ranking** (stage 4). The LLM in stage 5 does not see the numerical scores — it sees the actual step content (action text, network diffs, console errors). The scores determine *which* steps get elevated as candidates for the LLM; the LLM independently reasons about causality.

---

## 5. LLM Re-ranking & Diagnosis

The LLM pipeline runs three sequential calls, all via the [Groq](https://groq.com) API (free tier, fast inference):

### Call 1 — Re-ranking

**Input**: Top-10 heuristic candidates, each with action text, network error diff, and console error diff (raw content, not scores).

**Prompt** (`prompts/rerank_steps.txt`): Instructs the LLM to reason about which steps are root causes vs downstream symptoms, and return a JSON-ordered list of step IDs.

**Output**: Re-ordered list of step IDs.

### Call 2 — Root Cause Diagnosis

**Input**: Top-5 re-ranked steps (same content format).

**Prompt** (`prompts/root_cause.txt`): Asks for the primary fault step, a technical summary, the failure chain, and downstream affected steps.

**Output**: Structured JSON with `root_cause_step_id`, `root_cause_summary`, `failure_chain`, `downstream_steps`.

### Call 3 — Stakeholder Summary

**Input**: The technical diagnosis from Call 2.

**Prompt** (`prompts/stakeholder_summary.txt`): Translates the technical diagnosis into plain language for a non-technical audience.

**Output**: A short paragraph.

### Ranking consistency fix

Because the re-ranking call and the diagnosis call are independent, they can disagree (e.g., re-ranking leaves a downstream symptom at #1, but diagnosis correctly identifies a different step as the root cause). TraceLens resolves this by **promoting the diagnosed `root_cause_step_id` to rank #1** after all three calls complete. When this promotion changes the order, the report shows both the original heuristic ranking and the final adjusted ranking side by side, and the ranking mode label is set to `llm+diagnosis`.

### Rate limiting

The Groq free tier enforces request-per-minute limits. TraceLens handles this with:
- Exponential backoff on 429 errors: 10s → 20s → 40s → 80s (up to 5 retries)
- 2-second pause between the three calls within a trace
- 5-second inter-trace pause when running all traces

---

## 6. Evaluation Metrics

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

## 7. Data Sources

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

## 8. Project Structure

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
│   ├── llm_reasoner.py          # Groq API calls (re-rank, diagnose, summarise)
│   ├── evaluation.py            # Metrics computation
│   ├── report_generator.py      # Build + save JSON/text reports
│   └── vlm_reasoner.py          # Placeholder for Phase 2 screenshot analysis
│
├── prompts/
│   ├── rerank_steps.txt         # LLM prompt: re-rank candidates
│   ├── root_cause.txt           # LLM prompt: technical diagnosis
│   └── stakeholder_summary.txt  # LLM prompt: plain-language summary
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

## 9. Setup & Running

### Install dependencies

```bash
pip install -r requirements.txt
```

### Set up API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

Get a free key at [console.groq.com](https://console.groq.com).

### Run a single trace (with LLM)

```bash
python main.py --source efe_irem --trace gutenberg
```

### Run a single trace (heuristic only, no API needed)

```bash
python main.py --source efe_irem --trace gutenberg --no-llm
```

### Run all 22 traces

```bash
python main.py
```

> **Note on rate limits**: Running all 22 traces with the LLM makes ~66 API calls (3 per trace). The free Groq tier may throttle these. TraceLens handles this automatically with retry + inter-trace delays, but the full run takes ~10–15 minutes.

### Run all traces for one source

```bash
python main.py --source ersel
```

### Skip evaluation (no ground truth needed)

```bash
python main.py --source efe_irem --trace gutenberg --no-eval
```

---

## 10. Configuration

`config.yaml` controls all pipeline parameters:

```yaml
model:
  llm_model:   "llama3-8b-8192"   # Groq model name
  temperature: 0.1
  base_url:    "https://api.groq.com/openai/v1"

ranking:
  top_k:            5    # final ranked steps shown
  llm_candidates:   10   # steps sent to LLM for re-ranking

weights:
  network: 0.35
  console: 0.25
  action:  0.15
  intent:  0.25          # auto-redistributed when no intent data present

outputs:
  reports:  "outputs/reports"
  rankings: "outputs/rankings"
  metrics:  "outputs/metrics"
```

---

## 11. Output Files

After running, the following files are created:

| Path | Contents |
|------|----------|
| `outputs/reports/{source}/{trace_id}.json` | Full structured report including scores, diagnosis, evaluation, per-rank table |
| `outputs/reports/{source}/{trace_id}.txt` | Human-readable text version of the same report |
| `outputs/rankings/{source}/{trace_id}.json` | Top-5 ranked steps only |
| `outputs/metrics/aggregate.json` | Mean metrics across all evaluated traces |
| `outputs/metrics/per_trace.json` | Per-trace metrics for all evaluated traces |

### Report JSON structure

```json
{
  "trace_id": "gutenberg",
  "generated": "2026-05-21T...",
  "metadata": { "source": "efe_irem", "fault_type": "http_403" },
  "ranking_mode": "llm+diagnosis",
  "ranked_suspicious_steps": [ ... ],
  "heuristic_steps": [ ... ],
  "technical_diagnosis": {
    "root_cause_step_id": 8,
    "root_cause_summary": "403 Forbidden on ebook resource...",
    "failure_chain": "...",
    "downstream_steps": []
  },
  "stakeholder_summary": "When our system tried to access a specific book...",
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
