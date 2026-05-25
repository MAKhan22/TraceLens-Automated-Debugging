# TraceLens — Configuration Reference

All pipeline behaviour is controlled by `config.yaml` in the project root. This document explains every field, its valid values, and the effect of changing it.

---

## `data` section

```yaml
data:
  raw_base: "data/relative debugging-20260518T094544Z-3-001/relative debugging"
  processed_dir: "data/processed"
  ground_truth: "data/evaluation/ground_truth.json"
  sources:
    efe_irem:   { base: "...", traces: [...] }
    areeb_salem: { base: "...", traces: [...] }
    ersel:       { base: "...", traces: [...] }
```

| Field | Description |
|-------|-------------|
| `raw_base` | Root directory of the raw trace data folder (relative to project root) |
| `processed_dir` | Where parsed/cached data is written (currently unused; reserved for caching) |
| `ground_truth` | Path to the JSON file with per-trace `fault_step` ground truth labels |
| `sources` | Named groups of traces. Each source has a `base` subdirectory and a list of `{ id, pass, fail }` trace entries |

**Adding a new trace**: Add an entry under the appropriate source in `sources`. The `pass` and `fail` paths are relative to `data/<raw_base>/<source_base>/`.

---

## `model` section

Used when `--llm` is passed. Requires `OPENROUTER_API_KEY` (or the provider set in `api_key_env`) in `.env`.

```yaml
model:
  base_url:    "https://openrouter.ai/api/v1"
  llm_model:   "openai/gpt-oss-120b:free"
  api_key_env: "OPENROUTER_API_KEY"
  temperature: 0.1
```

| Field | Description |
|-------|-------------|
| `base_url` | API endpoint. Use `https://openrouter.ai/api/v1` for OpenRouter, `https://api.groq.com/openai/v1` for Groq, or `https://generativelanguage.googleapis.com/v1beta/openai/` for Gemini. |
| `llm_model` | Model identifier as accepted by the provider. See table below. |
| `api_key_env` | Name of the environment variable that holds the API key (read from `.env`). Changing this lets you switch providers without editing code. |
| `temperature` | LLM sampling temperature. `0.1` keeps outputs deterministic and JSON-safe. Do not go above `0.3` for structured output tasks. |

### Supported / tested providers

| Provider | `base_url` | `api_key_env` | Notes |
|----------|-----------|----------------|-------|
| **OpenRouter** (active) | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` | Aggregates many model backends. Free models marked `:free`. |
| Groq | `https://api.groq.com/openai/v1` | `GROQ_API_KEY` | Fast inference. Free tier: 100k tokens/day — exhausts after ~1.5 full runs across all 22 traces. |
| Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` | `GEMINI_API_KEY` | Free tier trains on your data. Known issues with API key configuration in free projects. |

### Recommended free models on OpenRouter

| Model ID | Backend | Context | Strength |
|----------|---------|---------|----------|
| `openai/gpt-oss-120b:free` | OpenAI OSS | 131K | Fast, clean JSON output. **Currently recommended.** |
| `qwen/qwen3-coder:free` | Venice | 1M | Strong reasoning, but Venice backend congests under load |
| `qwen/qwen3-next-80b-a3b-instruct:free` | Venice | 262K | Best reasoning quality when Venice is not congested |
| `google/gemma-4-31b-it:free` | — | 262K | Good JSON adherence, reliable fallback |
| `nvidia/nemotron-3-super-120b-a12b:free` | — | 1M | Large context, good reasoning |
| `meta-llama/llama-3.3-70b-instruct:free` | — | 131K | Reliable fallback; ~200 req/day limit |

To switch provider, comment out the active block and uncomment the desired one. The `config.yaml` file has all alternatives pre-commented for easy swapping.

### Rate limiting behaviour

`llm_reasoner.py` retries indefinitely on 429 errors and empty responses:

```
wait = min(2^attempt × 10s, 120s)   →   10s → 20s → 40s → 80s → 120s → 120s → ...
```

Only non-retriable errors (auth failure, bad request, malformed prompt) cause the LLM call to fail and fall back to heuristic mode. Ctrl+C always exits.

---

## `vlm` section

Used only when `--vlm` is passed on the command line. See [`docs/VLM.md`](VLM.md) for the full integration guide.

```yaml
vlm:
  base_url:    "https://openrouter.ai/api/v1"
  vlm_model:   "google/gemma-4-31b-it:free"
  api_key_env: "OPENROUTER_API_KEY"
  temperature: 0.1
  per_step: true
  ensemble_vlm_weight: 0.4
  top_k_for_vlm: 5
```

| Field | Default | Description |
|-------|---------|-------------|
| `base_url` | OpenRouter | Same OpenAI-compatible endpoint as the LLM. VLM and LLM can share one API key. |
| `vlm_model` | `google/gemma-4-31b-it:free` | OpenRouter vision model. Paid alternative: `qwen/qwen2.5-vl-72b-instruct`. |
| `api_key_env` | `OPENROUTER_API_KEY` | Env var read from `.env`. |
| `temperature` | `0.1` | Keep low for structured JSON output. |
| `per_step` | `true` | One API call per screenshot pair (more reliable than batching all pairs). |
| `ensemble_vlm_weight` | `0.4` | Used when both `--llm` and `--vlm`. 60% LLM position score + 40% VLM visual score. |
| `top_k_for_vlm` | `5` | Top-ranked steps whose pass/fail screenshot pairs are sent to the VLM. |

---

## `ranking` section

```yaml
ranking:
  top_k:     5
  pre_llm_k: 0
  heuristic_pixel_fallback: true
  heuristic_pixel_weight: 0.35
```

| Field | Default | Description |
|-------|---------|-------------|
| `top_k` | `5` | Number of steps in the final ranked output. Also the number shown in the per-rank breakdown table. |
| `pre_llm_k` | `0` | How many heuristic-sorted steps to send to the LLM when `--llm` is passed. **`0` means all steps**. |
| `heuristic_pixel_fallback` | `true` | In heuristic-only mode, boost ranking with pixel diff (override with `--no-pixel`). |
| `heuristic_pixel_weight` | `0.35` | Blend weight for pixel diff in heuristic-only mode. |

### Why `pre_llm_k: 0` matters

With `pre_llm_k > 0`, the LLM only re-ranks the top-K heuristic candidates. If the actual fault step is ranked outside the top-K by heuristics (e.g., it has a low score because it produces no network/console signal), the LLM **cannot discover it** — it simply never sees it.

With `pre_llm_k: 0`, all steps are passed to the LLM in slimmed form (action text + error-only network/console diffs). Each step is ~10–15 fields. For a 36-step trace this adds ~800–1200 tokens per call, which is negligible on the models listed above.

**Trade-off**: More tokens per call → slightly slower and marginally higher chance of hitting token-based rate limits. For free-tier usage across 22 traces, this is acceptable.

**If you want to cap it** (e.g., to test the effect of heuristic pre-filtering):
```yaml
pre_llm_k: 10   # only top-10 heuristic candidates go to LLM
pre_llm_k: 15   # slightly more headroom
```

---

## `weights` section

```yaml
weights:
  network: 0.35
  console: 0.25
  action:  0.15
  intent:  0.25
```

These control the `combined_score` formula used for **heuristic ranking only** (scores are not passed to the LLM):

```
combined = w_net × network_score + w_con × console_score
         + w_act × action_score  + w_int × intent_score
```

All four weights must sum to `1.0`.

| Signal | Default weight | Rationale |
|--------|---------------|-----------|
| `network` | 0.35 | HTTP errors and network anomalies are the most explicit fault indicators |
| `console` | 0.25 | JS errors and `net::ERR_*` messages are reliable but can be noisy |
| `action` | 0.15 | Action text divergence often reflects downstream symptoms more than root causes |
| `intent` | 0.25 | When present, the intent field is a direct oracle (e.g. "Verification failed"). Only present in ersel traces. |

### Auto-redistribution when `intent` is absent

When none of the steps in a trace have an `intent` field (all efe_irem and areeb_salem traces), the `intent` weight is automatically redistributed proportionally to the other three signals at runtime. This ensures `combined_score` still spans `[0, 1]`:

```
# With intent absent, effective weights become:
network: ≈0.467   (0.35 + 0.25 × 0.35/0.75)
console: ≈0.333   (0.25 + 0.25 × 0.25/0.75)
action:  ≈0.200   (0.15 + 0.25 × 0.15/0.75)
intent:  0.000
```

This redistribution is done in `anomaly_detector.py` and does not require any config change.

---

## `outputs` section

```yaml
outputs:
  rankings: "outputs/rankings"
  reports:  "outputs/reports"
  metrics:  "outputs/metrics"
```

| Field | Default | Description |
|-------|---------|-------------|
| `rankings` | `outputs/rankings` | Per-trace ranked step lists (`{source}/{trace_id}.json`) |
| `reports` | `outputs/reports` | Per-trace full diagnosis reports (`.json` + `.txt`) |
| `metrics` | `outputs/metrics` | `aggregate.json`, `per_trace.json`, and `runs/` timestamped run reports |

### Run reports

Each full pipeline run saves a timestamped JSON to `outputs/metrics/runs/`:

```
outputs/metrics/runs/
  llm_run_20260522_181805.json        ← LLM mode
  llm+vlm_run_20260522_181805.json    ← LLM + VLM mode (--llm --vlm)
  vlm_run_20260522_181805.json        ← VLM only (--vlm)
  heuristic_run_20260522_145029.json  ← heuristic-only (no flags)
```

The filename prefix reflects the run mode:
- `llm_run_` — `--llm` only
- `llm+vlm_run_` — `--llm --vlm`
- `vlm_run_` — `--vlm` only
- `heuristic_run_` — no flags

Each file contains `aggregate` metrics (Hit@k, rank distance, MAD@5) and a `per_trace` breakdown.

---

## Switching between modes (quick reference)

TraceLens uses two independent opt-in flags: `--llm` and `--vlm`. No flags = heuristic only.

| Command | Mode |
|---------|------|
| `python main.py` | Heuristic only (default) |
| `python main.py --no-pixel` | Heuristic only, text signals only (no pixel boost) |
| `python main.py --llm` | LLM re-rank + diagnosis |
| `python main.py --vlm` | VLM visual analysis only |
| `python main.py --llm --vlm` | Hybrid (60% LLM / 40% VLM) |

### Run with LLM + VLM (recommended for full evaluation)
```bash
python main.py --llm --vlm
```
Runs heuristic → LLM → VLM → ensemble merge. Saves `llm+vlm_run_TIMESTAMP.json`.

### Run with LLM only
```bash
python main.py --llm
```
Uses `model.llm_model` from config. LLM re-ranks all steps and diagnoses the root cause.

### Run VLM only (no text LLM)
```bash
python main.py --vlm
```
Heuristic ranking → VLM visual analysis on heuristic top-5. Useful for debugging visual-only faults.

### Run heuristic only (no API needed)
```bash
python main.py
```
No API calls. Pixel-diff boost on top-5 screenshots is **on by default**; pass `--no-pixel` for pure text-signal ranking.

### Run a single trace
```bash
python main.py --source efe_irem --trace gutenberg
```

### Run all traces for one source
```bash
python main.py --source areeb_salem
```

### Skip evaluation (no ground truth file needed)
```bash
python main.py --no-eval
```
