# TraceLens — Install & Run

Short guide for installing dependencies and running the system. Full documentation: [`README.md`](README.md).

## Requirements

- **Python 3.10+** (tested on 3.11)
- Network access for LLM/VLM API calls (optional for heuristic-only mode)
- Trace dataset on disk (see [Data](#data) below)

## Dependencies

All Python packages are listed in [`requirements.txt`](requirements.txt):

| Package | Purpose |
|---------|---------|
| `openai` | OpenAI-compatible client (OpenRouter, Groq, Gemini) |
| `pyyaml` | `config.yaml` loading |
| `python-dotenv` | `.env` API keys |
| `Pillow` | Screenshot pixel diff |
| `numpy` | Image arrays |
| `matplotlib` | Evaluation figures (`scripts/plot_metrics.py`) |

## Install

```bash
cd TraceLens-Automated-Debugging
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## API keys

Copy `.env.example` → `.env` in the project root (never commit or zip `.env`):

```env
OPENROUTER_API_KEY=your_openrouter_key    # required for --llm
GROQ_API_KEY=your_groq_key                # required for --vlm (default config)
```

- OpenRouter (LLM): [openrouter.ai/keys](https://openrouter.ai/keys)
- Groq (VLM): [console.groq.com](https://console.groq.com/)

Heuristic-only runs need **no** API keys. Provider and model IDs are in `config.yaml`; see [`docs/CONFIG.md`](docs/CONFIG.md) to switch providers.

## Data

Raw pass/fail traces are **not** in git (`data/` is gitignored). Unpack the course archive so paths in `config.yaml` resolve. **Full layout** (every trace, JSON paths, screenshot naming): [`docs/DATA_LAYOUT.md`](docs/DATA_LAYOUT.md).

Summary:

```
data/relative debugging-20260518T094544Z-3-001/relative debugging/
  efe- irem traces/          # JSON files + screenshots (two layout variants)
  areeb & salem traces/      # pass/ + fail/ per site
  ersel-ikra-merve traces/   # passing/ + failing/ folders per trace
data/evaluation/ground_truth.json
```

Update `data.raw_base` in `config.yaml` if your folder layout differs.

## Submitting to instructor

See [`SUBMISSION.md`](SUBMISSION.md): exclude `.env` and `.venv`, include trace data + ground truth, optional frozen metrics.

## Run

### v2 pipeline (paper / benchmark — recommended)

Entry point: `main2.py`. Outputs under `outputs/v2/`.

```bash
# All 22 traces, full multimodal evaluation
python main2.py --llm --vlm

# Single trace, no API (heuristic baseline)
python main2.py --source efe_irem --trace gutenberg

# One source
python main2.py --source areeb_salem --llm
```

### v1 pipeline (legacy guards + promoters)

Entry point: `main.py`. Outputs under `outputs/`. Same CLI flags.

```bash
python main.py --source efe_irem --trace gutenberg --llm
python main.py --llm --vlm
```

### Common flags

| Flag | Effect |
|------|--------|
| `--llm` | LLM reranking + diagnosis |
| `--vlm` | VLM screenshot analysis |
| `--no-pixel` | Disable local screenshot diff |
| `--no-eval` | Skip ground-truth metrics |
| `--source NAME` | Run one collection (`efe_irem`, `areeb_salem`, `ersel`) |
| `--trace ID` | Run one trace (comma-separated list allowed) |

### Regenerate paper figures

```bash
python scripts/plot_metrics.py
```

Uses frozen metrics in `scripts/metrics_manifest.yaml`. See [`paper/README.md`](paper/README.md).

## Outputs

| Pipeline | Reports | Metrics |
|----------|---------|---------|
| v2 (`main2.py`) | `outputs/v2/reports/` | `outputs/v2/metrics/` |
| v1 (`main.py`) | `outputs/reports/` | `outputs/metrics/` |

Per-trace JSON + `.txt` reports; aggregate run JSON under `metrics/runs/`.
