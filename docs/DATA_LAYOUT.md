# Trace Data Layout

This document describes **where trace files must live on disk** so `main.py` / `main2.py` can find them without code changes. The pipeline reads paths from `config.yaml`; your folder names must match (or you update `config.yaml`).

## How the pipeline resolves paths

1. **`data.raw_base`** in `config.yaml` is the root folder for all raw traces (default below).
2. Each **source** (`efe_irem`, `areeb_salem`, `ersel`) has a **`base`** subfolder under that root.
3. Each **trace** has **`pass`** and **`fail`** paths relative to `raw_base / base`.
4. **`ScreenshotResolver`** (`src/screenshot_resolver.py`) looks up PNGs using source-specific naming (see [Screenshots](#screenshots-by-source)).
5. **`data/processed/`** is created automatically at run time — you do not ship it unless you want cached parses.
6. **`data/evaluation/ground_truth.json`** holds the labeled fault step per trace for Hit@K metrics (0-indexed `fault_step`).

Default `raw_base` (from `config.yaml`):

```
data/relative debugging-20260518T094544Z-3-001/relative debugging/
```

If you unpack the instructor archive elsewhere, either mirror this path under `data/` or change `data.raw_base` in `config.yaml`.

---

## Top-level tree (required)

```
data/
├── evaluation/
│   └── ground_truth.json          # evaluation labels (include in submission)
│
└── relative debugging-20260518T094544Z-3-001/
    └── relative debugging/
        ├── efe- irem traces/       # note: space after "efe-"
        ├── areeb & salem traces/     # note: spaces around "&"
        └── ersel-ikra-merve traces/
```

The three folder names **must match exactly** — they are configured in `config.yaml` under `data.sources.<source>.base`.

---

## Source 1: `efe_irem` (`efe- irem traces/`)

**Format:** One JSON file per pass/fail run. Logs are embedded in the JSON.

**Parser:** `parse_efe_irem(json_path)` — pass/fail are **files**, not directories.

### Per-trace layout

| Trace ID | Pass JSON | Fail JSON |
|----------|-----------|-----------|
| `saucedemo_1` | `saucedemo_1/saucedemo_1_correct.json` | `saucedemo_1/saucedemo_1_incorrect.json` |
| `saucedemo_2` | `saucedemo_2/saucedemo_2_correct.json` | `saucedemo_2/saucedemo_2_incorrect.json` |
| `gutenberg` | `gutenberg/gutenberg_correct.json` | `gutenberg/gutenberg_incorrect.json` |
| `dictionary` | `dictionary/dictionary-correct.json` | `dictionary/dictionary-incorrect.json` |
| `webmd` | `webmd/webmd-correct.json` | `webmd/webmd-incorrect.json` |
| `wolfram` | `wolfram mathworld/wolfram-correct.json` | `wolfram mathworld/wolfram-incorrect.json` |
| `elinguistics` | `elinguistics/elinguistics_correct.json` | `elinguistics/elinguistics_incorrect.json` |
| `wikipedia` | `wikipedia/wikipedia_correct.json` | `wikipedia/wikipedia_incorrect.json` |

Example (gutenberg):

```
efe- irem traces/
└── gutenberg/
    ├── gutenberg_correct.json
    ├── gutenberg_incorrect.json
    └── screenshots/                    # Layout A (or see Layout B below)
        ├── correct/
        │   ├── step_001_after.png
        │   └── ...
        └── incorrect/
            ├── step_001_after.png
            └── ...
```

### Screenshots (efe_irem)

Step index in filenames is **1-based**, **3-digit** padded: `step_NNN_after.png`.

Two folder layouts exist; the resolver tries **Layout A** first, then **Layout B**:

| Layout | Pass folder | Fail folder |
|--------|-------------|-------------|
| **A** | `{trace}/screenshots/correct/` | `{trace}/screenshots/incorrect/` |
| **B** | `{trace}/correct screenshots/` | `{trace}/incorrect screenshots/` |

Layout A: saucedemo, gutenberg, elinguistics. Layout B: dictionary, webmd, wikipedia, wolfram.

---

## Source 2: `areeb_salem` (`areeb & salem traces/`)

**Format:** `trace_pass.json` / `trace_fail.json` with paths to sidecar `.txt` logs and PNGs.

**Parser:** `parse_areeb_salem(json_path)` — JSON lives under `pass/` or `fail/`.

### Per-trace layout

Each site is a folder with parallel `pass/` and `fail/` trees:

```
areeb & salem traces/
└── amazon/
    ├── pass/
    │   ├── trace_pass.json
    │   ├── console/
    │   │   └── console_step_00.txt    # referenced from JSON
    │   ├── network/
    │   │   └── network_step_00.txt    # inferred: console path → network/
    │   └── screenshots/
    │       └── step_00.png            # 0-based, 2-digit
    └── fail/
        ├── trace_fail.json
        ├── console/
        ├── network/
        └── screenshots/
```

Configured traces: `amazon`, `bbc`, `github`, `hackernews`, `imdb`, `npm`, `producthunt`, `pypi`, `wikipedia`, `youtube` — each follows the same `pass/` + `fail/` pattern.

### Screenshots (areeb_salem)

- Path: `{trace}/pass/screenshots/step_NN.png` and `{trace}/fail/screenshots/step_NN.png`
- **0-based**, **2-digit** padding (`step_00.png`, `step_01.png`, …)

---

## Source 3: `ersel` (`ersel-ikra-merve traces/`)

**Format:** A **directory** per pass/fail run (not a single JSON path).

**Parser:** `parse_ersel(trace_dir)` — `pass`/`fail` in config point to `.../passing` or `.../failing` folders.

### Per-trace layout

```
ersel-ikra-merve traces/
└── opencart_purchase_40/
    ├── passing/
    │   ├── steps.json
    │   ├── global_console_logs.json   # optional
    │   ├── global_network_logs.json   # optional
    │   ├── step_0_prev.png            # optional
    │   ├── step_0_post.png
    │   └── ...
    └── failing/
        ├── steps.json
        ├── global_console_logs.json
        ├── global_network_logs.json
        └── step_*_post.png
```

Configured traces:

| Trace ID | Pass dir | Fail dir |
|----------|----------|----------|
| `opencart_purchase_40` | `opencart_purchase_40/passing` | `opencart_purchase_40/failing` |
| `opencart_wishlist_35` | `opencart_wishlist_35/passing` | `opencart_wishlist_35/failing` |
| `opencart_manufacturer_404_35` | `opencart_manufacturer_404_35/passing` | `opencart_manufacturer_404_35/failing` |
| `opencart_contact_35` | `opencart_contact_35/passing` | `opencart_contact_35/failing` |

### Screenshots (ersel)

- Path: `{trace}/passing/step_N_post.png` and `{trace}/failing/step_N_post.png`
- **0-based**, **no** zero-padding (`step_0_post.png`, `step_12_post.png`)
- Optional `step_N_prev.png` on disk; VLM/pixel scan primarily uses `_post.png`

---

## Ground truth

```
data/evaluation/ground_truth.json
```

Structure (conceptually):

```json
{
  "efe_irem": {
    "gutenberg": { "fault_step": 3, "fault_type": "..." }
  },
  "areeb_salem": { ... },
  "ersel": { ... }
}
```

Used when evaluation is enabled (default). Skip with `--no-eval`.

---

## Quick sanity check

After placing data, verify one trace resolves:

```bash
python main2.py --source efe_irem --trace gutenberg --no-eval
```

If you see `Parsed: N pass steps, M fail steps`, paths are correct. Missing files usually mean a typo in folder names (especially spaces in `efe- irem traces` and `areeb & salem traces`) or a wrong `raw_base`.

---

## Related docs

- [`docs/VLM.md`](VLM.md) — screenshot pairing and VLM behavior
- [`docs/CONFIG.md`](CONFIG.md) — changing `raw_base` or trace entries
- [`SUBMISSION.md`](../SUBMISSION.md) — what to include when zipping for submission
