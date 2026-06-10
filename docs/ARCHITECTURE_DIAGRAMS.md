# TraceLens Architecture Diagrams (v1 & v2)

Publication-ready pipeline figures for the ICSE paper. Render with any Mermaid viewer, GitHub, or export via [mermaid.live](https://mermaid.live).

**Task framing:** *Relative debugging* — given a **pass** and **fail** trace of the same UI test, localize the earliest step whose divergence explains the regression.

---

## Diagram A — v1 pipeline (`main.py`)

Stacked stages with **multiple ranking overrides**. Hybrid mode runs LLM first, then VLM, then a **second** arbitration pass — the source of trace-specific brittleness.

```mermaid
flowchart TB
    subgraph INPUT[" "]
        P[("Pass trace")]
        F[("Fail trace")]
    end

    subgraph INGEST["Ingestion & relative diff"]
        PARSE["TraceParser\n(unify 3 schemas)"]
        ALIGN["TraceAligner\n(index or action similarity)"]
        DETECT["AnomalyDetector\npass–fail signal diff"]
    end

    subgraph HEUR["Heuristic ranking"]
        RANK["Ranker\nsort by combined_score"]
        POOL["Candidate pool\ntop-15 + injected causal steps"]
        PIX_LOCAL["Local pixel scan\n(top-5 report table)"]
    end

    subgraph LLM_PATH["Optional: --llm"]
        LLM_RR["LlmReasoner.rerank\ntext-only reorder"]
        LLM_GUARD["Hit@k guard\n(ranking_arbitrator)"]
        DET_PROM["Deterministic promote\n(wrong_nav, text_causal)"]
        DIAG["LLM diagnose\n(may promote #1)"]
    end

    subgraph VIS_PATH["Optional: --vlm"]
        VC["Visual-causal scan\nfull trace, local pixels"]
        VLM["VlmReasoner\npass/fail screenshot pairs"]
        ENS["Ensemble blend\n60% LLM + 40% VLM"]
        VLM_GUARD["Post-blend Hit@k guard\n(same arbitrator, vlm_path=True)"]
    end

    subgraph OUT["Output"]
        EVAL["Evaluation\nHit@k vs ground truth"]
        REP["ReportGenerator\nJSON + stakeholder text"]
    end

    P --> PARSE
    F --> PARSE
    PARSE --> ALIGN --> DETECT --> RANK --> POOL
    RANK --> PIX_LOCAL

    POOL -->|"--llm"| LLM_RR --> LLM_GUARD --> DET_PROM --> DIAG

    DIAG -->|"--vlm"| VC
    RANK -->|"--vlm only"| VC
    VC --> VLM --> ENS --> VLM_GUARD

    DIAG -->|"--llm only"| EVAL
    VLM_GUARD --> EVAL
    RANK -->|"heuristic only"| EVAL
    EVAL --> REP

    style LLM_GUARD fill:#ffe0e0
    style VLM_GUARD fill:#ffe0e0
    style DET_PROM fill:#ffe0e0
    style DIAG fill:#ffe0e0
```

### v1 mode summary

| Mode | Final rank decided by |
|------|------------------------|
| Heuristic | `Ranker` (+ pixel mutates rank if enabled) |
| `--llm` | LLM rerank → guards → deterministic → diagnosis |
| `--vlm` | Visual-causal re-rank → VLM → guards |
| `--llm --vlm` | LLM path **then** VLM ensemble **then** second guard pass |

**Red nodes** = stages that can **override** the previous ordering (fragmentation).

### Why v1 hybrid underperformed expectations

```mermaid
flowchart LR
    subgraph GOOD["Worked well in isolation"]
        H["Heuristic relative diff"]
        L["LLM causal rerank"]
        V["VLM + visual-causal"]
    end

    subgraph BAD["Hybrid pain point"]
        G1["Guard: promote action\n(hackernews)"]
        G2["Guard: lock verify pixel\n(youtube)"]
        G3["Guard: block spurious scroll\n(amazon)"]
        X["Sequential overrides\nfix A → break B"]
    end

    L --> G1
    V --> G2
    G1 --> X
    G2 --> X
    G3 --> X
```

Each guard encoded a **benchmark trace pattern**. Hybrid activated **both** LLM and VLM override paths, so conflicts surfaced more often than in single-modality modes.

---

## Diagram B — v2 pipeline (`main2.py`) — paper main figure

**Single fusion sort.** LLM and VLM contribute **soft priors**; diagnosis never changes rank.

```mermaid
flowchart TB
    subgraph INPUT[" "]
        P[("Pass trace")]
        F[("Fail trace")]
    end

    subgraph INGEST["Ingestion & relative diff"]
        PARSE["TraceParser"]
        ALIGN["TraceAligner"]
        DETECT["AnomalyDetector\n(relative signals)"]
    end

    subgraph FEATURES["Channel extraction (v2/features.py)"]
        CH_TEXT["text"]
        CH_NAV["navigation"]
        CH_CAUSAL["causal_action"]
        CH_OBS["observer_symptom"]
        CH_VC["visual_causal"]
        CH_PIX["pixel"]
        CH_LLM["llm_prior\n(if --llm)"]
        CH_VLM["vlm_prior\n(if --vlm)"]
    end

    subgraph PRIORS["Optional model priors"]
        LLM_RR["LLM rerank\n→ rank positions"]
        VC_SCAN["Visual-causal scan\n(if --vlm or --llm)"]
        VLM_API["VLM screenshot pairs\n(if --vlm)"]
    end

    subgraph FUSION["Unified fusion (v2/fusion.py)"]
        WEIGHT["Weighted sum\nweights from v2/config.yaml"]
        POLICY["Structural policies\nobserver cap, downstream penalty"]
        SORT["Single sort → top-k"]
    end

    subgraph OUT["Output"]
        DIAG["LLM diagnosis\n(read-only)"]
        EVAL["Evaluation"]
        REP["ReportGenerator"]
    end

    P --> PARSE
    F --> PARSE
    PARSE --> ALIGN --> DETECT

    DETECT --> CH_TEXT & CH_NAV & CH_CAUSAL & CH_OBS
    DETECT --> CH_PIX

    DETECT --> LLM_RR
    LLM_RR --> CH_LLM
    LLM_RR --> VC_SCAN
    VC_SCAN --> CH_VC
    VC_SCAN --> VLM_API
    VLM_API --> CH_VLM

    CH_TEXT & CH_NAV & CH_CAUSAL & CH_OBS & CH_VC & CH_PIX & CH_LLM & CH_VLM --> WEIGHT
    WEIGHT --> POLICY --> SORT

    SORT --> DIAG --> REP
    SORT --> EVAL --> REP

    style SORT fill:#e0ffe0
    style WEIGHT fill:#e0f0ff
```

### v2 mode summary

| Mode | Channels active | Ranking |
|------|-----------------|---------|
| Heuristic | text, nav, causal, observer, pixel | Fusion (no model priors) |
| `--llm` | + llm_prior | Same fusion function |
| `--vlm` | + visual_causal, vlm_prior | Same fusion function |
| `--llm --vlm` | All channels | Same fusion function |

**Green node** = exactly **one** final sort (no post-hoc promotes).

---

## Diagram C — Side-by-side (for paper appendix or slide)

```mermaid
flowchart TB
    subgraph V1["v1: sequential overrides"]
        direction TB
        v1a["Heuristic sort"] --> v1b["LLM hard rerank"]
        v1b --> v1c["Guards & promotes"]
        v1c --> v1d["VLM ensemble"]
        v1d --> v1e["More guards"]
    end

    subgraph V2["v2: unified relative score"]
        direction TB
        v2a["Extract pass–fail channels"]
        v2b["LLM/VLM soft priors"]
        v2c["Weighted fusion + policies"]
        v2d["Sort once"]
        v2a --> v2c
        v2b --> v2c
        v2c --> v2d
    end
```

---

## Relative debugging data flow (conceptual)

```mermaid
flowchart LR
    subgraph PASS["Pass trace step i"]
        PA["action_i"]
        PN["network_i"]
        PC["console_i"]
        PS["screenshot_i"]
    end

    subgraph FAIL["Fail trace step i"]
        FA["action_i"]
        FN["network_i"]
        FC["console_i"]
        FS["screenshot_i"]
    end

    subgraph REL["Relative features"]
        D1["action_changed"]
        D2["new_errors"]
        D3["nav_mismatch"]
        D4["pixel_diff"]
    end

    PA -.-> D1
    FA -.-> D1
    PN -.-> D2
    FN -.-> D2
    PS -.-> D4
    FS -.-> D4

    D1 & D2 & D3 & D4 --> SCORE["Fusion score\n(rank all steps)"]
```

---

## Exporting for LaTeX

1. Paste diagram into [mermaid.live](https://mermaid.live) → Export PNG/SVG.
2. Suggested filenames for paper:
   - `figures/arch_v1_pipeline.pdf`
   - `figures/arch_v2_pipeline.pdf` (Figure 1 main)
   - `figures/arch_v1_v2_compare.pdf`
3. Captions:
   - **Fig 1 (v2):** “TraceLens v2 relative debugging pipeline. Pass and fail traces are aligned and differenced into multimodal channels; optional LLM/VLM priors feed a single weighted fusion ranker.”
   - **Fig A (v1):** “v1 stacked pipeline with post-hoc Hit@1 guards (red). Hybrid mode applies LLM and VLM override stages sequentially.”

---

## Cross-references

- Paper plan: [`ICSE_PAPER_PLAN.md`](ICSE_PAPER_PLAN.md)
- v1 guard policy: [`RANKING_ARBITRATION.md`](RANKING_ARBITRATION.md)
- Mode details: [`MODES.md`](MODES.md)
- v2 README: [`../v2/README.md`](../v2/README.md)
