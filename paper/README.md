# TraceLens Paper

LaTeX draft for the CS 460 project on **multimodal relative debugging** of UI test regressions.

**Artifact:** [github.com/MAKhan22/TraceLens-Automated-Debugging](https://github.com/MAKhan22/TraceLens-Automated-Debugging)

## Contents

| File | Description |
|------|-------------|
| `main.tex` | Full paper (~10 pages when compiled) |
| `references.bib` | Bibliography (16 verified entries) |
| `figures/arch_v2_tikz.tex` | v2 pipeline (TikZ, rendered in LaTeX) |
| `figures/arch_v1_tikz.tex` | v1 pipeline (TikZ, rendered in LaTeX) |
| `figures/arch_tikz_styles.tex` | Shared TikZ styles |

## Build

Requires the [ACM `acmart`](https://www.acm.org/publications/proceedings-template) document class:

```bash
# Fedora / TeX Live
sudo tlmgr install acmart

cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Output: `main.pdf`

## Regenerate figures

```bash
# Evaluation figures (from frozen metrics)
python scripts/plot_metrics.py

# Architecture diagrams (optional PNG export)
python scripts/generate_arch_figures.py   # legacy; paper uses TikZ instead

# Copy into paper/figures/
cp outputs/figures/fig{1,2,3,6}_*.png paper/figures/
cp paper/figures/arch_v*.png paper/figures/   # already written by script
```

## Before submission

1. Verify all numbers against `scripts/metrics_manifest.yaml`.
2. Confirm page count against venue CFP (typically 10 pages + references).
3. Add artifact DOI / GitHub link when required by the venue.

## Related docs

- [`docs/ICSE_PAPER_PLAN.md`](../docs/ICSE_PAPER_PLAN.md) — section-by-section writing plan
- [`docs/ARCHITECTURE_DIAGRAMS.md`](../docs/ARCHITECTURE_DIAGRAMS.md) — Mermaid source diagrams
