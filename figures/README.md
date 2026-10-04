# Figures and Tables

Start with the [figure gallery](index.html), or open the PDFs in the folders below. The figures use formal run `20260908T224807Z` and retain all 595 target items.

| Folder | Contents |
|---|---|
| [main/](main/) | Figures 1–4 as PDFs and [captions](main/captions.md) |
| [supplementary/](supplementary/) | Figures S1–S3 as PDFs and [captions](supplementary/captions.md) |
| [tables/](tables/) | Tables 1–2 as PDFs and [copyable tables](tables/tables.md) |
| [source_data/](source_data/) | All figure source CSVs, fitted values, full-precision statistics and screening records |
| [formats/](formats/) | Additional PNG, editable SVG and TIFF exports, grouped by format |

PDF is the primary viewing format. PNG previews are 300 dpi; TIFF exports are 600 dpi. All original image files and numerical source data are preserved without rerendering. Validation reports and layout proofs are in [docs/figure_validation](../docs/figure_validation/README.md). Formal machine outputs remain in `results/formal/20260908T224807Z/`.

## Displays

| Display | PDF |
|---|---|
| Figure 1: experimental overview | [Figure 1](main/fig1_experiment_overview.pdf) |
| Figure 2: continuation alignment | [Figure 2](main/fig2_rq1_distribution.pdf) |
| Figure 3: overall association | [Figure 3](main/fig3_overall_association.pdf) |
| Figure 4: conditional association | [Figure 4](main/fig4_conditional_association.pdf) |
| Figure S1: candidate-order sensitivity | [Figure S1](supplementary/FigS1_candidate_order.pdf) |
| Figure S2: seed comparisons | [Figure S2](supplementary/FigS2_seed_comparison.pdf) |
| Figure S3: local surprisal | [Figure S3](supplementary/FigS3_surprisal.pdf) |
| Table 1: materials | [Table 1](tables/table1_materials.pdf) |
| Table 2: RQ2 models | [Table 2](tables/table2_rq2_models.pdf) |

## Reproduction

The plotting scripts use Python, matplotlib, NumPy, pandas, PyMuPDF and the external `nature-figure/scripts` Python validation utilities. Saved results are sufficient; no model weights or GPU are needed for plotting.

Run from the repository root:

```bash
python runpod/experiments/build_overview_figure.py
python runpod/experiments/build_manuscript_figures.py
python runpod/experiments/build_supplement_figures.py
python runpod/experiments/build_screening_audit.py
python runpod/experiments/qa_manuscript_main.py
python runpod/experiments/qa_manuscript_annotations.py
python runpod/experiments/package_manuscript_figures.py
```

The entry points write directly to the folders above. Packaging creates `transfer/manuscript_20260911.zip`; it does not place archives among the figures.
