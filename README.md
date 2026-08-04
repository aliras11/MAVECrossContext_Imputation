# MAVE Imputation Pipeline

This repository consolidates the executable code used to generate MTHFR
missing-data splits, run the MAVE imputation benchmarks, measure prediction
losses, calculate variance-decomposition summaries, perform the manuscript
statistical analyses, and generate the statistical figures.

The repository begins with a new Git history. Earlier repositories and branch
histories are recorded only as source provenance; they are not imported.

## Repository contract

The completed repository contains only material needed by someone rerunning
the computational pipeline:

- executable Python, R, and shell code;
- runtime-required metadata and model resources;
- the authoritative full MTHFR input CSV;
- dependency and command metadata needed to install and invoke the statistics
  package;
- the project license and Git ignore rules; and
- this README, including setup instructions, execution order, input and output
  contracts, and provenance.

The repository does not contain unit tests, test fixtures, development plans,
design specifications, exploratory scripts, superseded implementations,
generated splits, imputed predictions, loss-result CSVs, statistical output
tables, figures, manifests, logs, caches, Word-generation code, PDF-generation
code, LaTeX report code, or supplementary-material generation code.

## Repository layout

```text
MAVE-Imputation-Pipeline/
├── README.md
├── LICENSE
├── .gitignore
├── data/
│   └── mthfr_crossAllcontext_domainannotation.csv
├── imputation/
│   ├── split-generation code
│   ├── model definitions and model runners
│   ├── loss-measurement code
│   ├── decomposition/
│   └── blosum100.iij
├── statistics/
│   ├── pyproject.toml
│   └── code/
│       ├── cli/
│       ├── mave_statistics/
│       └── figures/
└── cluster/
    └── Pitt submission scripts
```

No nested `.git` directories or Git submodules are used.

## Authoritative input data

All split-generation workflows start from:

```text
data/mthfr_crossAllcontext_domainannotation.csv
```

The file contains 13,192 variant outcomes and 24 columns. It includes eight
MTHFR functional-score maps and their reported standard errors, together with
variant identity, amino-acid, position, and protein-domain annotations.

The required SHA-256 digest is:

```text
ed503e4cba97c46cfa5c1be7f2084d9f6ab535bbfe7db101faff146b75a81e4a
```

Generated splits and masks are derived from this file but are not tracked in
Git.

## Pipeline boundary

The runnable workflow has six stages:

1. Generate the regular splits, which inject missingness independently into
   every assay map while retaining the naturally missing values.
2. Generate the no-double-missing splits, which inject missingness into one
   target map at a time while leaving potential source maps unchanged except
   for their natural missingness.
3. Run the Python and R imputation methods on those splits.
4. Measure losses against the authoritative full-data CSV, using the generated
   mask files as the source of truth for synthetically held-out values.
5. Calculate variance-decomposition and split-variability outputs from the
   saved predictions, loss inputs, masks, and reported target standard errors.
6. Generate the statistical CSVs, focused Trp165 analysis, and statistical
   figures from the collated loss-result files.

The README will document the exact commands for each stage and distinguish
local execution from the Pitt Slurm submission wrappers.

## Runtime components

The imputation directory includes the current implementations of Column Mean,
BLOSUM100 KNN, PCA, SingleAE, DualAE, MICE-PMM, MICE-RF, and the implemented
linear and mixed-effects models. It includes both the regular and
no-double-missing runners and their corresponding loss-measurement scripts.

The decomposition directory includes the current shard builder, event and
sufficient-statistic calculations, model-output layout metadata, pooled and
pair/map summaries, reconciliation checks, across-split summaries, and Pitt
submission wrapper.

The statistics directory includes the production data loaders, cell-weighted
RMSE pooling, unpaired Mann–Whitney U inference, report writers, Trp165
analysis, figure-generation framework, individual figure scripts, and command
line entry points.

## Generated data products

Generated products remain outside Git. The final README will describe their
schemas and paths in these groups:

- split CSVs and authoritative injected-missingness mask CSVs;
- per-model imputed prediction CSVs;
- regular and no-double loss-measurement CSVs;
- variance-decomposition sufficient statistics, validation tables, pooled
  summaries, pair/map summaries, across-split summaries, and run manifests;
- headline and context-specific statistical CSVs;
- focused Trp165 event, split, pooled, diagnostic, and manifest files; and
- PNG/SVG statistical figures and their figure manifest.

## Source provenance

The imputation snapshot is based on the current
`feature/decomposition-pipeline` worktree at commit
`e6a62ccacfb4340baa3827394cad2add23ec73ad`, which includes the corrected
prediction-generation code and the current decomposition implementation.

The statistical snapshot is based on repository commit
`6ec548979781495f5cea6e39b2e106c8464cfc8d` plus the intentional uncommitted
label, figure, inference, and Trp165 updates present in that working tree at
consolidation time.

No remote is configured by this consolidation, and nothing is pushed.
