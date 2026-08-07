# Archived MTHFR source-score exports

These 51 CSV files were copied without content changes on 2026-08-06 from the
project's local MTHFR score-processing workspace.

## Contents

- `mthfr_wt_scores_thresh10/`: 24 wild-type-background CSVs covering four
  folinate conditions and the complete, enrichment, error-model, simple,
  amino-acid-level, and floored amino-acid-level exports.
- `mthfr_a222v_scores_thresh10/`: 24 A222V-background CSVs with the same four
  conditions and six export types.
- `mthfr_crossAllcontext_domainannotation.csv`: an earlier combined context
  table.
- `mthfr_crossAllcontext_domainannotation_new.csv`: the combined table that is
  byte-for-byte identical to the pipeline's authoritative
  `../mthfr_crossAllcontext_domainannotation.csv`.
- `mthfr_excel_scores.csv`: the accompanying combined score table.

These archival files are not consumed directly by the current pipeline. Split
generation, imputation, and loss measurement read the authoritative repository
CSV. Variance decomposition and the Trp165 analysis read a checksum-identical
copy staged under the run root, and the statistics commands read the resulting
loss tables.

The source folder's ZIP archives were omitted because they duplicate the CSV
directories. Its notebooks, helper scripts, and `.DS_Store` file were also
omitted because this repository is limited to the runtime analysis code and
scientific data.
