#!/usr/bin/env Rscript

# No-double-missing variant of mice_rf_run_onsplits.R (MICE Random Forest)
# Only the target column has injected missingness; source columns are fully observed.
# Iterates 7 source contexts for the given target (excludes self-pairs).
#
# Usage:
#   Rscript mice_rf_runon_nodouble_splits.R <splits_dir> <test_rate> <target>
#   Rscript mice_rf_runon_nodouble_splits.R ../data_splits_no_double_missing 10 av12

local_lib <- file.path(tempdir(), "r_lib")
dir.create(local_lib, showWarnings = FALSE, recursive = TRUE)
.libPaths(c(local_lib, .libPaths()))
options(repos = c(CRAN = "https://cloud.r-project.org"))
options(Ncpus = as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", "1")))

ensure_pkgs <- function(pkgs) {
  for (p in pkgs) {
    if (!requireNamespace(p, quietly = TRUE)) {
      install.packages(p, dependencies = TRUE)
    }
  }
}
ensure_pkgs(c("class", "R6", "data.table", "ranger", "miceRanger"))

suppressPackageStartupMessages({
  library(data.table)
  library(miceRanger)
})

args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 3) {
  cat("Usage: Rscript mice_rf_runon_nodouble_splits.R <splits_dir> <test_rate> <target>\n")
  cat("Example: Rscript mice_rf_runon_nodouble_splits.R ../data_splits_no_double_missing 10 av12\n")
  quit(status = 1)
}

splits_dir <- args[1]
test_rate  <- as.integer(args[2])
target     <- args[3]

m_rounds <- 1
maxit    <- 10
seed     <- 123

# Paths: read from tgt_{target}/test_frac_{rate}/
tgt_dir       <- file.path(splits_dir, sprintf("tgt_%s", target))
test_frac_dir <- file.path(tgt_dir, sprintf("test_frac_%d", test_rate))
if (!dir.exists(test_frac_dir)) {
  stop(sprintf("Directory not found: %s", test_frac_dir))
}

train_files <- list.files(test_frac_dir, pattern = "^train_split_.*\\.csv$", full.names = TRUE)
if (length(train_files) == 0) {
  stop(sprintf("No train_split files found in: %s", test_frac_dir))
}

cat(sprintf("Found %d train_split files (target=%s, rate=%d)\n", length(train_files), target, test_rate))

to_score <- paste0(target, "_score")
to_se    <- paste0(target, "_se")

for (train_file in train_files) {
  split_num <- sub(".*_s([0-9]+)\\.csv$", "\\1", basename(train_file))
  cat(sprintf("Processing split %s...\n", split_num))

  dat <- fread(train_file, data.table = FALSE)

  score_cols <- grep("_score$", names(dat), value = TRUE)
  if (!(to_score %in% score_cols)) {
    stop(sprintf("Target score column %s not found", to_score))
  }

  # Source columns: all score columns except target
  from_scores <- setdiff(score_cols, to_score)

  char_cols <- sapply(dat, is.character)
  dat[char_cols] <- lapply(dat[char_cols], as.factor)

  # Output: tgt_{target}/mice_test_rf2_frac_{rate}/split_{N}/
  split_output_dir <- file.path(tgt_dir, paste0("mice_test_rf2_frac_", test_rate), paste0("split_", split_num))
  dir.create(split_output_dir, recursive = TRUE, showWarnings = FALSE)

  for (i in from_scores) {
    i_base <- sub("(_score|_se)$", "", i)
    j_base <- target

    # Skip if output already exists
    output_filename <- sprintf("miceRF_imputed_%s_to_%s_split%s_rate%d.csv",
                                i_base, j_base, split_num, test_rate)
    output_path <- file.path(split_output_dir, output_filename)
    if (file.exists(output_path)) {
      cat(sprintf("  Skipping pair: %s -> %s (already exists)\n", i_base, j_base))
      next
    }

    cat(sprintf("  Processing pair: %s -> %s\n", i_base, j_base))

    m1 <- paste0(i_base, "_score")
    s1 <- paste0(i_base, "_se")
    m2 <- to_score
    s2 <- to_se

    covariates      <- c("aa_pos", "str_aa_wt", "str_aa_mut", "domain")
    identifier_cols <- c("hgvs_pro")

    keep_cols <- c(identifier_cols, covariates, m1, m2)
    if (s1 %in% names(dat)) keep_cols <- c(keep_cols, s1)
    if (s2 %in% names(dat)) keep_cols <- c(keep_cols, s2)
    keep_cols <- intersect(keep_cols, names(dat))

    if (!all(c(m1, m2) %in% keep_cols)) {
      cat(sprintf("    Skipping %s-%s: missing required columns\n", i_base, j_base))
      next
    }

    pair_data <- dat[, keep_cols, drop = FALSE]

    if (!any(is.na(pair_data[[m1]]) | is.na(pair_data[[m2]]))) {
      cat(sprintf("    Skipping %s-%s: no missing values in score columns\n", i_base, j_base))
      next
    }

    tryCatch({
      pair_no_id <- pair_data
      id_df <- NULL
      if ("hgvs_pro" %in% names(pair_no_id)) {
        id_df <- pair_no_id["hgvs_pro"]
        pair_no_id["hgvs_pro"] <- NULL
      }

      set.seed(seed)

      pred1 <- completeData(miceRanger::miceRanger(
        data          = pair_no_id,
        m             = m_rounds,
        maxiter       = maxit,
        verbose       = FALSE,
        seed          = seed,
        valueSelector = "value",
        returnModels  = FALSE,
      ))[[1]]

      if (!is.null(id_df)) {
        pred1$hgvs <- id_df
      }

      pred1$map_pair <- paste0(i_base, "_to_", j_base)

      data.table::fwrite(pred1, output_path)
      cat(sprintf("    Saved: %s\n", output_filename))

    }, error = function(e) {
      cat(sprintf("    Error with %s-%s: %s\n", i_base, j_base, e$message))
    })
  }
}

cat("MICE RF (no-double-missing) completed!\n")
