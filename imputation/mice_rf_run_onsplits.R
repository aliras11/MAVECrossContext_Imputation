#!/usr/bin/env Rscript

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


# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 2) {
  cat("Usage: Rscript mice_runonsplitsv1.2.R <splits_dir> <test_rate>\n")
  quit(status = 1)
}

# Get arguments from command line
splits_dir <- args[1]
test_rate <- as.integer(args[2])

# Fixed settings 
m_rounds <- 1
maxit <- 10
seed <- 123

# Construct paths
test_frac_dir <- file.path(splits_dir, paste0("test_frac_", test_rate))
if (!dir.exists(test_frac_dir)) {
  stop(sprintf("Directory not found: %s", test_frac_dir))
}

# Find all train_split files
train_files <- list.files(test_frac_dir, pattern = "^train_split_.*\\.csv$", full.names = TRUE)
if (length(train_files) == 0) {
  stop(sprintf("No train_split files found in: %s", test_frac_dir))
}

cat(sprintf("Found %d train_split files\n", length(train_files)))

# Process each split file
for (train_file in train_files) {
  split_num <- sub(".*_s([0-9]+)\\.csv$", "\\1", basename(train_file))
  cat(sprintf("Processing split %s...\n", split_num))

  dat <- fread(train_file, data.table = FALSE)

  # Identify score columns
  score_cols <- grep("_score$", names(dat), value = TRUE)
  if (length(score_cols) < 2) {
    cat(sprintf("Skipping split %s: insufficient score columns\n", split_num))
    next
  }

  # Convert character columns to factors for imputation
  char_cols <- sapply(dat, is.character)
  dat[char_cols] <- lapply(dat[char_cols], as.factor)

  # Create output directory for this split
  split_output_dir <- file.path(splits_dir, paste0("mice_test_rf2_frac_", test_rate), paste0("split_", split_num))
  dir.create(split_output_dir, recursive = TRUE, showWarnings = FALSE)

  # Loop over all pairs of score columns (DO NOT skip same-score pairs)
  for (i in score_cols) {
    for (j in score_cols) {
      i_base <- sub("(_score|_se)$", "", i)
      j_base <- sub("(_score|_se)$", "", j)


      # Skip if output already exists
      output_filename <- sprintf("miceRF_imputed_%s_to_%s_split%s_rate%d.csv",
                                  i_base, j_base, split_num, test_rate)
      output_path <- file.path(split_output_dir, output_filename)
      if (file.exists(output_path)) {
        cat(sprintf("  Skipping pair: %s -> %s (already exists)\n", i_base, j_base))
        next
      }

      cat(sprintf("  Processing pair: %s -> %s\n", i_base, j_base))

      # Define column names for this pair
      m1 <- paste0(i_base, "_score")
      s1 <- paste0(i_base, "_se")
      m2 <- paste0(j_base, "_score")
      s2 <- paste0(j_base, "_se")

      covariates <- c("aa_pos", "str_aa_wt", "str_aa_mut", "domain")
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

      # If nothing to impute in these two score columns, skip
      if (!any(is.na(pair_data[[m1]]) | is.na(pair_data[[m2]]))) {
        cat(sprintf("    Skipping %s-%s: no missing values in score columns\n", i_base, j_base))
        next
      }

      # Impute with miceRanger:
      # - Exclude identifier columns from predictors by removing them during imputation
      # - miceRanger will only impute columns that have NAs
      tryCatch({
        pair_no_id <- pair_data
        id_df <- NULL
        if ("hgvs_pro" %in% names(pair_no_id)) {
          id_df <- pair_no_id["hgvs_pro"]
          pair_no_id["hgvs_pro"] <- NULL
        }

        set.seed(seed)
        

      
        pred1<-completeData(peuter3 <- miceRanger::miceRanger(
        data     = pair_no_id,
        m        = m_rounds,
        maxiter  = maxit,
        verbose  = FALSE,
        seed     = seed,
        valueSelector = "value",
        returnModels = FALSE,

        ))[[1]]

        # Add back identifier column and reorder to original pair_data order
        if (!is.null(id_df)) {
          pred1$hgvs <- id_df
        }


        pred1$map_pair <- paste0(i_base, "_to_", j_base)

        output_filename <- sprintf("miceRF_imputed_%s_to_%s_split%s_rate%d.csv",
                                   i_base, j_base, split_num, test_rate)
        output_path <- file.path(split_output_dir, output_filename)

        data.table::fwrite(pred1, output_path)
        cat(sprintf("    Saved: %s\n", output_filename))

      }, error = function(e) {
        cat(sprintf("    Error with %s-%s: %s\n", i_base, j_base, e$message))
      })
    }
  }
}

cat("MICE (miceRanger) imputation completed!\n")