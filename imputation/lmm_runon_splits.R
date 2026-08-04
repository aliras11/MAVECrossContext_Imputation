#!/usr/bin/env Rscript

# Usage:
# Rscript lmm_runonsplits.R --splits_dir /Users/ser219/Desktop/imputation_splits_mthfr/data_splits --rate 10 --out_dir linear_models_outputs

suppressPackageStartupMessages({
  if (!requireNamespace("optparse", quietly = TRUE)) install.packages("optparse", repos = "https://cloud.r-project.org")
  if (!requireNamespace("data.table", quietly = TRUE)) install.packages("data.table", repos = "https://cloud.r-project.org")
  if (!requireNamespace("lme4", quietly = TRUE)) install.packages("lme4", repos = "https://cloud.r-project.org")
  library(optparse)
  library(data.table)
  library(lme4)
})

option_list <- list(
  make_option(c("--splits_dir"), type="character", default="data_splits",
              help="Base directory containing test_frac_* subfolders with train_split_*.csv"),
  make_option(c("--rate"), type="integer", default=10,
              help="Test fraction rate selecting subfolder test_frac_{rate} (e.g., 10)"),
  make_option(c("--out_dir"), type="character", default="linear_models_outputs",
              help="Output base directory for per-model imputation CSVs")
)
opt <- parse_args(OptionParser(option_list=option_list))
splits_dir <- opt$splits_dir
rate <- opt$rate
out_dir <- opt$out_dir

tf_dir <- file.path(splits_dir, sprintf("test_frac_%d", rate))
if (!dir.exists(tf_dir)) {
  stop(sprintf("Folder not found: %s", tf_dir))
}
split_files <- list.files(tf_dir, pattern="^train_split_.*\\.csv$", full.names=TRUE)
if (!length(split_files)) {
  stop(sprintf("No train_split_*.csv files in %s", tf_dir))
}

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

for (csv_path in split_files) {
  dat <- data.table::fread(csv_path, data.table = FALSE)

  # Identify score/map columns
  score_cols <- grep("_score$", names(dat), value = TRUE)
  if (length(score_cols) < 2) next

  # Optional factors if present
  if ("str_aa_mut" %in% names(dat)) dat$str_aa_mut <- as.factor(dat$str_aa_mut)
  if ("domain" %in% names(dat))     dat$domain     <- as.factor(dat$domain)

  # Split number
  split_num <- sub(".*_s([0-9]+)\\..*", "\\1", basename(csv_path))

  # Output dir for this split
  out_split_dir <- file.path(out_dir, sprintf("linear_model_output_%d", rate), sprintf("split_%s", split_num))
  dir.create(out_split_dir, recursive = TRUE, showWarnings = FALSE)

  # Identify non-numeric columns (excluding score/se columns)
  non_numeric_cols <- c()
  for (col in names(dat)) {
    if (!grepl("_score$|_se$", col) && !is.numeric(dat[[col]])) {
      non_numeric_cols <- c(non_numeric_cols, col)
    }
  }
  # Always include these specific columns if they exist
  important_cols <- c("hgvs_pro", "domain", "str_aa_mut", "aa_pos")
  non_numeric_cols <- unique(c(non_numeric_cols, intersect(important_cols, names(dat))))

  # Loop over ordered pairs
  for (from_col in score_cols) {
    for (to_col in score_cols) {
      if (from_col == to_col) next

      # Train on rows with both observed
      train_idx <- !is.na(dat[[from_col]]) & !is.na(dat[[to_col]])
      if (!any(train_idx)) next
      train_data <- dat[train_idx, , drop = FALSE]

      # Predict only for rows where target is missing but source present
      pred_idx <- is.na(dat[[to_col]]) & !is.na(dat[[from_col]])
      if (!any(pred_idx)) next
      newdata <- dat[pred_idx, , drop = FALSE]

      # Ensure numeric for modeling
      train_data[[from_col]] <- as.numeric(train_data[[from_col]])
      train_data[[to_col]]   <- as.numeric(train_data[[to_col]])
      newdata[[from_col]]    <- as.numeric(newdata[[from_col]])

      # Define SE columns for this pair
      from_se <- paste0(sub("_score$", "", from_col), "_se")
      to_se <- paste0(sub("_score$", "", to_col), "_se")

      # Define columns to keep for output
      keep_cols <- c(non_numeric_cols, from_col, to_col)
      if (from_se %in% names(dat)) keep_cols <- c(keep_cols, from_se)
      if (to_se %in% names(dat)) keep_cols <- c(keep_cols, to_se)
      keep_cols <- intersect(keep_cols, names(dat))  # Only existing columns

      # 1) Basic linear model: to ~ from
      tryCatch({
        m_lm <- lm(as.formula(paste(to_col, "~", from_col)), data = train_data)
        preds_lm <- predict(m_lm, newdata = newdata)

        # Create subset with only relevant columns and fill predictions
        out_lm <- dat[, keep_cols, drop = FALSE]
        out_lm[[to_col]][pred_idx] <- preds_lm
        out_lm$map_pair <- paste0(from_col, "_to_", to_col)
        outfile_lm <- file.path(
          out_split_dir,
          sprintf("basic_linear_%s_to_%s_s%s_r%d.csv", from_col, to_col, split_num, rate)
        )
        data.table::fwrite(out_lm, outfile_lm)
      }, error = function(e) {
        cat(sprintf("Error in basic_linear %s->%s: %s\n", from_col, to_col, e$message))
      })

      # 2) One-parameter nonlinear model: to ~ 1 - B*(1 - from)
      tryCatch({
        m_nls <- nls(as.formula(paste(to_col, "~ 1 - B*(1 -", from_col, ")")),
                     start = list(B = 1), data = train_data)
        preds_nls <- predict(m_nls, newdata = newdata)
        
        out_nls <- dat[, keep_cols, drop = FALSE]
        out_nls[[to_col]][pred_idx] <- preds_nls
        out_nls$map_pair <- paste0(from_col, "_to_", to_col)
        outfile_nls <- file.path(
          out_split_dir,
          sprintf("oneparam_linear_%s_to_%s_s%s_r%d.csv", from_col, to_col, split_num, rate)
        )
        data.table::fwrite(out_nls, outfile_nls)
      }, error = function(e) {
        cat(sprintf("Error in oneparam_linear %s->%s: %s\n", from_col, to_col, e$message))
      })

      # 3) Linear with interaction by domain (only if domain present)
      if ("domain" %in% names(dat)) {
        tryCatch({
          m_lm_int <- lm(as.formula(paste(to_col, "~", from_col, "*domain")), data = train_data)
          preds_lm_int <- predict(m_lm_int, newdata = newdata)
          
          out_lm_int <- dat[, keep_cols, drop = FALSE]
          out_lm_int[[to_col]][pred_idx] <- preds_lm_int
          out_lm_int$map_pair <- paste0(from_col, "_to_", to_col)
          outfile_lm_int <- file.path(
            out_split_dir,
            sprintf("full_interaction_linear_%s_to_%s_s%s_r%d.csv", from_col, to_col, split_num, rate)
          )
          data.table::fwrite(out_lm_int, outfile_lm_int)
        }, error = function(e) {
          cat(sprintf("Error in full_interaction_linear %s->%s: %s\n", from_col, to_col, e$message))
        })
      }

      # 4) Mixed effects with random intercepts (only if factors exist)
      if (all(c("domain", "str_aa_mut") %in% names(dat))) {
        # with interaction in fixed effects and random intercept by str_aa_mut
        tryCatch({
          m_lmer1 <- lmer(as.formula(paste(to_col, "~", from_col, "*domain + (1 | str_aa_mut)")),
                          data = train_data, REML = TRUE)
          preds_lmer1 <- predict(m_lmer1, newdata = newdata, allow.new.levels = TRUE)
          
          out_lmer1 <- dat[, keep_cols, drop = FALSE]
          out_lmer1[[to_col]][pred_idx] <- preds_lmer1
          out_lmer1$map_pair <- paste0(from_col, "_to_", to_col)
          outfile_lmer1 <- file.path(
            out_split_dir,
            sprintf("full_interaction_mixed_%s_to_%s_s%s_r%d.csv", from_col, to_col, split_num, rate)
          )
          data.table::fwrite(out_lmer1, outfile_lmer1)
        }, error = function(e) {
          cat(sprintf("Error in full_interaction_mixed %s->%s: %s\n", from_col, to_col, e$message))
        })

        # 5) random slope model
        tryCatch({
          m_lmer2 <- lmer(as.formula(paste(to_col, "~", from_col, " + domain + (1 +", from_col, "| str_aa_mut)")),
                          data = train_data, REML = TRUE)
          preds_lmer2 <- predict(m_lmer2, newdata = newdata, allow.new.levels = TRUE)
          
          out_lmer2 <- dat[, keep_cols, drop = FALSE]
          out_lmer2[[to_col]][pred_idx] <- preds_lmer2
          out_lmer2$map_pair <- paste0(from_col, "_to_", to_col)
          outfile_lmer2 <- file.path(
            out_split_dir,
            sprintf("mixed_random_%s_to_%s_s%s_r%d.csv", from_col, to_col, split_num, rate)
          )
          data.table::fwrite(out_lmer2, outfile_lmer2)
        }, error = function(e) {
          cat(sprintf("Error in mixed_random %s->%s: %s\n", from_col, to_col, e$message))
        })
      }
    }
  }
}

cat("Done.\n")