#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(mice)
})

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 2) {
  cat("Usage: Rscript mice_runon_splits.R <splits_dir> <test_rate>\n")
  cat("Example: Rscript mice_runon_splits.R /path/to/data_splits 10\n")
  quit(status = 1)
}

# Get arguments from command line
splits_dir <- args[1]
test_rate <- as.integer(args[2])

# Fixed settings (unchanged)
m_rounds <- 1
maxit <- 10
mice_method <- "pmm"
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
  # Extract split number from filename (e.g., train_split_r10_s5.csv -> 5)
  split_num <- sub(".*_s([0-9]+)\\.csv$", "\\1", basename(train_file))
  
  cat(sprintf("Processing split %s...\n", split_num))
  
  # Read the training data
  dat <- fread(train_file, data.table = FALSE)
  
  # Identify score columns
  score_cols <- grep("_score$", names(dat), value = TRUE)
  if (length(score_cols) < 2) {
    cat(sprintf("Skipping split %s: insufficient score columns\n", split_num))
    next
  }
  
  # Convert character columns to factors for MICE
  char_cols <- sapply(dat, is.character)
  dat[char_cols] <- lapply(dat[char_cols], as.factor)
  
  # Create output directory for this split
  split_output_dir <- file.path(splits_dir, paste0("mice_test_frac_", test_rate), paste0("split_", split_num))
  dir.create(split_output_dir, recursive = TRUE, showWarnings = FALSE)
  #dir.create(split_output_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Loop over all pairs of score columns
  for (i in score_cols) {
    for (j in score_cols) {
      # Extract base names (remove _score suffix)
      i_base <- sub("(_score|_se)$", "", i)
      j_base <- sub("(_score|_se)$", "", j)
      
      if (i_base == j_base) next  # Skip same score pairs
      
      cat(sprintf("  Processing pair: %s -> %s\n", i_base, j_base))
      
      # Define column names for this pair
      m1 <- paste0(i_base, "_score")
      s1 <- paste0(i_base, "_se")
      m2 <- paste0(j_base, "_score")
      s2 <- paste0(j_base, "_se")
      
      # Define covariates and identifier columns
      covariates <- c("aa_pos", "str_aa_wt", "str_aa_mut", "domain")
      identifier_cols <- c("hgvs_pro")  # Include hgvs column as identifier
      
      # Select columns for this pair: identifiers + covariates + the two score/se pairs
      keep_cols <- c(identifier_cols, covariates, m1, m2)
      # Add SE columns if they exist
      if (s1 %in% names(dat)) keep_cols <- c(keep_cols, s1)
      if (s2 %in% names(dat)) keep_cols <- c(keep_cols, s2)
      
      # Filter to existing columns
      keep_cols <- intersect(keep_cols, names(dat))
      
      # Check if we have the required score columns
      if (!all(c(m1, m2) %in% keep_cols)) {
        cat(sprintf("    Skipping %s-%s: missing required columns\n", i_base, j_base))
        next
      }
      
      # Subset data to only the columns for this pair
      pair_data <- dat[, keep_cols, drop = FALSE]
      
      # Check if there are any missing values to impute in the score columns
      if (!any(is.na(pair_data[[m1]]) | is.na(pair_data[[m2]]))) {
        cat(sprintf("    Skipping %s-%s: no missing values in score columns\n", i_base, j_base))
        next
      }
      
      # Configure MICE to only impute the score and SE columns, using covariates as predictors
      # Note: hgvs_pro should not be used as a predictor or imputed (it's just an identifier)
      tryCatch({
        # Set up MICE methods and predictor matrix
        init_mice <- mice(pair_data, maxit = 0, printFlag = FALSE)
        meth <- init_mice$method
        predM <- init_mice$predictorMatrix
        
        # Only impute score and SE columns (not covariates or identifiers)
        score_se_cols <- c(m1, m2)
        if (s1 %in% names(pair_data)) score_se_cols <- c(score_se_cols, s1)
        if (s2 %in% names(pair_data)) score_se_cols <- c(score_se_cols, s2)
        
        # Set methods: only score/SE columns get imputed
        meth[] <- ""
        meth[score_se_cols] <- mice_method
        
        # Set predictor matrix: covariates predict score/SE columns but identifiers don't
        predM[,] <- 0
        covar_present <- intersect(covariates, names(pair_data))
        
        for (target_col in score_se_cols) {
          # Each score/SE column can be predicted by covariates and other score/SE columns
          # but NOT by identifier columns like hgvs_pro
          # and NOT by its own corresponding SE/score pair
          
          # Determine the corresponding SE/score column for this target
          corresponding_col <- NULL
          if (grepl("_score$", target_col)) {
            # If target is a score, find its corresponding SE
            base_name <- sub("_score$", "", target_col)
            corresponding_col <- paste0(base_name, "_se")
          } else if (grepl("_se$", target_col)) {
            # If target is an SE, find its corresponding score
            base_name <- sub("_se$", "", target_col)
            corresponding_col <- paste0(base_name, "_score")
          }
          
          # Exclude target itself, identifier columns, and corresponding SE/score column
          exclude_cols <- c(target_col, identifier_cols)
          if (!is.null(corresponding_col) && corresponding_col %in% names(pair_data)) {
            exclude_cols <- c(exclude_cols, corresponding_col)
          }
          
          predictor_cols <- setdiff(names(pair_data), exclude_cols)
          predM[target_col, predictor_cols] <- 1
        }
        
        # Run MICE imputation
        set.seed(seed)
        mice_result <- mice(pair_data, m = m_rounds, maxit = maxit, 
                           method = meth, predictorMatrix = predM, printFlag = FALSE)
        
        # Get completed dataset (mean of m imputations)
        completed_data <- complete(mice_result)
        completed_data$map_pair <- paste0(i_base, "_to_", j_base)
        # Save only the pair-specific data (not the full dataset)
        output_filename <- sprintf("mice_imputed_%s_to_%s_split%s_rate%d.csv", 
                                 i_base, j_base, split_num, test_rate)
        output_path <- file.path(split_output_dir, output_filename)
        
        fwrite(completed_data, output_path)
        cat(sprintf("    Saved: %s\n", output_filename))
        
      }, error = function(e) {
        cat(sprintf("    Error with %s-%s: %s\n", i_base, j_base, e$message))
      })
    }
  }
}

cat("MICE imputation completed!\n")
