"""Shared constants for normalized imputation results."""

NORMALIZED_COLUMNS = (
    "dataset", "model", "rate", "split", "src", "tgt",
    "shift_type", "loss_type", "rmse", "n_points",
)

REGULAR_RESULT_FILES = (
    "single_AE3_rmse_results.csv",
    "dual_AE3_rmse_results.csv",
    "mice_loss_measurements_all_splits_rates2.csv",
    "mice_loss_measurements_all_splits_ratesrf2.csv",
    "linear_model_loss_measurements_all_splits_rates2.csv",
    "blosum_knn_direct_rmse_all_splits.csv",
    "col_mean_imputed_results.csv",
    "pca_rmse_results_all.csv",
)

NODOUBLE_RESULT_FILES = (
    "single_AE3_rmse_no_double_missing.csv",
    "dual_AE3_rmse_no_double_missing.csv",
    "mice_loss_no_double_missing.csv",
    "mice_rf_loss_no_double_missing.csv",
    "linear_model_loss_no_double_missing.csv",
)

MODEL_DISPLAY_NAMES = {
    "single_ae": "SingleAE",
    "dual_ae": "DualAE",
    "mice": "MICE-PMM",
    "mice_rf": "MICE-RF",
    "basic_linear": "Basic Linear",
    "oneparam_linear": "1-Param Nonlinear",
    "full_interaction_linear": "Linear + Domain",
    "mixed_random": "Mixed (rand. slope)",
    "full_interaction_mixed": "Mixed (rand. int.)",
    "col_mean": "Column Mean",
    "knn": "$k$NN-BLOSUM",
    "pca_k1": "PCA (k = 1)",
}

B1_MODELS = (
    "single_ae", "dual_ae", "mice", "mice_rf", "basic_linear",
    "oneparam_linear", "full_interaction_linear", "mixed_random",
    "full_interaction_mixed",
)
B0_MODELS = ("single_ae", "dual_ae", "mice", "mice_rf")
W_MODELS = ("single_ae", "mice_rf", "knn", "pca_k1")
REGULAR_RATES = (10, 20, 40, 60, 80, 90)
NODOUBLE_RATES = (10, 40, 80, 99, 999)

EXPECTED_STATISTICS_CSVS = frozenset({
    "nodouble_model_rate_completeness.csv",
    "pairwise_mwu_regression_test.csv",
    "pairwise_mwu_double_missing.csv",
    "pairwise_mwu_within_map.csv",
    "pairwise_mwu_nodouble_regression_test.csv",
    "rmse_summary_regression_test.csv",
    "rmse_summary_double_missing.csv",
    "rmse_summary_within_map.csv",
    "rmse_summary_nodouble_regression_test.csv",
    "vs_colmean_regression_test.csv",
    "vs_colmean_double_missing.csv",
    "vs_colmean_within_map.csv",
    "pairwise_mwu_by_context_regression_test.csv",
    "pairwise_mwu_by_context_double_missing.csv",
    "pairwise_mwu_by_context_within_map.csv",
    "pairwise_mwu_by_context_nodouble_regression_test.csv",
})
