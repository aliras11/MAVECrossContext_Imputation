###updated measure loss on mice RF imputation outputs - based on code ran locally to obtain final results ###

import pandas as pd
import numpy as np
from pathlib import Path
import re

from runtime_paths import FULL_DATA_CSV, LOSS_RESULTS_DIR, REGULAR_SPLITS_DIR




###measure losses on MICE RF outputs###
full_data_df = pd.read_csv(FULL_DATA_CSV)
base_dir = REGULAR_SPLITS_DIR

def extract_r_s(name: str):
    # Pull integers following 'r' and 's' anywhere in the filename
    rm = re.search(r"r(\d+)", name, flags=re.IGNORECASE)
    sm = re.search(r"s(\d+)", name, flags=re.IGNORECASE)
    r = int(rm.group(1)) if rm else None
    s = int(sm.group(1)) if sm else None
    return r, s

def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    pair = pd.concat([y_true, y_pred], axis=1).dropna()
    if pair.empty:
        return float("nan")
    diff = pair.iloc[:, 0].to_numpy() - pair.iloc[:, 1].to_numpy()
    return float(np.sqrt(np.mean(diff * diff)))



all_results = []
rates = [10,20,40,60,80,90]



for r in rates:
    print(f"\nTest fraction r: {r}")
    mice_model_out = base_dir / f"mice_test_rf2_frac_{r}" #open relevant test fraction folder for model outputs
    masked_data_folder = base_dir / f"test_frac_{r}" #open the relevant training data for that test fraction
    for split_file in masked_data_folder.iterdir():

        if split_file.is_file() and split_file.name.startswith("train_split_"):
            print(f"  Split file: {split_file.name}")
            masked_data_split = pd.read_csv(split_file)
            r , sp = extract_r_s(split_file.name)
            print(f"    r,s extracted: {r}, {sp}")
            for model_out_split in mice_model_out.iterdir():
                if model_out_split.is_dir() and model_out_split.name == f"split_{sp}":
                    print("hello")
                    print(f"   Model output split found: {model_out_split.name}")
                    for file in model_out_split.iterdir():
                        if file.is_file() and file.suffix == ".csv":
                            print(f"      File: {file.name}")
                            model_output_df = pd.read_csv(file)
                            model_output_df.rename(columns={'hgvs': 'hgvs_pro'}, inplace=True)
                            src, tgt = re.search(r'([^_]+)_to_([^_]+)', file.name).groups()
                            print(f"      File: {file.name} -> src: {src}, tgt: {tgt}")
                            src_score = f"{src}_score"
                            tgt_score = f"{tgt}_score"
                            if src_score == tgt_score:

                                merged = model_output_df.merge(masked_data_split[['hgvs_pro', tgt_score]], on="hgvs_pro", how="inner", suffixes=('', '_masked')) \
                                .merge(full_data_df[['hgvs_pro', tgt_score]], on="hgvs_pro", how="inner", suffixes=('', '_full'))
                                test_mask = merged[f'{tgt_score}_masked'].isna()
                                tempdf = merged[test_mask]
                                rmse_value = rmse(tempdf[f'{tgt_score}_full'], tempdf[tgt_score])

                                # Per-bucket point counts — only cells where both truth and
                                # prediction are non-NaN (matches what rmse() actually scored).
                                # Compute n_training from the UNFILTERED merged frame (the prior
                                # version computed it on the test-only subset, so n_training was
                                # always 0).
                                elig = merged[[f'{tgt_score}_full', tgt_score]].notna().all(axis=1)
                                n_test = int((test_mask & elig).sum())
                                n_training = int((~test_mask & elig).sum())
                                per_bucket_rmse_wide = pd.DataFrame({
                                'rmse_regression': [pd.NA],
                                'rmse_train': [pd.NA],
                                'rmse_double_missing': [pd.NA],
                                'rmse_src_missing_only': [pd.NA],
                                'rmse_within_map_tgt': [rmse_value],
                                'rate': [r],
                                'split': [sp],
                                'src-tgt': [f"{src}->{tgt}"],
                                'model': ['mice_rf'],
                                'num_test_points_for_tgt': [n_test],
                                'n_test': [n_test],
                                'n_training': [n_training],
                                'n_regression_test_loss': [pd.NA],
                                'n_double_missing': [pd.NA],
                                'n_no_src_prediction': [pd.NA],
                                })
                                all_results.append(per_bucket_rmse_wide)
                                continue
                            if src_score == tgt_score:
                                raise RuntimeError(
                                    f"Terminating: src_score == tgt_score ({src_score}) in file {file.name}"
                                )
                            na_flags = masked_data_split[['hgvs_pro']].assign(
                                double_missing=(masked_data_split[src_score].isna()) & (masked_data_split[tgt_score].isna()),
                                src_missing_only = masked_data_split[src_score].isna() & ~(masked_data_split[tgt_score].isna()),
                                training = ~(masked_data_split[src_score].isna()) & ~(masked_data_split[tgt_score].isna()),
                                regression_test_loss = ~(masked_data_split[src_score].isna()) & (masked_data_split[tgt_score].isna())
                            )

                            # Merge into model_output_df
                            merged_with_flags = model_output_df.merge(na_flags, on='hgvs_pro', how='left')
                            merged_with_truth = merged_with_flags.merge(
                                full_data_df[['hgvs_pro', src_score, tgt_score]],
                                on='hgvs_pro',
                                how='left',
                                suffixes=('', '_true')
                            )

                            temp_regression_loss = merged_with_truth.loc[merged_with_truth['regression_test_loss'].astype(bool)]
                            rmse_regression = rmse(
                                temp_regression_loss[tgt_score + '_true'],
                                temp_regression_loss[tgt_score]
                            )

                            rmse_train = rmse(
                                merged_with_truth.loc[merged_with_truth['training'].astype(bool), tgt_score + '_true'],
                                merged_with_truth.loc[merged_with_truth['training'].astype(bool), tgt_score]
                            )

                            rmse_double_missing = rmse(
                                merged_with_truth.loc[merged_with_truth['double_missing'].astype(bool), tgt_score + '_true'],
                                merged_with_truth.loc[merged_with_truth['double_missing'].astype(bool), tgt_score]
                            )

                            rmse_src_missing_only = rmse(
                                merged_with_truth.loc[merged_with_truth['src_missing_only'].astype(bool), tgt_score + '_true'],
                                merged_with_truth.loc[merged_with_truth['src_missing_only'].astype(bool), tgt_score]
                            )
                            print(f"RMSE for regression_test_loss: {rmse_regression}")
                            # Per-bucket point counts — only cells where both truth and
                            # prediction are non-NaN (matches what rmse() actually scored).
                            elig = merged_with_truth[[tgt_score + '_true', tgt_score]].notna().all(axis=1)
                            n_training = int((merged_with_truth['training'].astype(bool) & elig).sum())
                            n_regression_test_loss = int((merged_with_truth['regression_test_loss'].astype(bool) & elig).sum())
                            n_double_missing = int((merged_with_truth['double_missing'].astype(bool) & elig).sum())
                            n_no_src_prediction = int((merged_with_truth['src_missing_only'].astype(bool) & elig).sum())
                            per_bucket_rmse_wide = pd.DataFrame({
                                'rmse_regression': [rmse_regression],
                                'rmse_train': [rmse_train],
                                'rmse_double_missing': [rmse_double_missing],
                                'rmse_src_missing_only': [rmse_src_missing_only],
                                'rmse_within_map_tgt': [pd.NA],
                                'rate': [r],
                                'split': [sp],
                                'src-tgt': [f"{src}->{tgt}"],
                                'model': ['mice_rf'],
                                'num_test_points_for_tgt': [temp_regression_loss.shape[0]],
                                'n_training': [n_training],
                                'n_regression_test_loss': [n_regression_test_loss],
                                'n_double_missing': [n_double_missing],
                                'n_no_src_prediction': [n_no_src_prediction],
                                'n_test': [pd.NA],
                            })
                            all_results.append(per_bucket_rmse_wide)
if all_results:
    combined_results = pd.concat(all_results, ignore_index=True)
    print(f"\nCombined results shape: {combined_results.shape}")
    combined_results
else:
    print("No results found")
output_dir = LOSS_RESULTS_DIR
output_dir.mkdir(parents=True, exist_ok=True)
combined_results.to_csv(output_dir / "mice_loss_measurements_all_splits_ratesrf2.csv", index=False)
