"""Measure losses on MICE RF outputs from no-double-missing splits.

No self-pairs in nodouble analysis — only between-map (src != tgt).

Usage:
    python loss_measure_micerf_no_double_missing.py
    python loss_measure_micerf_no_double_missing.py --target av12 --rates 10 40
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import argparse

CONTEXTS = ['av12', 'av25', 'av100', 'av200', 'wt12', 'wt25', 'wt100', 'wt200']

full_data_df = pd.read_csv(Path(__file__).resolve().parent.parent / "full_data" / "mthfr_crossAllcontext_domainannotation.csv")
default_base_dir = Path(__file__).resolve().parent.parent / "data_splits_no_double_missing"

def extract_r_s(name: str):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=default_base_dir)
    parser.add_argument("--target", nargs="+", default=CONTEXTS)
    parser.add_argument("--rates", nargs="+", type=int, default=[10, 40, 80, 99, 999])
    args = parser.parse_args()

    all_results = []
    for target in args.target:
        tgt_dir = args.base_dir / f"tgt_{target}"
        for r in args.rates:
            print(f"\nTarget: {target}, rate: {r}")
            mice_model_out = tgt_dir / f"mice_test_rf2_frac_{r}"
            masked_data_folder = tgt_dir / f"test_frac_{r}"

            if not masked_data_folder.exists():
                print(f"  Skipping — {masked_data_folder} not found")
                continue

            for split_file in sorted(masked_data_folder.iterdir()):
                if not (split_file.is_file() and split_file.name.startswith("train_split_")):
                    continue

                masked_data_split = pd.read_csv(split_file)
                r_val, sp = extract_r_s(split_file.name)

                if not mice_model_out.exists():
                    print(f"  Skipping — {mice_model_out} not found")
                    continue

                for model_out_split in sorted(mice_model_out.iterdir()):
                    if not (model_out_split.is_dir() and model_out_split.name == f"split_{sp}"):
                        continue

                    for file in sorted(model_out_split.iterdir()):
                        if not (file.is_file() and file.suffix == ".csv"):
                            continue

                        model_output_df = pd.read_csv(file)
                        # MICE RF outputs 'hgvs' instead of 'hgvs_pro'
                        model_output_df.rename(columns={'hgvs': 'hgvs_pro'}, inplace=True)

                        src, tgt = re.search(r'([^_]+)_to_([^_]+)', file.name).groups()
                        src_score = f"{src}_score"
                        tgt_score = f"{tgt}_score"

                        # Skip self-pairs (shouldn't exist in nodouble, but be safe)
                        if src == tgt:
                            continue

                        na_flags = masked_data_split[['hgvs_pro']].assign(
                            double_missing=(masked_data_split[src_score].isna()) & (masked_data_split[tgt_score].isna()),
                            src_missing_only=masked_data_split[src_score].isna() & ~(masked_data_split[tgt_score].isna()),
                            training=~(masked_data_split[src_score].isna()) & ~(masked_data_split[tgt_score].isna()),
                            regression_test_loss=~(masked_data_split[src_score].isna()) & (masked_data_split[tgt_score].isna()),
                        )

                        merged_with_flags = model_output_df.merge(na_flags, on='hgvs_pro', how='left')
                        merged_with_truth = merged_with_flags.merge(
                            full_data_df[['hgvs_pro', src_score, tgt_score]],
                            on='hgvs_pro', how='left', suffixes=('', '_true')
                        )

                        temp_regression_loss = merged_with_truth.loc[merged_with_truth['regression_test_loss'].astype(bool)]
                        rmse_regression = rmse(temp_regression_loss[tgt_score + '_true'], temp_regression_loss[tgt_score])
                        rmse_train = rmse(
                            merged_with_truth.loc[merged_with_truth['training'].astype(bool), tgt_score + '_true'],
                            merged_with_truth.loc[merged_with_truth['training'].astype(bool), tgt_score],
                        )
                        rmse_double_missing = rmse(
                            merged_with_truth.loc[merged_with_truth['double_missing'].astype(bool), tgt_score + '_true'],
                            merged_with_truth.loc[merged_with_truth['double_missing'].astype(bool), tgt_score],
                        )
                        rmse_src_missing_only = rmse(
                            merged_with_truth.loc[merged_with_truth['src_missing_only'].astype(bool), tgt_score + '_true'],
                            merged_with_truth.loc[merged_with_truth['src_missing_only'].astype(bool), tgt_score],
                        )

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
                            'rate': [r_val],
                            'split': [sp],
                            'src-tgt': [f"{src}->{tgt}"],
                            'model': ['mice_rf'],
                            'target_context': [target],
                            'num_test_points_for_tgt': [temp_regression_loss.shape[0]],
                            'n_training': [n_training],
                            'n_regression_test_loss': [n_regression_test_loss],
                            'n_double_missing': [n_double_missing],
                            'n_no_src_prediction': [n_no_src_prediction],
                        })
                        all_results.append(per_bucket_rmse_wide)

    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        print(f"\nCombined results shape: {combined_results.shape}")
        project_root = Path(__file__).resolve().parent.parent
        output_dir = project_root / "splits_results_0506"
        output_dir.mkdir(parents=True, exist_ok=True)
        combined_results.to_csv(output_dir / "mice_rf_loss_no_double_missing.csv", index=False)
        print(f"Saved to {output_dir / 'mice_rf_loss_no_double_missing.csv'}")
    else:
        print("No results found")
