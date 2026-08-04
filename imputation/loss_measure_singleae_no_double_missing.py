"""Measure losses on SingleAE outputs from no-double-missing splits.

Usage:
    python loss_measure_singleae_no_double_missing.py
    python loss_measure_singleae_no_double_missing.py --target av12 --rates 10 40
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import argparse

from runtime_paths import FULL_DATA_CSV, LOSS_RESULTS_DIR, NODOUBLE_SPLITS_DIR

CONTEXTS = ['av12', 'av25', 'av100', 'av200', 'wt12', 'wt25', 'wt100', 'wt200']

full_data_df = pd.read_csv(FULL_DATA_CSV)
default_base_dir = NODOUBLE_SPLITS_DIR

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
            ae_folder = tgt_dir / f"single_AE3_testfrac{r}"
            masked_data_folder = tgt_dir / f"test_frac_{r}"

            if not masked_data_folder.exists():
                print(f"  Skipping — {masked_data_folder} not found")
                continue

            for split_file in sorted(masked_data_folder.iterdir()):
                if not (split_file.is_file() and split_file.name.startswith("train_split_")):
                    continue

                masked_data_split = pd.read_csv(split_file)
                r_val, sp = extract_r_s(split_file.name)

                if not ae_folder.exists():
                    print(f"  Skipping — {ae_folder} not found")
                    continue

                for model_out_split in sorted(ae_folder.iterdir()):
                    if not (model_out_split.is_dir() and model_out_split.name == f"split_{sp}"):
                        continue

                    for file in sorted(model_out_split.iterdir()):
                        if not (file.is_file() and file.suffix == ".csv"):
                            continue

                        src, tgt = re.search(r'([^_]+)_to_([^_]+)', file.name).groups()
                        model_output_df = pd.read_csv(file)
                        src_score = f"{src}_score"
                        tgt_score = f"{tgt}_score"

                        s = masked_data_split[src_score].notna()
                        t = masked_data_split[tgt_score].notna()

                        if src == tgt:
                            model_output_df['loss_mask'] = np.select(
                                [t, ~t], ["training", "test"], default="unknown"
                            )
                        else:
                            model_output_df['loss_mask'] = np.select(
                                [~s & ~t, s & ~t, s & t, ~s & t],
                                ["double_missing", "regression_test_loss", "training", "no_src_prediction"],
                                default="unknown"
                            )

                        temp_df = (
                            full_data_df[["hgvs_pro", tgt_score]]
                            .merge(model_output_df[["hgvs_pro", f"{tgt}_imputed", "loss_mask", 'seen_in_training_tgt']],
                                   on="hgvs_pro", how="inner")
                        )

                        per_bucket_rmse = (
                            temp_df.groupby("loss_mask")
                            .apply(lambda df: rmse(df[tgt_score], df[f"{tgt}_imputed"]), include_groups=False)
                            .rename("rmse")
                        )

                        temp_df2 = temp_df.dropna(subset=["seen_in_training_tgt"]).copy()
                        temp_df2["seen_in_training_tgt"] = temp_df2["seen_in_training_tgt"].astype(bool)
                        non_train_mask = ~temp_df2["seen_in_training_tgt"]
                        rmse_non_training = rmse(
                            temp_df2.loc[non_train_mask, tgt_score],
                            temp_df2.loc[non_train_mask, f"{tgt}_imputed"],
                        )

                        per_bucket_rmse["rmse_non_training"] = rmse_non_training
                        per_bucket_rmse_wide = per_bucket_rmse.to_frame().T

                        # Per-bucket point counts — only cells where both truth and
                        # prediction are non-NaN (matches what rmse() actually scored).
                        bucket_counts = (
                            temp_df.dropna(subset=[tgt_score, f"{tgt}_imputed"])
                                   .groupby("loss_mask").size()
                        )
                        for bucket_name, count in bucket_counts.items():
                            per_bucket_rmse_wide[f'n_{bucket_name}'] = int(count)

                        per_bucket_rmse_wide['rate'] = r_val
                        per_bucket_rmse_wide['split'] = sp
                        per_bucket_rmse_wide['src-tgt'] = f"{src}-to-{tgt}"
                        per_bucket_rmse_wide['target_context'] = target
                        per_bucket_rmse_wide['num_test_points_for_tgt'] = non_train_mask.sum()

                        all_results.append(per_bucket_rmse_wide)

    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        print(f"\nCombined results shape: {combined_results.shape}")
        output_dir = LOSS_RESULTS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        combined_results.to_csv(output_dir / "single_AE3_rmse_no_double_missing.csv", index=False)
        print(f"Saved to {output_dir / 'single_AE3_rmse_no_double_missing.csv'}")
    else:
        print("No results found")
