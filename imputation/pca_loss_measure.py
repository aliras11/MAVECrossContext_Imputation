"""Loss measurement for PCA within-map imputer outputs.
Reads from pca_k{n_comp}_testfrac{r}/split_{s}/ folders.
Measures test RMSE (held-out entries) and training RMSE per context.
Output format matches colmean loss measure.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import re

full_data_df = pd.read_csv(Path(__file__).resolve().parent.parent / "full_data" / "mthfr_crossAllcontext_domainannotation.csv")
base_dir = Path(__file__).resolve().parent.parent / "data_splits"

def extract_r_s(name: str):
    rm = re.search(r"r(\d+)", name, flags=re.IGNORECASE)
    sm = re.search(r"s(\d+)", name, flags=re.IGNORECASE)
    r = int(rm.group(1)) if rm else None
    s = int(sm.group(1)) if sm else None
    return r, s

def rmse(y_true, y_pred):
    pair = pd.concat([y_true, y_pred], axis=1).dropna()
    if pair.empty:
        return float("nan")
    diff = pair.iloc[:, 0].to_numpy() - pair.iloc[:, 1].to_numpy()
    return float(np.sqrt(np.mean(diff * diff)))

n_components_list = [1, 2, 4, 10, 20, 22]
rates = [10, 20, 40, 60, 80, 90]
score_cols = [f'{ctx}_score' for ctx in
              ['av12', 'av25', 'av100', 'av200', 'wt12', 'wt25', 'wt100', 'wt200']]

all_results = []

for n_comp in n_components_list:
    for r in rates:
        folder_name = f"pca_k{n_comp}_testfrac{r}"
        pca_folder = base_dir / folder_name
        if not pca_folder.exists():
            print(f"Skipping k={n_comp} rate={r}: {pca_folder} not found")
            continue

        masked_data_folder = base_dir / f"test_frac_{r}"
        for split_file in masked_data_folder.iterdir():
            if not (split_file.is_file() and split_file.name.startswith("train_split_")):
                continue

            masked_data_split = pd.read_csv(split_file)
            r_val, sp = extract_r_s(split_file.name)

            for model_out_split in pca_folder.iterdir():
                if not (model_out_split.is_dir() and model_out_split.name == f"split_{sp}"):
                    continue

                for file in model_out_split.iterdir():
                    if not (file.is_file() and file.suffix == ".csv"):
                        continue

                    model_output_df = pd.read_csv(file)
                    merged_df = full_data_df.merge(
                        model_output_df,
                        on="hgvs_pro",
                        suffixes=("_true", "_pred"),
                    ).merge(
                        masked_data_split[["hgvs_pro"] + score_cols],
                        on="hgvs_pro",
                    )

                    for col in score_cols:
                        ctx = col.replace('_score', '')
                        true_col = f"{col}_true"
                        pred_col = f"{col}_pred"
                        test_points = merged_df[col].isna()
                        train_points = merged_df[col].notna()

                        test_loss = rmse(
                            merged_df.loc[test_points, true_col],
                            merged_df.loc[test_points, pred_col])
                        train_loss = rmse(
                            merged_df.loc[train_points, true_col],
                            merged_df.loc[train_points, pred_col])

                        # Per-bucket point counts — only cells where both truth and
                        # prediction are non-NaN (matches what rmse() actually scored).
                        elig = merged_df[[true_col, pred_col]].notna().all(axis=1)
                        all_results.append({
                            'test': test_loss,
                            'training': train_loss,
                            'n_test': int((test_points & elig).sum()),
                            'n_training': int((train_points & elig).sum()),
                            'rate': r_val,
                            'split': sp,
                            'src-tgt': f'{ctx}-to-{ctx}',
                            'n_components': n_comp,
                        })

print(f"\nTotal results: {len(all_results)}")

if all_results:
    combined = pd.DataFrame(all_results)
    project_root = Path(__file__).resolve().parent.parent

    # Save one combined CSV with all component counts
    combined.to_csv(project_root / "pca_rmse_results_all.csv", index=False)
    print(f"Saved pca_rmse_results_all.csv ({len(combined)} rows)")

    # Also save per-component CSVs for easy loading
    for nc in n_components_list:
        sub = combined[combined['n_components'] == nc]
        if len(sub) > 0:
            sub.to_csv(project_root / f"pca_k{nc}_rmse_results.csv", index=False)
            print(f"Saved pca_k{nc}_rmse_results.csv ({len(sub)} rows)")
else:
    print("No results found")
