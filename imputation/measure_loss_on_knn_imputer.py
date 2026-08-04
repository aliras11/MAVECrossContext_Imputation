##measure loss on knn blosum imputed data##
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import re


parser = argparse.ArgumentParser(description="Measure RMSE on KNN-BLOSUM imputed splits.")
parser.add_argument(
    "--sim_mode",
    choices=["diff", "direct"],
    default="diff",
    help="Similarity mode used during imputation. Determines which output folders to read.",
)
args = parser.parse_args()

full_data_df = pd.read_csv(Path(__file__).resolve().parent.parent / "full_data" / "mthfr_crossAllcontext_domainannotation.csv")
base_dir = Path(__file__).resolve().parent.parent / "data_splits"

sim_tag = f"_{args.sim_mode}" if args.sim_mode != "diff" else ""


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
    knn_impute_model_out = f"blosum_knn_{r}_BLOSUM100_k5_w0{sim_tag}" #open relevant test fraction folder for model outputs
    knn_impute_model_out = base_dir / Path(knn_impute_model_out) #add the base directory path so that it can be found
    print(f"Model output directory: {knn_impute_model_out}")
    masked_data_folder = base_dir / Path(f"test_frac_{r}") #open the relevant training data for that test fraction
    for model_out_split in knn_impute_model_out.iterdir():
        print(f"   Checking model output split: {model_out_split.name}")
        split_number_match = re.search(r"split_(\d+)", model_out_split.name)
        sp = int(split_number_match.group(1)) if split_number_match else None
        print(f"    Split number extracted: {sp}")
        split_file_name = masked_data_folder / f"train_split_r{r}_s{sp}.csv"
        masked_data_split = pd.read_csv(split_file_name)
        if model_out_split.is_dir() and model_out_split.name == f"split_{sp}":
            print(f"   Model output split found: {model_out_split.name}")
            for file in model_out_split.iterdir():
                if file.is_file() and file.suffix == ".csv":
                    print(f"      File: {file.name}")
                    model_output_df = pd.read_csv(file)
                    imputed_cols = ["hgvs_pro"] + [c for c in model_output_df.columns if c.endswith("_imputed")]
                    model_output_df = pd.read_csv(file)

                    # keep only hgvs_pro + imputed columns
                    imputed_cols = ["hgvs_pro"] + [c for c in model_output_df.columns if c.endswith("_imputed")]
                    model_output_df = model_output_df[imputed_cols]

                    numeric_cols = [
                        c for c in model_output_df.select_dtypes(include=["number"]).columns
                        if c.endswith("score_imputed") or c.endswith("se_imputed")
                    ]

                    for col in numeric_cols:
                        base_col = col.replace("_imputed", "")  # e.g. wt12_score_imputed -> wt12_score

                        # Merge the relevant column from each dataset on hgvs_pro (OUTER so no rows dropped)
                        df = (
                            full_data_df[["hgvs_pro", base_col]]
                            .rename(columns={base_col: "true"})
                            .merge(
                                model_output_df[["hgvs_pro", col]].rename(columns={col: "pred"}),
                                on="hgvs_pro",
                                how="outer",
                            )
                            .merge(
                                masked_data_split[["hgvs_pro", base_col]].rename(columns={base_col: "split_val"}),
                                on="hgvs_pro",
                                how="outer",
                            )
                        )

                        # test points = missing in the injected train split
                        test_mask = df["split_val"].isna()

                        loss = rmse(df.loc[test_mask, "true"], df.loc[test_mask, "pred"])

                        # Per-bucket point counts — only cells where both truth and
                        # prediction are non-NaN (matches what rmse() actually scored).
                        elig = df[["true", "pred"]].notna().all(axis=1)
                        n_test = int((test_mask & elig).sum())
                        n_training = int((~test_mask & elig).sum())
                        all_results.append({
                            "test_fraction": r,
                            "split": sp,
                            "model_file": file.name,
                            "metric": "rmse",
                            "column": col,
                            "loss": loss,
                            "n_test": n_test,
                            "n_training": n_training,
                        })

                        print(f"        Column: {col}, RMSE: {loss:.4f}")
results_df = pd.DataFrame(all_results)
project_root = Path(__file__).resolve().parent.parent
out_path = project_root / f"blosum_knn{sim_tag}_rmse_all_splits.csv"
results_df.to_csv(out_path, index=False)
print(f"Saved {len(results_df)} rows -> {out_path}")