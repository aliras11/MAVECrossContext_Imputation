import pandas as pd
import numpy as np
from pathlib import Path
import re




###measure losses on Column Mean outputs###
full_data_df = pd.read_csv(Path(__file__).resolve().parent.parent / "full_data" / "mthfr_crossAllcontext_domainannotation.csv")
base_dir = Path(__file__).resolve().parent.parent / "data_splits"

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



for rate in rates:
    print(f"\nTest fraction r: {rate}")
    mean_impute_model_out = f"mean_imputed_{rate}" #open relevant test fraction folder for model outputs
    mean_impute_model_out = base_dir / Path(mean_impute_model_out) #add the base directory path so that it can be found
    masked_data_folder = base_dir / Path(f"test_frac_{rate}") #open the relevant training data for that test fraction
    for split_file in masked_data_folder.iterdir():

        if split_file.is_file() and split_file.name.startswith("train_split_"):
            print(f"  Split file: {split_file.name}")
            masked_data_split = pd.read_csv(split_file)
            r , sp = extract_r_s(split_file.name)
            print(f"    r,s extracted: {r}, {sp}")
            for model_out_split in mean_impute_model_out.iterdir():
                if model_out_split.is_dir() and model_out_split.name == f"split_{sp}":
                    print(f"   Model output split found: {model_out_split.name}")
                    for file in model_out_split.iterdir():
                        if file.is_file() and file.suffix == ".csv":
                            print(f"      File: {file.name}")
                            model_output_df = pd.read_csv(file)
                            numeric_cols = [
                                c for c in model_output_df.select_dtypes(include=['number']).columns
                                if c.endswith('score') or c.endswith('se')]
                            merged_df = full_data_df.merge(
                                model_output_df,
                                on="hgvs_pro",
                                suffixes=("_true", "_pred"),
                            ).merge(
                                masked_data_split[["hgvs_pro"]+numeric_cols],
                                on="hgvs_pro",
                            ) #dont need a suffix here since these arent overlapping columns, merged to ensure alignment

                            for col in numeric_cols:
                                true_col = f"{col}_true"
                                pred_col = f"{col}_pred"
                                mask_to_measure_only_test_points = merged_df[col].isna()
                                loss = rmse(merged_df.loc[mask_to_measure_only_test_points,true_col], merged_df.loc[mask_to_measure_only_test_points,pred_col])
                                # Per-bucket point counts — only cells where both truth and
                                # prediction are non-NaN (matches what rmse() actually scored).
                                elig = merged_df[[true_col, pred_col]].notna().all(axis=1)
                                n_test = int((mask_to_measure_only_test_points & elig).sum())
                                n_training = int((~mask_to_measure_only_test_points & elig).sum())
                                result = {
                                    "test_fraction": rate,
                                    "split": sp,
                                    "model_file": file.name,
                                    "metric": "rmse",
                                    "column": col,
                                    "loss": loss,
                                    "n_test": n_test,
                                    "n_training": n_training,
                                }
                                all_results.append(result)
                                print(f"        Column: {col}, RMSE: {loss:.4f}")
                            

if all_results:
    project_root = Path(__file__).resolve().parent.parent
    save_path = project_root / "col_mean_imputed_results.csv"
    combined_results = pd.DataFrame(all_results)
    combined_results.to_csv(save_path, index=False)
    print(f"Saved combined results with {len(combined_results)} rows")
