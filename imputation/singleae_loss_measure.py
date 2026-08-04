import pandas as pd
import numpy as np
from pathlib import Path
import re


###measure losses on Single AE outputs###
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
rates = [10, 20, 40, 60, 80, 90]
for r in rates:
    print(f"\nTest fraction r: {r}")
    single_ae_test_frac_folder = f"single_AE3_testfrac{r}" #open relevant test fraction folder for model outputs
    single_ae_test_frac_folder = base_dir / Path(single_ae_test_frac_folder) #add the base directory path so that it can be found
    masked_data_folder = base_dir / Path(f"test_frac_{r}") #open the relevant training data for that test fraction
    for split_file in masked_data_folder.iterdir():

        if split_file.is_file() and split_file.name.startswith("train_split_"):
            print(f"  Split file: {split_file.name}")
            masked_data_split = pd.read_csv(split_file)
            r , sp = extract_r_s(split_file.name)
            print(f"    r,s extracted: {r}, {sp}")
            for model_out_split in single_ae_test_frac_folder.iterdir():
                if model_out_split.is_dir() and model_out_split.name == f"split_{sp}":
                    print(f"    Model output split found: {model_out_split.name}")
                    for file in model_out_split.iterdir():
                        if file.is_file() and file.suffix == ".csv":
                            src, tgt = re.search(r'([^_]+)_to_([^_]+)', file.name).groups()
                            print(f"      File: {file.name} -> src: {src}, tgt: {tgt}")
                            model_output_df = pd.read_csv(file)
                            src_score = f"{src}_score"
                            tgt_score = f"{tgt}_score"
                            s = masked_data_split[src_score].notna() #gets double inverted later on
                            t = masked_data_split[tgt_score].notna()
                            if src == tgt:
                               model_output_df['loss_mask'] = np.select(
                                   [t, ~t],
                                   ["training", "test"], default="unknown"
                               )
                            else:
                                model_output_df['loss_mask'] = np.select(
                                    [~s & ~t,  s & ~t,  s & t,  ~s & t], #not s and not t means was NA in src and tgt
                                    ["double_missing", "regression_test_loss", "training", "no_src_prediction"], default="unknown"
                                )
                          
                            full_data_df[['hgvs_pro', src_score, tgt_score]]
                            temp_df = (
                                full_data_df[["hgvs_pro", tgt_score]]
                                .merge(model_output_df[["hgvs_pro", f"{tgt}_imputed", "loss_mask", 'seen_in_training_tgt']], on="hgvs_pro", how="inner")
                            )
                            
                            # RMSE per loss bucket
                            per_bucket_rmse = (
                            temp_df.groupby("loss_mask")
                                .apply(lambda df: rmse(df[tgt_score], df[f"{tgt}_imputed"]),include_groups=False)
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
                            per_bucket_rmse_wide['rate'] = r
                            per_bucket_rmse_wide['split'] = sp
                            per_bucket_rmse_wide['src-tgt'] = f"{src}-to-{tgt}"
                            per_bucket_rmse_wide['num_test_points_for_tgt'] = non_train_mask.sum()

                            all_results.append(per_bucket_rmse_wide)
               

if all_results:
    combined_results = pd.concat(all_results, ignore_index=True)
    print(f"\nCombined results shape: {combined_results.shape}")
    combined_results
else:
    print("No results found")    

project_root = Path(__file__).resolve().parent.parent
output_dir = project_root / "splits_results_0506"
output_dir.mkdir(parents=True, exist_ok=True)
combined_results.to_csv(output_dir / "single_AE3_rmse_results.csv", index=True)
