import pandas as pd
import numpy as np
from pathlib import Path
import re




###measure losses on Linear Model outputs###
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

pattern = re.compile(
    r'^(?P<model>basic_linear|full_interaction_linear|mixed_random|oneparam_linear|full_interaction_mixed)'
    r'_(?P<src>(?:av|wt)\d+)_score_to_(?P<tgt>(?:av|wt)\d+)_score'
    r'(?:_s(?P<split>\d+)_r(?P<rate>\d+))?'
    r'(?:\.csv)?$'
)


all_results = []
rates = [10,20,40,60,80,90]
for r in rates:
    print(f"\nTest fraction r: {r}")
    linear_model_out = f"linear_model_output_{r}" #open relevant test fraction folder for model outputs
    linear_model_out = base_dir / Path(linear_model_out) #add the base directory path so that it can be found
    masked_data_folder = base_dir / Path(f"test_frac_{r}") #open the relevant training data for that test fraction
    for split_file in masked_data_folder.iterdir():

        if split_file.is_file() and split_file.name.startswith("train_split_"):
            print(f"  Split file: {split_file.name}")
            masked_data_split = pd.read_csv(split_file)
            r , sp = extract_r_s(split_file.name)
            print(f"    r,s extracted: {r}, {sp}")
            for model_out_split in linear_model_out.iterdir():
                if model_out_split.is_dir() and model_out_split.name == f"split_{sp}":
                    print("hello")
                    print(f"   Model output split found: {model_out_split.name}")
                    for file in model_out_split.iterdir():
                        if file.is_file() and file.suffix == ".csv":
                            m = pattern.search(file.name)
                            print(file.name, "->", m.group("model"), m.group("src"), m.group("tgt"), m.group("split"), m.group("rate"))

                            src_score = f"{m.group('src')}_score"
                            tgt_score = f"{m.group('tgt')}_score"
                            model_output_df = pd.read_csv(file)

                        
                            # Build boolean NA flags for src/tgt in masked_data_split
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
                            rmse_test = rmse(
                                temp_regression_loss[tgt_score + '_true'],
                                temp_regression_loss[f"{m.group('tgt')}_score"]
                            )
                            rmse_train = rmse(
                                merged_with_truth.loc[merged_with_truth['training'].astype(bool), tgt_score + '_true'],
                                merged_with_truth.loc[merged_with_truth['training'].astype(bool), f"{m.group('tgt')}_score"]
                            )

                            rmse_double_missing = rmse(
                                merged_with_truth.loc[merged_with_truth['double_missing'].astype(bool), tgt_score + '_true'],
                                merged_with_truth.loc[merged_with_truth['double_missing'].astype(bool), f"{m.group('tgt')}_score"]
                            )

                            rmse_src_missing_only = rmse(
                                merged_with_truth.loc[merged_with_truth['src_missing_only'].astype(bool), tgt_score + '_true'],
                                merged_with_truth.loc[merged_with_truth['src_missing_only'].astype(bool), f"{m.group('tgt')}_score"]
                            )
                            print(f"RMSE for regression_test_loss: {rmse_test}")
                            # Per-bucket point counts — only cells where both truth and
                            # prediction are non-NaN (matches what rmse() actually scored).
                            pred_col = f"{m.group('tgt')}_score"
                            elig = merged_with_truth[[tgt_score + '_true', pred_col]].notna().all(axis=1)
                            n_training = int((merged_with_truth['training'].astype(bool) & elig).sum())
                            n_regression_test_loss = int((merged_with_truth['regression_test_loss'].astype(bool) & elig).sum())
                            n_double_missing = int((merged_with_truth['double_missing'].astype(bool) & elig).sum())
                            n_no_src_prediction = int((merged_with_truth['src_missing_only'].astype(bool) & elig).sum())
                            per_bucket_rmse_wide = pd.DataFrame({
                                'rmse_test': [rmse_test],
                                'rmse_train': [rmse_train],
                                'rmse_double_missing': [rmse_double_missing],
                                'rmse_src_missing_only': [rmse_src_missing_only],
                                'rate': [r],
                                'split': [sp],
                                'src-tgt': [f"{m.group('src')}->{m.group('tgt')}"],
                                'model': [m.group('model')],
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
    combined_results
else:
    print("No results found")


project_root = Path(__file__).resolve().parent.parent
combined_results.to_csv(project_root / "linear_model_loss_measurements_all_splits_rates2.csv", index=False)