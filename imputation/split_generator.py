'''script meant to take full dataset of MTHFR scores and generate train split csv files
with injected missingness for imputation experiments - can recover test points by comparing
to full dataset - set output dir to a folder outside of where the script is being invoked'''

import pandas as pd
import numpy as np
import re
from pathlib import Path
import os 



def load_pair(base_dir='data_splits', s=None, rate=None):
    """
    Load a train/mask pair by split number s (e.g., 7 or 's07'), optionally filtered by rate (e.g., 10 or 20).
    Returns (train_df, mask_df, meta)
    """
    if s is None:
        raise ValueError("Provide s (split number), e.g., 7 or 's07'.")
    s_int = int(str(s).lstrip('sS').lstrip('0') or '0')

    base = Path(base_dir)
    pat = re.compile(r'^train_split_(r(\d+)_s0*(\d+))\.csv$')
    candidates = []

    for train_path in base.rglob('train_split_*.csv'):
        m = pat.match(train_path.name)
        if not m:
            continue
        rate_found = int(m.group(2))
        s_found = int(m.group(3))
        if s_found != s_int:
            continue
        if rate is not None and rate_found != int(rate):
            continue
        split_name = m.group(1)
        mask_path = train_path.with_name(f'mask_{split_name}.csv')
        if mask_path.exists():
            candidates.append({
                'rate': rate_found,
                's': s_found,
                'split_name': split_name,
                'train': train_path,
                'mask': mask_path
            })

    if not candidates:
        raise FileNotFoundError("No matching train/mask pair found.")
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous match, refine with rate. Candidates: {[e['split_name'] for e in candidates]}")
    e = candidates[0]
    return pd.read_csv(e['train']), pd.read_csv(e['mask']), e



def generate_one_split(data_full, missing_rate, split_name, output_dir='data_splits', seed=None):
    '''generate one train split with injected missingness'''
    if seed:
        rng = np.random.default_rng(seed) 
    else:
        rng = np.random.default_rng()
    test_data = data_full.copy()
    score_cols = [c for c in data_full.columns if isinstance(c, str) and c.endswith('_score')]
    se_cols = [c for c in data_full.columns if isinstance(c, str) and c.endswith('_se')]
    for score_col,se_col in zip(score_cols, se_cols):
        true_missing_indices = data_full.index[data_full[score_col].isna()]
        non_missing_indices = test_data.index[~test_data[score_col].isna()]
        n_inject = int(np.ceil(missing_rate * len(non_missing_indices)))
        inject_indices = rng.choice(non_missing_indices, size=n_inject, replace=False)
        inject_indices = inject_indices.tolist()
        test_data.loc[inject_indices, score_col] = pd.NA
        test_data.loc[inject_indices,se_col] = pd.NA
        assert len(true_missing_indices) + n_inject == test_data[score_col].isna().sum(), "Unexpected number of missing values after injection!"
    assert test_data.isna().sum().sum() > data_full.isna().sum().sum(), "No missingness was injected!"
    assert test_data.shape == data_full.shape, "Data shape changed after injecting missingness!"
    assert test_data.index.equals(data_full.index), "Data indices changed after injecting missingness!"
    os.makedirs(output_dir, exist_ok=True)
    output_path = Path(output_dir) / f'train_split_{split_name}.csv'
    test_data.to_csv(output_path, index=False)
    print(f'Split {split_name} saved to {output_path}')
    return test_data 


def create_test_mask(data_full, train_data):
    '''create a mask indicating identifying injected NA/missing values'''
    score_cols = [c for c in data_full.columns if isinstance(c, str) and c.endswith('_score')]
    se_cols = [c for c in data_full.columns if isinstance(c, str) and c.endswith('_se')]
    added_na_indices = {col: [] for col in score_cols + se_cols}
    missing_count_per_col = []
    for score_col,se_col in zip(score_cols,se_cols):
        true_missing_indices = data_full.index[data_full[score_col].isna()]
        all_missing_indices = train_data.index[train_data[score_col].isna()]
        injected_missing_indices = set(all_missing_indices) - set(true_missing_indices) #set diff to identify injected missingness
        added_na_indices[score_col] = list(injected_missing_indices)
        added_na_indices[se_col] = list(injected_missing_indices)
        missing_count_per_col.append(len(injected_missing_indices))
        missing_count_per_col.append(len(injected_missing_indices))

        print(f"Injected missing values for {score_col}: {len(injected_missing_indices)}")
    mask_df = pd.DataFrame(0, index=data_full.index, columns=data_full.columns, dtype='int8')
    ####!!!!!!keep the original values for all non-score and non-se columns!!!!#######
    mask_df[data_full.columns[~data_full.columns.str.endswith(('_score', '_se'), na=False)]] = data_full[data_full.columns[~data_full.columns.str.endswith(('_score', '_se'), na=False)]]
    
    # place 1s at listed row indices per column
    for col, rows in added_na_indices.items():
        if col in mask_df.columns:
            valid_rows = mask_df.index.intersection(rows)  # ensure indices exist
            if len(valid_rows):
                mask_df.loc[valid_rows, col] = 1  
    ####sum over only score and se columns to check injected missingness counts#####
    missing_mask_sums = mask_df[data_full.columns[data_full.columns.str.endswith(('_score', '_se'), na=False)]].sum(axis=0, numeric_only=True).to_list()
    missing_counts = [i for i in missing_mask_sums if i > 0]
    assert missing_counts == missing_count_per_col, "Mismatch in expected missing counts!"
    return mask_df # a 1 in this mask indicates an injected missing value

def generate_splits(data_full, missing_rates, n_splits_per_rate, output_dir='data_splits', base_seed=None):
    '''generate multiple train splits with varying levels of injected missingness'''
    for rate in missing_rates:
        output_dir2 = Path(output_dir) / Path(f'test_frac_{int(rate*100)}')
        for split_num in range(n_splits_per_rate):
            split_name = f'r{int(rate*100)}_s{split_num+1}'
            seed = base_seed + split_num  # ensure different splits have different seeds
            print(f'Generating split {split_name} with missing rate {rate}')
            train_data = generate_one_split(data_full, rate, split_name, output_dir2, seed)
            mask_df = create_test_mask(data_full, train_data)
            mask_output_path = Path(output_dir2) / f'mask_{split_name}.csv'
            mask_df.to_csv(mask_output_path, index=False)
            print(f'Mask for split {split_name} saved to {mask_output_path}')


if __name__ == "__main__":
    import argparse


    parser = argparse.ArgumentParser(description="Generate imputation train splits and masks from a full dataset.")
    parser.add_argument("--full-data", required=True, help="Path to the full dataset CSV.")
    parser.add_argument("--n-splits", type=int, required=True, help="Number of splits to generate per missing rate.")
    parser.add_argument(
        "--rate", dest="rates", action="append", default=None,
        help="Missing rate per split. Can be provided multiple times. Accepts percent (e.g., 10 or 10%) or fraction (e.g., 0.1). Default: 10"
    )
    parser.add_argument("--output-dir", default="data_splits", help="Output directory for splits and masks.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for RNG.")
    args = parser.parse_args()

    # Parse rates into fractions in (0,1]
    parsed_rates = []
    if not args.rates:
        args.rates = ['10']  # default to 10%
    for r in args.rates:
        r_str = str(r).strip()
        if r_str.endswith("%"):
            val = float(r_str[:-1]) / 100.0
        else:
            val = float(r_str)
            if val > 1.0:
                val = val / 100.0
        if not (0 < val <= 1):
            raise ValueError(f"Invalid rate {r}: provide 0<rate<=1 or percent.")
        parsed_rates.append(val)

    df_full_cli = pd.read_csv(Path(args.full_data))
    generate_splits(
        data_full=df_full_cli,
        missing_rates=parsed_rates,
        n_splits_per_rate=args.n_splits,
        output_dir=args.output_dir,
        base_seed=args.seed,
    )