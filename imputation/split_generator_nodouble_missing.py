'''Script to generate train split CSV files with injected missingness in only ONE
target context column at a time, leaving all other columns (potential sources)
untouched.  This eliminates the "double missing" scenario: for any src→tgt pair
the source is always fully observed (modulo natural missingness).

Output structure:
  {output_dir}/tgt_{target}/test_frac_{rate}/train_split_r{rate}_s{N}.csv
  {output_dir}/tgt_{target}/test_frac_{rate}/mask_r{rate}_s{N}.csv
'''

import pandas as pd
import numpy as np
import os
from pathlib import Path

from split_generator import load_pair, create_test_mask


CONTEXTS = ['av12', 'av25', 'av100', 'av200', 'wt12', 'wt25', 'wt100', 'wt200']


def rate_to_label(rate):
    """Convert fractional rate to integer label for folder/file names.
    Handles sub-percent precision: 0.999 -> 999, 0.99 -> 99, 0.10 -> 10."""
    pct = rate * 100
    if abs(pct - round(pct)) < 0.01:
        return int(round(pct))
    return int(round(rate * 1000))


def generate_one_split_single_target(data_full, missing_rate, split_name,
                                     target_context, output_dir='data_splits_no_double_missing',
                                     seed=None):
    '''Generate one train split with injected missingness in only the target context.'''
    rng = np.random.default_rng(seed) if seed else np.random.default_rng()

    test_data = data_full.copy()
    score_col = f'{target_context}_score'
    se_col = f'{target_context}_se'

    assert score_col in data_full.columns, f"Target score column {score_col} not found"
    assert se_col in data_full.columns, f"Target SE column {se_col} not found"

    true_missing_indices = data_full.index[data_full[score_col].isna()]
    non_missing_indices = test_data.index[~test_data[score_col].isna()]
    n_inject = int(np.ceil(missing_rate * len(non_missing_indices)))
    inject_indices = rng.choice(non_missing_indices, size=n_inject, replace=False).tolist()
    test_data.loc[inject_indices, score_col] = pd.NA
    test_data.loc[inject_indices, se_col] = pd.NA

    assert len(true_missing_indices) + n_inject == test_data[score_col].isna().sum(), \
        "Unexpected number of missing values after injection!"
    assert test_data.isna().sum().sum() > data_full.isna().sum().sum(), \
        "No missingness was injected!"
    assert test_data.shape == data_full.shape, \
        "Data shape changed after injecting missingness!"
    assert test_data.index.equals(data_full.index), \
        "Data indices changed after injecting missingness!"

    # Verify other score columns are untouched
    other_score_cols = [c for c in data_full.columns
                        if c.endswith('_score') and c != score_col]
    for col in other_score_cols:
        assert test_data[col].isna().sum() == data_full[col].isna().sum(), \
            f"Column {col} was modified but should be untouched!"

    os.makedirs(output_dir, exist_ok=True)
    output_path = Path(output_dir) / f'train_split_{split_name}.csv'
    test_data.to_csv(output_path, index=False)
    print(f'Split {split_name} (target={target_context}) saved to {output_path}')
    return test_data


def create_test_mask_single_target(data_full, train_data):
    '''Create a mask identifying injected NA/missing values.
    Works correctly when only a subset of columns have injected missingness
    (fixes assertion from original create_test_mask that assumed all columns masked).
    '''
    score_cols = [c for c in data_full.columns
                  if isinstance(c, str) and c.endswith('_score')]
    se_cols = [c for c in data_full.columns
               if isinstance(c, str) and c.endswith('_se')]

    added_na_indices = {col: [] for col in score_cols + se_cols}
    missing_count_per_col = []

    for score_col, se_col in zip(score_cols, se_cols):
        true_missing_indices = data_full.index[data_full[score_col].isna()]
        all_missing_indices = train_data.index[train_data[score_col].isna()]
        injected_missing_indices = set(all_missing_indices) - set(true_missing_indices)
        added_na_indices[score_col] = list(injected_missing_indices)
        added_na_indices[se_col] = list(injected_missing_indices)
        missing_count_per_col.append(len(injected_missing_indices))
        missing_count_per_col.append(len(injected_missing_indices))

        if len(injected_missing_indices) > 0:
            print(f"Injected missing values for {score_col}: {len(injected_missing_indices)}")

    mask_df = pd.DataFrame(0, index=data_full.index, columns=data_full.columns, dtype='int8')
    # Keep original values for non-score/se columns
    non_score_se = data_full.columns[~data_full.columns.str.endswith(('_score', '_se'), na=False)]
    mask_df[non_score_se] = data_full[non_score_se]

    for col, rows in added_na_indices.items():
        if col in mask_df.columns:
            valid_rows = mask_df.index.intersection(rows)
            if len(valid_rows):
                mask_df.loc[valid_rows, col] = 1

    # Verify: compare per-column sums (including zeros) to expected counts
    score_se_cols = data_full.columns[data_full.columns.str.endswith(('_score', '_se'), na=False)]
    missing_mask_sums = mask_df[score_se_cols].sum(axis=0, numeric_only=True).to_list()
    assert missing_mask_sums == missing_count_per_col, \
        f"Mismatch in expected missing counts!\n  mask sums: {missing_mask_sums}\n  expected:  {missing_count_per_col}"

    return mask_df


def generate_splits_nodouble(data_full, missing_rates, n_splits_per_rate,
                             output_dir='data_splits_no_double_missing',
                             base_seed=None, target_contexts=None):
    '''Generate splits with missingness injected into one target column at a time.'''
    if target_contexts is None:
        target_contexts = CONTEXTS

    for target in target_contexts:
        print(f'\n=== Target context: {target} ===')
        for rate in missing_rates:
            tgt_rate_dir = Path(output_dir) / f'tgt_{target}' / f'test_frac_{rate_to_label(rate)}'
            for split_num in range(n_splits_per_rate):
                split_name = f'r{rate_to_label(rate)}_s{split_num + 1}'
                seed = base_seed + split_num

                print(f'Generating split {split_name} target={target} rate={rate}')
                train_data = generate_one_split_single_target(
                    data_full, rate, split_name, target, tgt_rate_dir, seed)

                mask_df = create_test_mask_single_target(data_full, train_data)
                mask_output_path = tgt_rate_dir / f'mask_{split_name}.csv'
                mask_df.to_csv(mask_output_path, index=False)
                print(f'Mask for split {split_name} saved to {mask_output_path}')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate no-double-missing splits: inject missingness into one target column only.")
    parser.add_argument("--full-data", required=True,
                        help="Path to the full dataset CSV.")
    parser.add_argument("--n-splits", type=int, required=True,
                        help="Number of splits per rate per target.")
    parser.add_argument("--rate", dest="rates", action="append", default=None,
                        help="Missing rate (percent or fraction). Can repeat. Default: 10 40 80 99")
    parser.add_argument("--output-dir", default="data_splits_no_double_missing",
                        help="Output base directory.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed for RNG.")
    parser.add_argument("--target", dest="targets", action="append", default=None,
                        help="Target context to generate for. Can repeat. Default: all 8.")
    args = parser.parse_args()

    # Parse rates
    if not args.rates:
        args.rates = ['10', '40', '80', '99']
    parsed_rates = []
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

    # Parse targets
    target_contexts = None
    if args.targets:
        for t in args.targets:
            if t not in CONTEXTS:
                raise ValueError(f"Unknown target context '{t}'. Valid: {CONTEXTS}")
        target_contexts = args.targets

    df_full = pd.read_csv(Path(args.full_data))

    generate_splits_nodouble(
        data_full=df_full,
        missing_rates=parsed_rates,
        n_splits_per_rate=args.n_splits,
        output_dir=args.output_dir,
        base_seed=args.seed,
        target_contexts=target_contexts,
    )
