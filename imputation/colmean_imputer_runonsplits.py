import itertools
import pandas as pd
from pathlib import Path
import argparse
import re
from split_generator import load_pair

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run column mean imputation across pre-generated splits.")
    parser.add_argument("--base-dir", default="../data_splits", help="Base directory containing split CSVs (train/mask).")
    parser.add_argument(
        "--out-subdirectory",
        default="mean_imputed_{missing_rate}/split_{split_number}",
        help="Output subdirectory template. You can use {split_number} and {missing_rate} placeholders."
    )
    parser.add_argument("--num-splits", type=int, default=50, help="Number of split indices to process (starting at 1).")
    parser.add_argument("--missing-rate", type=int, default=10, help="Missing rate used when selecting split files.")
    args = parser.parse_args()

    base_dir = args.base_dir
    num_splits = args.num_splits
    missing_rate = args.missing_rate

    for split_number in range(1, num_splits + 1):
        try:
            train_df, mask_df, meta = load_pair(base_dir=base_dir, s=split_number, rate=missing_rate)
        except Exception as e:
            raise Exception(
                f"Relevant Train/Mask splits not found for split={split_number}, rate={missing_rate} in {base_dir}"
            ) from e

        train_df['aa_pos'] = pd.to_numeric(train_df['aa_pos'])
        protein_len = int(train_df['aa_pos'].max())

        ###mean imputation per column####
        train_df['aa_pos'] = pd.to_numeric(train_df['aa_pos'])
        protein_len = int(train_df['aa_pos'].max())
        numeric_cols = [
        c for c in train_df.select_dtypes(include=['number']).columns
        if c.endswith('score') or c.endswith('se')
        ]

        for col in numeric_cols:
            per_group_mean = train_df.groupby('pos_aminoacid')[col].transform('mean')
            col_global_mean = train_df[col].mean()
            train_df[col] = train_df[col].fillna(per_group_mean)
            train_df[col] = train_df[col].fillna(col_global_mean)
        out_subdirectory = Path(
            args.out_subdirectory.format(
                split_number=split_number,
                missing_rate=missing_rate,
            )
        )
        out_dir = Path(base_dir) / out_subdirectory
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f'mean_imputed_split{split_number}.csv'
        print(out_path)
        train_df.to_csv(out_path, index=False)
