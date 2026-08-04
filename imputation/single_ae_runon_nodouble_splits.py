"""No-double-missing variant of single_ae_runon_splits.py (SimpleAE).

Only the target column has injected missingness; source columns are fully observed.
Iterates 7 source contexts for the given target.

Usage:
    python single_ae_runon_nodouble_splits.py --base-dir ../data_splits_no_double_missing \
        --missing-rate 10 --target av12 --num-splits 3
"""

import re
import pandas as pd
from pathlib import Path
import argparse

from preprocess import (
    csv_to_maskedarray,
    maskedarray_to_torch_tensor,
    place_ones_wt_aa_pos,
    place_trues_wt_aa_pos,
    mthfr_proteinseq_wt,
    mthfr_proteinseq_alt,
    amino_acid_dict_num_to_3,
)
from training_regimes import train_cross_map_with_simpleae
from performance_measurement import tensor_to_hgvs_dataframe
from split_generator import load_pair

CONTEXTS = ['av12', 'av25', 'av100', 'av200', 'wt12', 'wt25', 'wt100', 'wt200']


def get_proteinseq(name):
    """Return the appropriate protein sequence for a context name."""
    if re.match(r'(?i)^av', name):
        return mthfr_proteinseq_alt
    return mthfr_proteinseq_wt


def build_maps(train_df):
    """Load all 8 context maps from a train split dataframe into tensors."""
    train_df['aa_pos'] = pd.to_numeric(train_df['aa_pos'])
    protein_len = int(train_df['aa_pos'].max())

    maps = {}
    for name in CONTEXTS:
        ma = csv_to_maskedarray(train_df, f'{name}_score', protein_len)
        tensor, mask = maskedarray_to_torch_tensor(ma)
        seq = get_proteinseq(name)
        tensor = place_ones_wt_aa_pos(tensor, seq)
        mask = place_trues_wt_aa_pos(mask, seq)
        maps[name] = (tensor, mask)
    return maps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SimpleAE on no-double-missing splits.")
    parser.add_argument("--base-dir", default="../data_splits_no_double_missing",
                        help="Base directory containing tgt_* subfolders.")
    parser.add_argument("--num-splits", type=int, default=50)
    parser.add_argument("--missing-rate", type=int, default=10)
    parser.add_argument("--target", required=True,
                        help="Target context (e.g., av12, wt25).")
    args = parser.parse_args()

    target = args.target
    if target not in CONTEXTS:
        raise ValueError(f"Unknown target '{target}'. Valid: {CONTEXTS}")

    missing_rate = args.missing_rate
    # Read splits from tgt_{target}/ subdirectory
    tgt_base_dir = Path(args.base_dir) / f"tgt_{target}"
    out_sub_tpl = Path(f"single_AE3_testfrac{missing_rate}")

    source_contexts = [c for c in CONTEXTS if c != target]

    for split_number in range(1, args.num_splits + 1):
        try:
            train_df, mask_df, meta = load_pair(
                base_dir=str(tgt_base_dir), s=split_number, rate=missing_rate)
        except Exception as e:
            raise Exception(
                f"Train/Mask splits not found for split={split_number}, "
                f"rate={missing_rate} in {tgt_base_dir}"
            ) from e

        maps = build_maps(train_df)

        out_dir = tgt_base_dir / out_sub_tpl / f"split_{split_number}"
        out_dir.mkdir(parents=True, exist_ok=True)

        for src_name in source_contexts:
            print(f"Split {split_number}: SimpleAE {src_name} -> {target}")

            src_tensor, src_train_mask = maps[src_name]
            tgt_tensor, tgt_train_mask = maps[target]

            model, _norms, tgt_pred_n, tgt_pred = train_cross_map_with_simpleae(
                src=src_tensor, tgt=tgt_tensor,
                src_train_mask=src_train_mask, tgt_train_mask=tgt_train_mask,
                latent_dim=12, epochs=1000, batch_size=128,
                lr=1e-4, weight_decay=0.0, verbose=False,
                impute_mask=tgt_train_mask, seed=42
            )

            mthfr_proteinseq_src = get_proteinseq(src_name)
            mthfr_proteinseq_tgt = get_proteinseq(target)

            tgt_df = tensor_to_hgvs_dataframe(
                tgt_pred, mthfr_proteinseq_tgt, amino_acid_dict_num_to_3)
            tgt_df.rename(columns={'score': f'{target}_imputed'}, inplace=True)

            train_mask_target_df = tensor_to_hgvs_dataframe(
                tgt_train_mask, mthfr_proteinseq_tgt, amino_acid_dict_num_to_3)
            train_mask_target_df.rename(
                columns={'score': 'seen_in_training_tgt'}, inplace=True)

            train_mask_source_df = tensor_to_hgvs_dataframe(
                src_train_mask, mthfr_proteinseq_src, amino_acid_dict_num_to_3)
            train_mask_source_df.rename(
                columns={'score': 'seen_in_training_src'}, inplace=True)

            merged = (
                train_df[['hgvs_pro', f'{src_name}_score']]
                .merge(train_mask_source_df[['hgvs_pro', 'seen_in_training_src']],
                       on='hgvs_pro', how='left')
                .merge(tgt_df[['hgvs_pro', f'{target}_imputed']],
                       on='hgvs_pro', how='left')
                .merge(train_mask_target_df[['hgvs_pro', 'seen_in_training_tgt']],
                       on='hgvs_pro', how='left')
            )
            merged['seen_in_training_src'] = merged['seen_in_training_src'].astype('boolean').fillna(False)
            merged['seen_in_training_tgt'] = merged['seen_in_training_tgt'].astype('boolean').fillna(False)
            merged['tgt_src_relation'] = f'{src_name}->{target}'
            merged['model'] = 'SimpleAE'

            out_path = out_dir / f'{src_name}_to_{target}_singleAE.csv'
            merged.to_csv(out_path, index=False)
