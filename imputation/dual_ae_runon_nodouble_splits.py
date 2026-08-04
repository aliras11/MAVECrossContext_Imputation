"""No-double-missing variant of dual_ae_runon_splits.py (DualAE).

Only the target column has injected missingness; source columns are fully observed.
Iterates 7 source contexts for the given target.

Usage:
    python dual_ae_runon_nodouble_splits.py --base-dir ../data_splits_no_double_missing \
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
from training_regimes import (
    train_dual_column_imputer,
    impute_all_dual_columns,
    zscore_by_position,
    un_zscore,
)
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
        description="Run DualAE on no-double-missing splits.")
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
    tgt_base_dir = Path(args.base_dir) / f"tgt_{target}"
    out_sub_tpl = f"Dual_AE3_testfrac{missing_rate}"

    source_contexts = [c for c in CONTEXTS if c != target]

    for split_number in range(1, args.num_splits + 1):
        print(f"Processing split {split_number} (rate={missing_rate}, target={target})")
        train_df, mask_df, meta = load_pair(
            base_dir=str(tgt_base_dir), s=split_number, rate=missing_rate)

        maps = build_maps(train_df)

        out_dir = tgt_base_dir / out_sub_tpl / f"split_{split_number}"
        out_dir.mkdir(parents=True, exist_ok=True)

        for src_name in source_contexts:
            print(f"  DualAE {src_name} -> {target}")

            src_tensor, src_train_mask = maps[src_name]
            tgt_tensor, tgt_train_mask = maps[target]

            mthfr_proteinseq_src = get_proteinseq(src_name)
            mthfr_proteinseq_tgt = get_proteinseq(target)

            result = train_dual_column_imputer(
                src_tensor, src_train_mask,
                tgt_tensor, tgt_train_mask,
                encoder_dim=12,
                epochs=1000,
                batch_size=128,
                seed=42,
                lr=1e-4,
                recon_weight=0.7,
            )

            # Support different return signatures
            if isinstance(result, tuple) and len(result) == 4:
                model, _norms, imputed_src, imputed_tgt = result
            else:
                model = result
                src_n, s_mu, s_sd = zscore_by_position(src_tensor, src_train_mask)
                tgt_n, t_mu, t_sd = zscore_by_position(tgt_tensor, tgt_train_mask)
                imputed_src_n, imputed_tgt_n = impute_all_dual_columns(
                    model, src_n, tgt_n, src_train_mask, tgt_train_mask)
                imputed_src = un_zscore(imputed_src_n, s_mu, s_sd)
                imputed_tgt = un_zscore(imputed_tgt_n, t_mu, t_sd)

            tgt_df = tensor_to_hgvs_dataframe(
                imputed_tgt, mthfr_proteinseq_tgt, amino_acid_dict_num_to_3)
            tgt_df.rename(columns={'score': f'{target}_imputed'}, inplace=True)

            src_df = tensor_to_hgvs_dataframe(
                imputed_src, mthfr_proteinseq_src, amino_acid_dict_num_to_3)
            src_df.rename(columns={'score': f'{src_name}_imputed'}, inplace=True)

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
                .merge(src_df[['hgvs_pro', f'{src_name}_imputed']],
                       on='hgvs_pro', how='left')
                .merge(train_mask_source_df[['hgvs_pro', 'seen_in_training_src']],
                       on='hgvs_pro', how='left')
                .merge(train_df[['hgvs_pro', f'{target}_score']],
                       on='hgvs_pro', how='left')
                .merge(tgt_df[['hgvs_pro', f'{target}_imputed']],
                       on='hgvs_pro', how='left')
                .merge(train_mask_target_df[['hgvs_pro', 'seen_in_training_tgt']],
                       on='hgvs_pro', how='left')
            )

            merged.rename(columns={
                f'{src_name}_score': f'{src_name}_score_trainset',
                f'{target}_score': f'{target}_score_trainset',
            }, inplace=True)

            merged['seen_in_training_src'] = merged['seen_in_training_src'].astype('boolean').fillna(False)
            merged['seen_in_training_tgt'] = merged['seen_in_training_tgt'].astype('boolean').fillna(False)
            merged['tgt_src_relation'] = f'{src_name}->{target}'
            merged['model'] = 'DualAE'

            out_path = out_dir / f"{src_name}_to_{target}_DualAE.csv"
            merged.to_csv(out_path, index=False)
