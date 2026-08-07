import itertools
import pandas as pd
import argparse
import re
from pathlib import Path
from preprocess import (
    hgvs_pro_aminos_pos,
    hgvs_mut_aa,
    hgvs_wt_aa,
    hgvs_num_aa,
    hgvs_pro_aminos_wt,
    hgvs_pro_aminos_alt,
    csv_to_maskedarray,
    maskedarray_to_torch_tensor,
    place_ones_wt_aa_pos,
    place_trues_wt_aa_pos,
    mthfr_proteinseq_wt,
    mthfr_proteinseq_alt,
    amino_acid_dict_num_to_3,
)
from training_regimes import (
    train_cross_map_with_simpleae,
    train_dual_column_imputer,
    impute_all_dual_columns,
    zscore_by_position,
    un_zscore  
)

from performance_measurement import (
    tensor_to_hgvs_dataframe)


from split_generator import(
    load_pair
)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run DualAE imputation across pre-generated splits.")
    parser.add_argument("--base-dir", default="../data_splits", help="Base directory containing split CSVs (train/mask).")
    parser.add_argument(
        "--out-subdirectory",
        default="Dual_AE3_testfrac{missing_rate}/split_{split_number}",
        help="Output subdirectory template. You can use {split_number} and {missing_rate} placeholders.",
    )
    parser.add_argument("--num-splits", type=int, default=50, help="Number of splits to process (1..N).")
    parser.add_argument("--missing-rate", type=int, default=10, help="Missing rate used to select split files.")
    args = parser.parse_args()

    base_dir = args.base_dir
    num_splits = args.num_splits
    missing_rate = args.missing_rate

    for split_number in range(1, num_splits + 1):
        print(f"Processing split {split_number} (rate={missing_rate})")
        train_df, mask_df, meta = load_pair(base_dir=base_dir, s=split_number, rate=missing_rate)

        train_df['aa_pos'] = pd.to_numeric(train_df['aa_pos'])
        protein_len = int(train_df['aa_pos'].max())

        av25map = csv_to_maskedarray(train_df, 'av25_score', protein_len)
        av12map = csv_to_maskedarray(train_df, 'av12_score', protein_len)
        av100map = csv_to_maskedarray(train_df, 'av100_score', protein_len)
        av200map = csv_to_maskedarray(train_df, 'av200_score', protein_len)

        wt25map = csv_to_maskedarray(train_df, 'wt25_score', protein_len)
        wt12map = csv_to_maskedarray(train_df, 'wt12_score', protein_len)
        wt100map = csv_to_maskedarray(train_df, 'wt100_score', protein_len)
        wt200map = csv_to_maskedarray(train_df, 'wt200_score', protein_len)

        # Convert to tensors and apply WT placements (per repeat)
        av12_tensor, av12_mask = maskedarray_to_torch_tensor(av12map)
        av25_tensor, av25_mask = maskedarray_to_torch_tensor(av25map)
        av100_tensor, av100_mask = maskedarray_to_torch_tensor(av100map)
        av200_tensor, av200_mask = maskedarray_to_torch_tensor(av200map)

        av12_tensor = place_ones_wt_aa_pos(av12_tensor, mthfr_proteinseq_alt)
        av25_tensor = place_ones_wt_aa_pos(av25_tensor, mthfr_proteinseq_alt)
        av100_tensor = place_ones_wt_aa_pos(av100_tensor, mthfr_proteinseq_alt)
        av200_tensor = place_ones_wt_aa_pos(av200_tensor, mthfr_proteinseq_alt)

        av12_mask = place_trues_wt_aa_pos(av12_mask, mthfr_proteinseq_alt)
        av25_mask = place_trues_wt_aa_pos(av25_mask, mthfr_proteinseq_alt)
        av100_mask = place_trues_wt_aa_pos(av100_mask, mthfr_proteinseq_alt)
        av200_mask = place_trues_wt_aa_pos(av200_mask, mthfr_proteinseq_alt)

        wt12_tensor, wt12_mask = maskedarray_to_torch_tensor(wt12map)
        wt25_tensor, wt25_mask = maskedarray_to_torch_tensor(wt25map)
        wt100_tensor, wt100_mask = maskedarray_to_torch_tensor(wt100map)
        wt200_tensor, wt200_mask = maskedarray_to_torch_tensor(wt200map)

        wt12_tensor = place_ones_wt_aa_pos(wt12_tensor, mthfr_proteinseq_wt)
        wt25_tensor = place_ones_wt_aa_pos(wt25_tensor, mthfr_proteinseq_wt)
        wt100_tensor = place_ones_wt_aa_pos(wt100_tensor, mthfr_proteinseq_wt)
        wt200_tensor = place_ones_wt_aa_pos(wt200_tensor, mthfr_proteinseq_wt)

        wt12_mask = place_trues_wt_aa_pos(wt12_mask, mthfr_proteinseq_wt)
        wt25_mask = place_trues_wt_aa_pos(wt25_mask, mthfr_proteinseq_wt)
        wt100_mask = place_trues_wt_aa_pos(wt100_mask, mthfr_proteinseq_wt)
        wt200_mask = place_trues_wt_aa_pos(wt200_mask, mthfr_proteinseq_wt)

        maps = {
            'av12':  (av12_tensor,  av12_mask),
            'av25':  (av25_tensor,  av25_mask),
            'av100': (av100_tensor, av100_mask),
            'av200': (av200_tensor, av200_mask),
            'wt12':  (wt12_tensor,  wt12_mask),
            'wt25':  (wt25_tensor,  wt25_mask),
            'wt100': (wt100_tensor, wt100_mask),
            'wt200': (wt200_tensor, wt200_mask),
        }

        out_subdirectory = Path(
            args.out_subdirectory.format(
                split_number=split_number,
                missing_rate=missing_rate,
            )
        )
        out_dir = Path(base_dir) / out_subdirectory
        out_dir.mkdir(parents=True, exist_ok=True)

        for src_name, tgt_name in itertools.product(maps.keys(), maps.keys()):

            if src_name == tgt_name:
                continue
            print(f"DualAE {src_name} -> {tgt_name}")
            src_tensor, src_train_mask = maps[src_name]
            tgt_tensor, tgt_train_mask = maps[tgt_name]

            if bool(re.match(r'(?i)^av', src_name)): #use relevant protein sequence based on source map type
                mthfr_proteinseq_src = mthfr_proteinseq_alt
            else:
                mthfr_proteinseq_src = mthfr_proteinseq_wt

            if bool(re.match(r'(?i)^av', tgt_name)):
                mthfr_proteinseq_tgt = mthfr_proteinseq_alt
            else:
                mthfr_proteinseq_tgt = mthfr_proteinseq_wt

            out_path = out_dir / f"{src_name}_to_{tgt_name}_DualAE.csv"

            encoder_dim = 12
            epochs = 1000
            batch_size = 128
            seed = 42

            result = train_dual_column_imputer(
                src_tensor, src_train_mask,
                tgt_tensor, tgt_train_mask,
                encoder_dim=encoder_dim,
                epochs=epochs,
                batch_size=batch_size,
                seed=seed,
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
                imputed_src_n, imputed_tgt_n = impute_all_dual_columns(model, src_n, tgt_n, src_train_mask, tgt_train_mask)
                imputed_src = un_zscore(imputed_src_n, s_mu, s_sd)
                imputed_tgt = un_zscore(imputed_tgt_n, t_mu, t_sd)

            tgt_df = tensor_to_hgvs_dataframe(imputed_tgt, mthfr_proteinseq_tgt, amino_acid_dict_num_to_3)
            tgt_df.rename(columns={'score': tgt_name+'_imputed'}, inplace=True)

            src_df = tensor_to_hgvs_dataframe(imputed_src, mthfr_proteinseq_src, amino_acid_dict_num_to_3)
            src_df.rename(columns={'score': f'{src_name}_imputed'}, inplace=True)

            train_mask_target_df = tensor_to_hgvs_dataframe(tgt_train_mask, mthfr_proteinseq_tgt, amino_acid_dict_num_to_3)
            train_mask_target_df.rename(columns={'score': 'seen_in_training_tgt'}, inplace=True)

            train_mask_source_df = tensor_to_hgvs_dataframe(src_train_mask, mthfr_proteinseq_src, amino_acid_dict_num_to_3)
            train_mask_source_df.rename(columns={'score': 'seen_in_training_src'}, inplace=True)

            merged = (
                train_df[['hgvs_pro', f'{src_name}_score']]
                .merge(src_df[['hgvs_pro', f'{src_name}_imputed']], on='hgvs_pro', how='left')
                .merge(train_mask_source_df[['hgvs_pro', 'seen_in_training_src']], on='hgvs_pro', how='left')
                .merge(train_df[['hgvs_pro', f'{tgt_name}_score']], on='hgvs_pro', how='left')
                .merge(tgt_df[['hgvs_pro', f'{tgt_name}_imputed']], on='hgvs_pro', how='left')
                .merge(train_mask_target_df[['hgvs_pro', 'seen_in_training_tgt']], on='hgvs_pro', how='left')
            )

            merged.rename(columns={f'{src_name}_score': f'{src_name}_score_trainset'}, inplace=True)
            merged.rename(columns={f'{tgt_name}_score': f'{tgt_name}_score_trainset'}, inplace=True)

            merged['seen_in_training_src'] = merged['seen_in_training_src'].astype('boolean').fillna(False)
            merged['seen_in_training_tgt'] = merged['seen_in_training_tgt'].astype('boolean').fillna(False)
            merged['tgt_src_relation'] = f'{src_name}->{tgt_name}'
            merged['model'] = 'DualAE'
            out_path = out_dir / f"{src_name}_to_{tgt_name}_DualAE.csv"
            merged.to_csv(out_path, index=False)
