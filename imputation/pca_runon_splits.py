"""
PCA-based within-map imputer for MTHFR variant effect scores.
Tests how many principal components are needed to reconstruct held-out entries,
giving a direct measure of intrinsic dimensionality.

Each position (656 total) is a sample, each amino acid (22) is a feature.
Outputs one CSV per split (like colmean imputer) with all 8 score columns
imputed via PCA reconstruction.

Usage:
    python pca_imputer.py --base-dir ../data_splits --num-splits 5 --missing-rate 40 --n-components 1
    python pca_imputer.py --base-dir ../data_splits --all-rates --n-components 2
"""
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')

from preprocess import (
    csv_to_maskedarray,
    maskedarray_to_torch_tensor,
    place_ones_wt_aa_pos,
    place_trues_wt_aa_pos,
    mthfr_proteinseq_wt,
    mthfr_proteinseq_alt,
    amino_acid_dict_num_to_3,
)
from performance_measurement import tensor_to_hgvs_dataframe
from split_generator import load_pair

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONTEXTS = ['av12', 'av25', 'av100', 'av200', 'wt12', 'wt25', 'wt100', 'wt200']


def pca_impute_within_map(data_tensor, train_mask, n_components, max_iter=20, tol=1e-4):
    """
    Iterative PCA imputation for a single context (within-map).

    Args:
        data_tensor: (22, 656) numpy array with mean-filled missing values
        train_mask: (22, 656) boolean, True = observed in training split
        n_components: number of PCA components to use
        max_iter: max EM iterations
        tol: convergence tolerance (max change in imputed values)

    Returns:
        reconstructed: (22, 656) numpy array with PCA reconstruction
        pca: fitted PCA object
    """
    X = data_tensor.T.copy()  # (656, 22)
    mask = train_mask.T.copy()  # (656, 22)

    col_means = np.nanmean(np.where(mask, X, np.nan), axis=0)
    col_means = np.nan_to_num(col_means, nan=0.0)
    X_filled = X.copy()
    X_filled[~mask] = np.broadcast_to(col_means, X.shape)[~mask]

    n_comp = min(n_components, min(X.shape))

    for iteration in range(max_iter):
        pca = PCA(n_components=n_comp)
        scores = pca.fit_transform(X_filled)
        reconstructed = pca.inverse_transform(scores)

        X_new = X_filled.copy()
        X_new[~mask] = reconstructed[~mask]

        change = np.max(np.abs(X_new[~mask] - X_filled[~mask])) if (~mask).any() else 0.0
        X_filled = X_new
        if change < tol:
            break

    pca = PCA(n_components=n_comp)
    scores = pca.fit_transform(X_filled)
    reconstructed = pca.inverse_transform(scores)

    return reconstructed.T, pca


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PCA within-map imputer')
    parser.add_argument('--base-dir', default='../data_splits')
    parser.add_argument('--num-splits', type=int, default=5)
    parser.add_argument('--missing-rate', type=int, default=40)
    parser.add_argument('--all-rates', action='store_true')
    parser.add_argument('--n-components', type=int, nargs='+', default=[1, 2, 4, 10, 20, 22],
                        help='Number of PCA components to test')
    args = parser.parse_args()

    rates = [10, 20, 40, 60, 80, 90] if args.all_rates else [args.missing_rate]

    for n_comp in args.n_components:
        for rate in rates:
            for split in range(1, args.num_splits + 1):
                out_dir = Path(args.base_dir) / f'pca_k{n_comp}_testfrac{rate}' / f'split_{split}'
                out_path = out_dir / f'pca_k{n_comp}_split{split}.csv'
                if out_path.exists():
                    print(f'  Skipping k={n_comp} rate={rate} split={split} (exists)')
                    continue

                print(f'k={n_comp}, Rate {rate}%, Split {split}')
                train_df, mask_df, meta = load_pair(base_dir=args.base_dir, s=split, rate=rate)
                train_df['aa_pos'] = pd.to_numeric(train_df['aa_pos'])
                protein_len = int(train_df['aa_pos'].max())

                # Start with the train_df and fill each score column via PCA
                result_df = train_df.copy()

                for context in CONTEXTS:
                    score_col = f'{context}_score'
                    protseq = mthfr_proteinseq_alt if context.startswith('av') else mthfr_proteinseq_wt

                    masked_arr = csv_to_maskedarray(train_df, score_col, protein_len)
                    tensor, mask = maskedarray_to_torch_tensor(masked_arr)
                    tensor = place_ones_wt_aa_pos(tensor, protseq)
                    mask = place_trues_wt_aa_pos(mask, protseq)

                    reconstructed, pca_obj = pca_impute_within_map(
                        tensor.numpy(), mask.numpy(), n_comp)

                    # Convert reconstructed tensor back to hgvs dataframe
                    import torch
                    recon_tensor = torch.tensor(reconstructed, dtype=torch.float32)
                    recon_df = tensor_to_hgvs_dataframe(recon_tensor, protseq, amino_acid_dict_num_to_3)
                    recon_df.rename(columns={'score': f'{context}_pca_imputed'}, inplace=True)

                    # Merge imputed values into result
                    result_df = result_df.merge(
                        recon_df[['hgvs_pro', f'{context}_pca_imputed']],
                        on='hgvs_pro', how='left')

                    # Fill original NaN score values with PCA imputed values
                    missing_mask = result_df[score_col].isna()
                    result_df.loc[missing_mask, score_col] = result_df.loc[missing_mask, f'{context}_pca_imputed']
                    result_df.drop(columns=[f'{context}_pca_imputed'], inplace=True)

                out_dir.mkdir(parents=True, exist_ok=True)
                result_df.to_csv(out_path, index=False)
                print(f'  Saved {out_path}')

    print('Done.')
