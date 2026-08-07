import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Align import substitution_matrices

from split_generator import load_pair  # see: AE_imputation_maves/split_generator.py

CANONICAL_AAS = list("ACDEFGHIKLMNPQRSTVWY")
AA_3_TO_1 = {
    'Ala': 'A', 'Cys': 'C', 'Asp': 'D', 'Glu': 'E', 'Phe': 'F',
    'Gly': 'G', 'His': 'H', 'Ile': 'I', 'Lys': 'K', 'Leu': 'L',
    'Met': 'M', 'Asn': 'N', 'Pro': 'P', 'Gln': 'Q', 'Arg': 'R',
    'Ser': 'S', 'Thr': 'T', 'Val': 'V', 'Trp': 'W', 'Tyr': 'Y'
}
AA_1_TO_3 = {v: k for k, v in AA_3_TO_1.items()}


def load_blosum_iij(filepath):
    """
    Parse a BLOSUM .iij file (integer matrix in 1/2 bit units) into a
    Bio.Align.substitution_matrices.Array so it's a drop-in replacement
    for substitution_matrices.load().
    """
    filepath = Path(filepath)
    lines = filepath.read_text().splitlines()

    # Find the header line with amino acid letters
    header = None
    data_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#') or stripped == '':
            continue
        tokens = stripped.split()
        if all(len(t) == 1 and t.isalpha() for t in tokens) and len(tokens) >= 20:
            header = tokens
            data_lines = []
            continue
        if header is not None:
            data_lines.append(tokens)

    if header is None:
        raise ValueError(f"Could not find amino acid header in {filepath}")

    n = len(header)
    mat = substitution_matrices.Array(alphabet=''.join(header), dims=2)

    for i, row_tokens in enumerate(data_lines[:n]):
        for j, val in enumerate(row_tokens):
            mat[header[i], header[j]] = float(val)
            mat[header[j], header[i]] = float(val)  # symmetric

    return mat


def load_blosum_matrix(matrix_name, matrix_file=None):
    """
    Load a BLOSUM matrix. If matrix_file is provided (path to .iij),
    parse it directly. Otherwise fall back to BioPython built-ins.
    """
    if matrix_file is not None:
        return load_blosum_iij(matrix_file)
    return substitution_matrices.load(matrix_name)


def knn_blosum_impute(from_aa_3, to_aa_3, pos, blosum_mat, k=5, dataframe=None, column_name="wt12_score", pos_window=10, sim_mode="diff"):
    # pos_window controls the positional search range: for a missing value at
    # position P, only substitutions at positions [P - pos_window, P + pos_window]
    # are considered as neighbors. window=0 means only the same position,
    # window=10 means positions P-10 through P+10 (21 positions total).
    canonical_aas = CANONICAL_AAS
    score_se_cols = [col for col in dataframe.columns if col.endswith('score') or col.endswith('se')]
    non_score_cols = [col for col in dataframe.columns if col not in score_se_cols]

    if to_aa_3 == 'Ter':
        # Ter mutations can't use BLOSUM similarity, so we use a spatial approach:
        # Grab Ter scores from positions [pos-2, pos-1] (left) and [pos+1, pos+2] (right).
        # If both sides have data and their means are close (|diff| < 0.2), use
        # the mean of all available Ters in that ±2 window. If the means diverge
        # (e.g. domain boundary) or fewer than 2 total Ters are available, fall back to 0.
        ter_rows = dataframe[
            (dataframe['str_aa_mut'] == 'Ter')
        ].copy()
        ter_rows = ter_rows[non_score_cols + [column_name]].dropna(subset=[column_name])

        left = ter_rows[(ter_rows['aa_pos'] >= pos - 2) & (ter_rows['aa_pos'] < pos)]
        right = ter_rows[(ter_rows['aa_pos'] > pos) & (ter_rows['aa_pos'] <= pos + 2)]

        if len(left) + len(right) < 2:
            return (0, 'ter_fallback')

        left_mean = left[column_name].mean() if len(left) > 0 else None
        right_mean = right[column_name].mean() if len(right) > 0 else None

        if left_mean is not None and right_mean is not None:
            if abs(left_mean - right_mean) < 0.2:
                window_all = pd.concat([left, right])
                return (window_all[column_name].mean(), 'ter_impute')
            else:
                return (0, 'ter_fallback')
        else:
            # Only one side has data — use it if at least 2 points
            side = left if len(left) >= 2 else right if len(right) >= 2 else None
            if side is not None:
                return (side[column_name].mean(), 'ter_impute')
            return (0, 'ter_fallback')

    aa_pair = (AA_3_TO_1[from_aa_3], AA_3_TO_1[to_aa_3])

    # Find k most similar substitutions by BLOSUM score
    substitutions = []
    og_pair_score = blosum_mat[(aa_pair[0], aa_pair[1])]  # BLOSUM(wt, mut) — constant per call
    for i in canonical_aas:
        if sim_mode == "direct":
            # Rank by BLOSUM(mut, candidate) — highest = most similar to mutant AA
            score = blosum_mat[(aa_pair[1], i)]
            substitutions.append((score, i, (AA_1_TO_3[aa_pair[0]], AA_1_TO_3[i])))
        else:
            # "diff": rank by |BLOSUM(wt, candidate) - BLOSUM(wt, mut)| — lowest = closest
            candidate_pair_score = blosum_mat[(aa_pair[0], i)]
            diff = abs(candidate_pair_score - og_pair_score)
            substitutions.append((diff, i, (AA_1_TO_3[aa_pair[0]], AA_1_TO_3[i])))

    if sim_mode == "direct":
        substitutions.sort(key=lambda x: x[0], reverse=True)   # highest first
    else:
        substitutions.sort(key=lambda x: x[0], reverse=False)  # lowest first
    closest_k = substitutions[:k]
    top_k_similar = [sub[2][1] for sub in closest_k]

    # Filter rows: same wt AA, similar mut AA, nearby position
    filtered_rows = dataframe[
        (dataframe['str_aa_wt'] == from_aa_3) &
        (dataframe['str_aa_mut'].isin(top_k_similar))
    ].copy()

    # Keep only rows within position window
    filtered_rows = filtered_rows[
        (filtered_rows['aa_pos'] >= pos - pos_window) &
        (filtered_rows['aa_pos'] <= pos + pos_window)
    ]

    # Select relevant columns and drop NAs for the target column
    filtered_rows = filtered_rows[non_score_cols + [column_name]].dropna(subset=[column_name])

    # If insufficient data, fall back to position mean, then global mean
    if len(filtered_rows) < 2:
        col_pos_mean = dataframe[dataframe['aa_pos'] == pos][column_name].mean()
        if pd.isna(col_pos_mean):
            return (dataframe[column_name].mean(), 'global_mean')
        return (col_pos_mean, 'pos_mean')

    return (filtered_rows[column_name].mean(), 'knn')


def impute_all_columns_knn_blosum(dataframe, blosum_mat, k=5, pos_window=10, sim_mode="diff"):
    """
    Apply KNN-BLOSUM imputation to all score/se columns in the dataframe.

    Args:
        dataframe: DataFrame with mutation data
        blosum_mat: Pre-loaded BLOSUM substitution matrix
        k: number of similar substitutions to consider
        pos_window: position proximity window
        sim_mode: "diff" (original) or "direct" (rank by BLOSUM(mut, candidate))

    Returns:
        DataFrame with imputed columns added (suffix '_knn_blosum')
    """
    df = dataframe.copy()

    # Identify all score and se columns
    score_se_cols = [col for col in df.columns if col.endswith('score') or col.endswith('se')]

    print(f"Found {len(score_se_cols)} score/se columns to impute: {score_se_cols}")

    for col in score_se_cols:
        print(f"\nImputing column: {col}")
        imp_col = f"{col}_imputed"
        method_col = f"{col}_method"

        def _impute_row(row):
            if pd.notna(row[col]):
                return (row[col], 'observed')
            return knn_blosum_impute(
                from_aa_3=row['str_aa_wt'],
                to_aa_3=row['str_aa_mut'],
                pos=row['aa_pos'],
                blosum_mat=blosum_mat,
                k=k,
                dataframe=df,
                column_name=col,
                pos_window=pos_window,
                sim_mode=sim_mode
            )

        results = df.apply(_impute_row, axis=1)
        df[imp_col] = results.apply(lambda x: x[0])
        df[method_col] = results.apply(lambda x: x[1])

        counts = df[method_col].value_counts()
        print(f"  Methods: {counts.to_dict()}")

    return df



def main():
    parser = argparse.ArgumentParser(description="Run BLOSUM KNN imputation across pre-generated splits.")
    parser.add_argument(
        "--base-dir",
        default="../data_splits",
        help="Base directory containing split CSVs (train/mask).",
    )
    parser.add_argument(
        "--num-splits",
        type=int,
        default=50,
        help="Number of split indices to process (starting at 1).",
    )
    parser.add_argument(
        "--missing-rate",
        type=int,
        default=10,
        help="Missing rate used when selecting split files.",
    )
    parser.add_argument(
        "--matrix",
        default="BLOSUM100",
        help="BLOSUM matrix name (e.g., BLOSUM62, BLOSUM80, BLOSUM90, BLOSUM100).",
    )
    parser.add_argument(
        "--matrix-file",
        default=None,
        help="Path to a .iij BLOSUM matrix file. Overrides --matrix for matrices not in BioPython.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of similar substitutions to consider.",
    )
    parser.add_argument(
        "--pos-window",
        type=int,
        default=0,
        help="+/- amino-acid position window. 0 = same position only.",
    )
    parser.add_argument(
        "--sim_mode",
        choices=["diff", "direct"],
        default="diff",
        help="Similarity mode: 'diff' = |BLOSUM(wt,candidate) - BLOSUM(wt,mut)| (original), "
             "'direct' = BLOSUM(mut, candidate) highest first.",
    )
    parser.add_argument(
        "--out-subdir",
        default=None,
        help=(
            "Output folder name under base-dir. Default: "
            "blosum_knn_{rate}_{matrix}_k{k}_w{pos_window}; appends "
            "_direct when --sim_mode direct."
        ),
    )

    args = parser.parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    rate = int(args.missing_rate)

    # Load BLOSUM matrix once
    matrix_file = args.matrix_file
    if matrix_file is None and args.matrix == "BLOSUM100":
        # Auto-detect bundled blosum100.iij
        bundled = Path(__file__).resolve().parent / "blosum100.iij"
        if bundled.exists():
            matrix_file = str(bundled)
    blosum_mat = load_blosum_matrix(args.matrix, matrix_file=matrix_file)
    print(f"Loaded matrix: {args.matrix}" + (f" from {matrix_file}" if matrix_file else " (BioPython built-in)"))

    sim_tag = f"_{args.sim_mode}" if args.sim_mode != "diff" else ""
    out_name = args.out_subdir or f"blosum_knn_{rate}_{args.matrix}_k{args.k}_w{args.pos_window}{sim_tag}"
    out_root = base_dir / out_name
    out_root.mkdir(parents=True, exist_ok=True)

    for split_number in range(1, int(args.num_splits) + 1):
        try:
            train_df, _mask_df, _meta = load_pair(base_dir=str(base_dir), s=split_number, rate=rate)
        except Exception as e:
            raise RuntimeError(
                f"Relevant Train/Mask splits not found for split={split_number}, rate={rate} in {base_dir}"
            ) from e

        # ensure numeric positions (required by the position window filtering)
        train_df["aa_pos"] = pd.to_numeric(train_df["aa_pos"], errors="coerce")

        imputed = impute_all_columns_knn_blosum(
            train_df,
            blosum_mat=blosum_mat,
            k=args.k,
            pos_window=args.pos_window,
            sim_mode=args.sim_mode,
        )

        out_dir = out_root / f"split_{split_number}"
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"blosum_knn_imputed_split{split_number}_r{rate}.csv"
        imputed.to_csv(out_path, index=False)
        print(f"[saved] split={split_number} rate={rate} -> {out_path}")

if __name__ == "__main__":
    main()
