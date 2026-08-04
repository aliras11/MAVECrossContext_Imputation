
import numpy as np
from preprocess import amino_acid_dict_1_to_num, amino_acid_dict_3_to_num, amino_acid_dict_3_to_1, amino_acid_dict_1_to_3
import pandas as pd


def masktensor_to_dfcol(mask_tensor, protein_seq, aa_dict_num_to_3):
    """
    Convert a mask tensor to a DataFrame column with 1 for observed, 0 for missing.
    Returns:
        pd.Series with 1 for observed positions, 0 for missing.
    """
    records = []
    n_aa, n_pos = mask_tensor.shape
    for aa_idx in range(n_aa):
        for pos_idx in range(n_pos):
            wt_aa = amino_acid_dict_1_to_3.get(protein_seq[pos_idx], None)
            mut_aa_3 = aa_dict_num_to_3.get(aa_idx + 1, None)
            if wt_aa is not None and mut_aa_3 is not None:
                    if wt_aa == mut_aa_3: 
                        mut_aa_3 = mut_aa_3 + '_wt' 
            hgvs = f"p.{wt_aa}{pos_idx+1}{mut_aa_3}"
            observed = int(mask_tensor[aa_idx, pos_idx].item())
            records.append({'hgvs_pro': hgvs, 'observed': observed})
    return pd.DataFrame(records)

def reconstruct_hgvs_from_tensor(scores_tensor, protein_seq, aa_dict_num_to_3):
    """
    Reconstruct HGVS protein variant annotations from a scores tensor and mask.
    Only returns annotations for observed (non-masked) positions.
    Args:
        scores_tensor: torch.Tensor, shape (n_aa, n_pos)
        protein_seq: str, wildtype protein sequence (1-letter codes)
        aa_dict_num_to_3: dict, maps integer to 3-letter amino acid code
    Returns:
        List of HGVS strings, e.g. ['p.Ala1Val', ...]
    """
    hgvs_list = []
    n_aa, n_pos = scores_tensor.shape
    for aa_idx in range(n_aa):
        for pos_idx in range(n_pos):
            wt_aa = amino_acid_dict_1_to_3.get(protein_seq[pos_idx], None)
            mut_aa_3 = aa_dict_num_to_3.get(aa_idx + 1, None)
            if mut_aa_3 is not None:
                hgvs = f"p.{wt_aa}{pos_idx+1}{mut_aa_3}"
                hgvs_list.append(hgvs)
    return hgvs_list

def tensor_to_hgvs_dataframe(scores_tensor, protein_seq, aa_dict_num_to_3):
    """
    Generate a DataFrame with columns: hgvs_pro, score, not_imputed.
    Args:
        scores_tensor: torch.Tensor, shape (n_aa, n_pos)
        mask_tensor: torch.BoolTensor, same shape as scores_tensor
        protein_seq: str, wildtype protein sequence (1-letter codes)
        aa_dict_num_to_3: dict, maps integer to 3-letter amino acid code
    Returns:
        pd.DataFrame with columns ['hgvs_pro', 'score', 'not_imputed']
    """
    records = []
    n_aa, n_pos = scores_tensor.shape
    for aa_idx in range(n_aa):
        for pos_idx in range(n_pos):
            wt_aa = amino_acid_dict_1_to_3.get(protein_seq[pos_idx], None)
            mut_aa_3 = aa_dict_num_to_3.get(aa_idx + 1, None)
            if wt_aa is not None and mut_aa_3 is not None:
                if wt_aa == mut_aa_3: 
                    mut_aa_3 = mut_aa_3 + '_wt' 
                hgvs = f"p.{wt_aa}{pos_idx+1}{mut_aa_3}"
                score = scores_tensor[aa_idx, pos_idx].item()
                records.append({'hgvs_pro': hgvs, 'score': score})
    return pd.DataFrame(records)



def create_final_dataframe(score_tensor1, score_tensor2, imputed_tensor1, imputed_tensor2,
                          wt_mask1, wt_mask2, train_mask1, train_mask2, 
                          val_mask1, val_mask2, protein_seq, amino_acid_dict_num_to_3,
                          tensor1_name='score1', tensor2_name='score2'):
    """
    Create a comprehensive DataFrame from tensors and masks for dual autoencoder analysis.
    
    Args:
        score_tensor1: First score tensor
        score_tensor2: Second score tensor
        imputed_tensor1: Imputed values for first tensor
        imputed_tensor2: Imputed values for second tensor
        wt_mask1: Original mask for first tensor
        wt_mask2: Original mask for second tensor
        train_mask1: Training mask for first tensor
        train_mask2: Training mask for second tensor
        val_mask1: Validation mask for first tensor
        val_mask2: Validation mask for second tensor
        protein_seq: Protein sequence string
        amino_acid_dict_num_to_3: Dictionary mapping numbers to 3-letter amino acid codes
        tensor1_name: Name prefix for first tensor columns (default 'wt12')
        tensor2_name: Name prefix for second tensor columns (default 'wt200')
    
    Returns:
        pandas.DataFrame: Comprehensive dataframe with all scores and mask information
    """
    
    # Convert masks to dataframes
    train_mask1_df = masktensor_to_dfcol(train_mask1, protein_seq, amino_acid_dict_num_to_3)
    train_mask1_df.rename(columns={'observed': f'train_{tensor1_name}_observed'}, inplace=True)
    
    train_mask2_df = masktensor_to_dfcol(train_mask2, protein_seq, amino_acid_dict_num_to_3)
    train_mask2_df.rename(columns={'observed': f'train_{tensor2_name}_observed'}, inplace=True)
    
    val_mask1_df = masktensor_to_dfcol(val_mask1, protein_seq, amino_acid_dict_num_to_3)
    val_mask1_df.rename(columns={'observed': f'val_{tensor1_name}_observed'}, inplace=True)
    
    val_mask2_df = masktensor_to_dfcol(val_mask2, protein_seq, amino_acid_dict_num_to_3)
    val_mask2_df.rename(columns={'observed': f'val_{tensor2_name}_observed'}, inplace=True)
    
    wt_mask1_df = masktensor_to_dfcol(wt_mask1, protein_seq, amino_acid_dict_num_to_3)
    wt_mask1_df.rename(columns={'observed': f'{tensor1_name}_observed'}, inplace=True)
    
    wt_mask2_df = masktensor_to_dfcol(wt_mask2, protein_seq, amino_acid_dict_num_to_3)
    wt_mask2_df.rename(columns={'observed': f'{tensor2_name}_observed'}, inplace=True)
    
    # Convert score tensors to dataframes
    wt_scores1_df = tensor_to_hgvs_dataframe(score_tensor1, protein_seq, amino_acid_dict_num_to_3)
    wt_scores1_df.rename(columns={'score': f'{tensor1_name}_score'}, inplace=True)
    
    wt_scores2_df = tensor_to_hgvs_dataframe(score_tensor2, protein_seq, amino_acid_dict_num_to_3)
    wt_scores2_df.rename(columns={'score': f'{tensor2_name}_score'}, inplace=True)
    
    imputed1_df = tensor_to_hgvs_dataframe(imputed_tensor1, protein_seq, amino_acid_dict_num_to_3)
    imputed1_df.rename(columns={'score': f'imputed_{tensor1_name}_score'}, inplace=True)
    
    imputed2_df = tensor_to_hgvs_dataframe(imputed_tensor2, protein_seq, amino_acid_dict_num_to_3)
    imputed2_df.rename(columns={'score': f'imputed_{tensor2_name}_score'}, inplace=True)
    
    # Merge all dataframes
    final_df = wt_scores1_df.merge(wt_scores2_df, on='hgvs_pro', how='outer') \
        .merge(imputed1_df, on='hgvs_pro', how='outer') \
        .merge(imputed2_df, on='hgvs_pro', how='outer') \
        .merge(train_mask1_df, on='hgvs_pro', how='outer') \
        .merge(train_mask2_df, on='hgvs_pro', how='outer') \
        .merge(val_mask1_df, on='hgvs_pro', how='outer') \
        .merge(val_mask2_df, on='hgvs_pro', how='outer') \
        .merge(wt_mask1_df, on='hgvs_pro', how='outer') \
        .merge(wt_mask2_df, on='hgvs_pro', how='outer')
    
    return final_df

def calculate_validation_losses(final_df, tensor1_name='wt12', tensor2_name='wt200'):
    """
    Calculate validation losses for dual autoencoder results.
    
    Args:
        final_df: pandas DataFrame with autoencoder results and validation masks
        tensor1_name: Name of the first tensor (reconstruction target)
        tensor2_name: Name of the second tensor (prediction target)
    
    Returns:
        dict: Dictionary containing all calculated losses
    """
    
    # Define column names based on tensor names
    val_col1 = f'val_{tensor1_name}_observed'
    val_col2 = f'val_{tensor2_name}_observed'
    train_col1 = f'train_{tensor1_name}_observed'
    train_col2 = f'train_{tensor2_name}_observed'
    score_col1 = f'{tensor1_name}_score'
    score_col2 = f'{tensor2_name}_score'
    imputed_col1 = f'imputed_{tensor1_name}_score'
    imputed_col2 = f'imputed_{tensor2_name}_score'
    
    results = {}
    
    # Double missing (present in both validation sets)
    doublemissing = final_df[
        (~final_df['hgvs_pro'].str.endswith('wt') & final_df[val_col1].astype(bool)) & (final_df[val_col2].astype(bool))
    ]
    
    if len(doublemissing) > 0:
        # Reconstruction loss for double missing
        double_missing_subset = doublemissing[doublemissing[score_col1].notnull() & doublemissing[imputed_col1].notnull()]
        if len(double_missing_subset) > 0:
            imputed_scores_combined = pd.concat([double_missing_subset[imputed_col1], double_missing_subset[imputed_col2]], ignore_index=True)
            actual_scores_combined = pd.concat([double_missing_subset[score_col1], double_missing_subset[score_col2]], ignore_index=True)
            results['double_missing_total_rmse'] = np.sqrt(np.mean((actual_scores_combined - imputed_scores_combined)**2))
        else:
            results['double_missing_total_rmse'] = np.nan

    double_missing_calced_on_tgt = final_df[
        (~final_df['hgvs_pro'].str.endswith('wt') & final_df[val_col1].astype(bool)) & (final_df[val_col2].astype(bool))
    ]
    if len(double_missing_calced_on_tgt) > 0:
        # Reconstruction loss for double missing
        double_missing_subset = double_missing_calced_on_tgt[double_missing_calced_on_tgt[score_col2].notnull() & double_missing_calced_on_tgt[imputed_col2].notnull()]
        if len(double_missing_subset) > 0:
            imputed_scores_tgt= double_missing_subset[imputed_col2]
            actual_scores_tgt = double_missing_subset[score_col2]
            results['double_missing_tgt'] = np.sqrt(np.mean((imputed_scores_tgt - actual_scores_tgt)**2))
        else:
            results['double_missing_tgt'] = np.nan

    double_missing_calced_on_src = final_df[
        (~final_df['hgvs_pro'].str.endswith('wt') & final_df[val_col1].astype(bool)) & (final_df[val_col2].astype(bool))
    ]
    if len(double_missing_calced_on_src) > 0:
        # Reconstruction loss for double missing
        double_missing_subset = double_missing_calced_on_src[double_missing_calced_on_src[score_col1].notnull() & double_missing_calced_on_src[imputed_col1].notnull()]
        if len(double_missing_subset) > 0:
            imputed_scores_src= double_missing_subset[imputed_col1]
            actual_scores_src = double_missing_subset[score_col1]
            results['double_missing_src'] = np.sqrt(np.mean((imputed_scores_src - actual_scores_src)**2))
        else:
            results['double_missing_src'] = np.nan

    singlemissing_pred = final_df[
        ~final_df['hgvs_pro'].str.endswith('wt') & (~final_df[val_col1].astype(bool) & final_df[val_col2].astype(bool)) #if in validation, means it was not in training
    ]

    if len(singlemissing_pred) > 0:
        pred_single = singlemissing_pred[
            singlemissing_pred[score_col2].notnull() & singlemissing_pred[imputed_col2].notnull()
        ]
        if len(pred_single) > 0:
            results['regression_loss_equivalent'] = np.sqrt(np.mean(
                (pred_single[score_col2] - pred_single[imputed_col2])**2
            ))
        else:
            results['regression_loss_equivalent'] = np.nan
    else:
        results['regression_loss_equivalent'] = np.nan

   
    allmissing_loss = final_df[~final_df['hgvs_pro'].str.endswith('wt')&(final_df[val_col1].astype(bool) | (final_df[val_col2].astype(bool)))] #all points in validation
    
    if len(allmissing_loss) > 0:
        recon_all = allmissing_loss[
            allmissing_loss[score_col1].notnull() & allmissing_loss[imputed_col1].notnull()
        ]
        if len(recon_all) > 0:
            imputed_scores_combined = pd.concat([recon_all[imputed_col1], recon_all[imputed_col2]], ignore_index=True)
            actual_scores_combined = pd.concat([recon_all[score_col1], recon_all[score_col2]], ignore_index=True)
            results['allmissing_rmse_concat'] = np.sqrt(np.mean(
                (actual_scores_combined - imputed_scores_combined)**2
            ))
        else:
            results['allmissing_rmse'] = np.nan
    else:
        results['allmissing_rmse'] = np.nan
    #all missing loss on src only
    if len(allmissing_loss) > 0:
        recon_all = allmissing_loss[
            allmissing_loss[score_col1].notnull() & allmissing_loss[imputed_col1].notnull()
        ]
        if len(recon_all) > 0:
            imputed_scores_src = recon_all[imputed_col1]
            actual_scores_src = recon_all[score_col1]
            results['allmissing_rmse_src'] = np.sqrt(np.mean(
                (actual_scores_combined - imputed_scores_src)**2
            ))
        else:
            results['allmissing_rmse_src'] = np.nan
    else:
        results['allmissing_rmse_src'] = np.nan

    #all missing loss on tgt only
    if len(allmissing_loss) > 0:
        recon_all = allmissing_loss[
            allmissing_loss[score_col2].notnull() & allmissing_loss[imputed_col2].notnull()
        ]
        if len(recon_all) > 0:
            imputed_scores_tgt = recon_all[imputed_col2]
            actual_scores_tgt = recon_all[score_col2]
            results['allmissing_rmse_tgt'] = np.sqrt(np.mean(
                (actual_scores_tgt - imputed_scores_tgt)**2
            ))
        else:
            results['allmissing_rmse_tgt'] = np.nan
    else:
        results['allmissing_rmse_tgt'] = np.nan

    # Add counts for reference
    results['doublemissing_count'] = len(doublemissing)
    results['doublemissing_calced_on_tgt_count'] = len(double_missing_calced_on_tgt)
    results['doublemissing_calced_on_src_count'] = len(double_missing_calced_on_src)
    results['regression_loss_equivalent_count'] = len(singlemissing_pred)
    results['allmissing_count'] = len(allmissing_loss)
    results['pair_label'] = f"{tensor1_name}_to_{tensor2_name}"

    return results

