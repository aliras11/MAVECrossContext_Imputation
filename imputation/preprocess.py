import pandas as pd
import glob
import re 
import glob
import numpy as np
import torch


amino_acid_dict_3_to_1 = {'Ala': 'A',
 'Cys': 'C',
 'Asp': 'D',
 'Glu': 'E',
 'Phe': 'F',
 'Gly': 'G',
 'His': 'H',
 'Ile': 'I',
 'Lys': 'K',
 'Leu': 'L',
 'Met': 'M',
 'Asn': 'N',
 'Pro': 'P',
 'Gln': 'Q',
 'Arg': 'R',
 'Ser': 'S',
 'Thr': 'T',
 'Val': 'V',
 'Trp': 'W',
 'Tyr': 'Y'}

amino_acid_dict_3_to_num = {'Ala': 1,
 'Cys': 19,
 'Asp': 12,
 'Glu': 13,
 'Phe': 6,
 'Gly': 18,
 'His': 10,
 'Ile': 4,
 'Lys': 11,
 'Leu': 3,
 'Met': 5,
 'Asn': 16,
 'Pro': 20,
 'Gln': 17,
 'Arg': 9,
 'Ser': 14,
 'Thr': 15,
 'Val': 2,
 'Trp': 8,
 'Tyr': 7,
  '=':21,
  'Ter':22}

amino_acid_dict_1_to_num ={'A': 1,
 'C': 19,
 'D': 12,
 'E': 13,
 'F': 6,
 'G': 18,
 'H': 10,
 'I': 4,
 'K': 11,
 'L': 3,
 'M': 5,
 'N': 16,
 'P': 20,
 'Q': 17,
 'R': 9,
 'S': 14,
 'T': 15,
 'V': 2,
 'W': 8,
 'Y': 7,
  '=':21,
  'Ter':22}

amino_acid_dict_num_to_3 = {1: 'Ala',
 19: 'Cys',
 12: 'Asp',
 13: 'Glu',
 6: 'Phe',
 18: 'Gly',
 10: 'His',
 4: 'Ile',
 11: 'Lys',
 3: 'Leu',
 5: 'Met',
 16: 'Asn',
 20: 'Pro',
 17: 'Gln',
 9: 'Arg',
 14: 'Ser',
 15: 'Thr',
 2: 'Val',
 8: 'Trp',
 7: 'Tyr',
 21: '=',
 22: 'Ter'}

amino_acid_dict_1_to_3 = {
    'A': 'Ala',
    'C': 'Cys',
    'D': 'Asp',
    'E': 'Glu',
    'F': 'Phe',
    'G': 'Gly',
    'H': 'His',
    'I': 'Ile',
    'K': 'Lys',
    'L': 'Leu',
    'M': 'Met',
    'N': 'Asn',
    'P': 'Pro',
    'Q': 'Gln',
    'R': 'Arg',
    'S': 'Ser',
    'T': 'Thr',
    'V': 'Val',
    'W': 'Trp',
    'Y': 'Tyr'
}



mthfr_proteinseq = "MVNEARGNSSLNPCLEGSASSGSESSKDSSRCSTPGLDPERHERLREKMRRRLESGDKWFSLEFFPPRTAEGAVNLISRFDRMAAGGPLYIDVTWHPAGDPGSDKETSSMMIASTAVNYC\
GLETILHMTCCRQRLEEITGHLHKAKQLGLKNIMALRGDPIGDQWEEEEGGFNYAVDLVK\
HIRSEFGDYFDICVAGYPKGHPEAGSFEADLKHLKEKVSAGADFIITQLFFEADTFFRFV\
KACTDMGITCPIVPGIFPIQGYHSLRQLVKLSKLEVPQEIKDVIEPIKDNDAAIRNYGIE\
LAVSLCQELLASGLVPGLHFYTLNREMATTEVLKRLGMWTEDPRRPLPWALSAHPKRREE\
DVRPIFWASRPKSYIYRTQEWDEFPNGRWGNSSSPAFGELKDYYLFYLKSKSPKEELLKM\
WGEELTSEESVFEVFVLYLSGEPNRNGHKVTCLPWNDEPLAAETSLLKEELLRVNRQGIL\
TINSQPNINGKPSSDPIVGWGPSGGYVFQKAYLEFFTSRETAEALLQVLKKYELRVNYHL\
VNVKGENITNAPELQPNAVTWGIFPGREIIQPTVVDPVSFMFWKDEAFALWIERWGKLYE\
EESPSRTIIQYIHDNYFLVNLVDNDFPLDNCLWQVVEDTLELLNRPTQNARETEAP"

mthfr_proteinseq_wt = "MVNEARGNSSLNPCLEGSASSGSESSKDSSRCSTPGLDPERHERLREKMRRRLESGDKWFSLEFFPPRTAEGAVNLISRFDRMAAGGPLYIDVTWHPAGDPGSDKETSSMMIASTAVNYC\
GLETILHMTCCRQRLEEITGHLHKAKQLGLKNIMALRGDPIGDQWEEEEGGFNYAVDLVK\
HIRSEFGDYFDICVAGYPKGHPEAGSFEADLKHLKEKVSAGADFIITQLFFEADTFFRFV\
KACTDMGITCPIVPGIFPIQGYHSLRQLVKLSKLEVPQEIKDVIEPIKDNDAAIRNYGIE\
LAVSLCQELLASGLVPGLHFYTLNREMATTEVLKRLGMWTEDPRRPLPWALSAHPKRREE\
DVRPIFWASRPKSYIYRTQEWDEFPNGRWGNSSSPAFGELKDYYLFYLKSKSPKEELLKM\
WGEELTSEESVFEVFVLYLSGEPNRNGHKVTCLPWNDEPLAAETSLLKEELLRVNRQGIL\
TINSQPNINGKPSSDPIVGWGPSGGYVFQKAYLEFFTSRETAEALLQVLKKYELRVNYHL\
VNVKGENITNAPELQPNAVTWGIFPGREIIQPTVVDPVSFMFWKDEAFALWIERWGKLYE\
EESPSRTIIQYIHDNYFLVNLVDNDFPLDNCLWQVVEDTLELLNRPTQNARETEAP"


mthfr_proteinseq_alt = mthfr_proteinseq_wt[:221] + 'V' + mthfr_proteinseq_wt[222:]


def protein_seq_to_num(protein_seq):
    """
    Convert a protein sequence (1-letter codes) to a list of integers.
    Args:
        protein_seq: str, protein sequence in 1-letter code format
    Returns:
        list of integers representing the amino acids, 0 indexed
    """
    return [amino_acid_dict_1_to_num.get(aa, 0)-1 for aa in protein_seq]



def hgvs_pro_aminos_wt(p_string):
  '''take in an hgvs protein variant string and return the relevant WT aa'''
  aa_list = re.split(r'\d+',p_string)
  return aa_list[0][2:]

def hgvs_pro_aminos_alt(p_string):
  '''take in an hgvs protein variant string and return the relevant mutated aa'''
  aa_list = re.split(r'\d+',p_string)
  if aa_list[1] == '=': #replace the equal sign with the wt amino acid 
    aa_list = re.split(r'\d+',p_string)
    return aa_list[0][2:]
  return aa_list[1]

def hgvs_pro_aminos_pos(p_string):
  '''take in an hgvs protein variant string and return the relevant aa position in protein seq'''
  aa_pos = re.search(r'\d+',p_string)
  return p_string[aa_pos.start():aa_pos.end()]

def hgvs_mut_aa(p_string):
  c = re.search(r'([a-zA-Z]{3})(\d{1,})([a-zA-Z]{3}|=)', p_string)
  return amino_acid_dict_3_to_num[c.group(3)]

def hgvs_wt_aa(p_string):
  c = re.search(r'([a-zA-Z]{3})(\d{1,})([a-zA-Z]{3}|=)', p_string)
  return amino_acid_dict_3_to_num[c.group(1)]

def hgvs_num_aa(p_string):
  c = re.search(r'([a-zA-Z]{3})(\d{1,})([a-zA-Z]{3}|=)', p_string)
  return c.group(2)


def csv_to_maskedarray(df,score_name,protein_length):
  '''take in a pandas dataframe and place scores in a masked array'''
  if score_name.startswith('wt'):
     df = df[~df['hgvs_pro'].str.contains(r'p\.Val222', na=False)]
  elif score_name.startswith('av'):
      df = df[~df['hgvs_pro'].str.contains(r'p\.Ala222', na=False)] #exclude the Ala222 mutation if the score is in background variant context

  scores_to_putinarray = df[[f'{score_name}','aa_pos','alt_aminoacid','hgvs_pro']].to_numpy()
  score_array = np.ones((22,protein_length))*99999 #  this is hard coded for each map for convenience
  for score in scores_to_putinarray:
    col = int(score[1])-1
    row = amino_acid_dict_3_to_num['=']-1 if str(score[3]).endswith('=') else int(score[2])-1
    if np.isnan(score[0]):
        score_array[row,col] = 99999
    else:   
        score_array[row,col] = score[0]
  score_array_masked = np.ma.masked_values(score_array,99999)
  return score_array_masked

def csv_to_mask_overlay(df,protein_length):
  '''take in a pandas dataframe and return a mask array with 0 at the missing aa positions'''

  scores_to_putinarray = df[['score','pos_aminoacid','aaalt_num']].to_numpy()
  score_array = np.zeros((22,protein_length+1)) # this is hard coded for each map for convenience
  for score in scores_to_putinarray:
    col = int(score[1]-1)
    row = int(score[2]-1)
    score_array[row,col] = 1

  return score_array

def maskedarray_to_torch_tensor(masked_arr):
    """
    Convert a numpy masked array to a torch tensor, filling masked values with the mean of the unmasked values.

    Args:
        masked_arr (np.ma.MaskedArray): Input masked array.

    Returns:
        torch.Tensor: Tensor suitable for autoencoder input.
    """
    fill_val = masked_arr.mean()
    filled = masked_arr.filled(fill_val)
    tensor = torch.tensor(filled, dtype=torch.float32)
    mask = (~masked_arr.mask).astype(bool)  # 1 for observed, 0 for masked
    mask_tensor = torch.tensor(mask, dtype=torch.bool)
    return tensor, mask_tensor


def place_ones_wt_aa_pos(scores_tensor, protein_seq):
    """place ones in the corresponding positions of the wild type amino acids in the scores tensor."""
    n_aa, n_pos = scores_tensor.shape
    scores_tensor = scores_tensor.clone()
    for pos_idx in range(n_pos):
        wt_aa = amino_acid_dict_1_to_num.get(protein_seq[pos_idx],0) - 1
        if wt_aa >= 0 and wt_aa < n_aa:
            scores_tensor[wt_aa, pos_idx] = 1
    return scores_tensor

def place_trues_wt_aa_pos(mask_tensor, protein_seq):
    """place True in the corresponding positions of the wild type amino acids in the mask tensor
    use only with torch mask tensors as the 1 indicates that the postion is observed."""
    n_aa, n_pos = mask_tensor.shape
    mask_tensor = mask_tensor.clone()
    for pos_idx in range(n_pos):
        wt_aa = amino_acid_dict_1_to_num.get(protein_seq[pos_idx],0) - 1
        if wt_aa >= 0 and wt_aa < n_aa:
            mask_tensor[wt_aa, pos_idx] = True
    return mask_tensor

def mean_fill_tensor(tensor, mask):
    """Fill the unknown positions in the tensor with the mean of the known values, used during training
    to accurately simulate how a map with actually missing values would be handled."""
    filled_tensor = tensor.clone()
    mean_value = tensor[mask].mean()
    filled_tensor[~mask] = mean_value
    return filled_tensor
