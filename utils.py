import random
import numpy as np
import warnings
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import logging
from joblib import Parallel, delayed


rdBase.DisableLog('rdApp.error')


################ For data process ################

def tokenize(peptides, tokens=None):
    word_index_dict = {'b': '0', 'a': '1', 'r': '2', 'n': '3', 'd': '4', 'c': '5', 'q': '6', 'e': '7', 'g': '8',
                       'h': '9', 'i': '10', 'l': '11', 'k': '12', 'm': '13', 'f': '14', 'p': '15', 's': '16', 't': '17',
                       'w': '18', 'y': '19', 'v': '20', 'x': '21', 'z': '22', 'u': '23', 'j': '24', 'o': '25'}
    tokens = ['b', 'a', 'r', 'n', 'd', 'c', 'q', 'e', 'g', 'h', 'i', 'l', 'k', 'm', 'f', 'p', 's', 't', 'w', 'y', 'v',
              'x', 'z', 'u', 'j', 'o']

    token2idx = {'b': 0, 'a': 1, 'r': 2, 'n': 3, 'd': 4, 'c': 5, 'q': 6, 'e': 7, 'g': 8,
                 'h': 9, 'i': 10, 'l': 11, 'k': 12, 'm': 13, 'f': 14, 'p': 15, 's': 16, 't': 17,
                 'w': 18, 'y': 19, 'v': 20, 'x': 21, 'z': 22, 'u': 23, 'j': 24, 'o': 25}
    num_tokens = len(tokens)
    return tokens, token2idx, num_tokens

def read_peptides_from_file(filename, unique=True, add_start_end_tokens=False):
    """
    Reads peptides from file. File must contain one peptides sequences per line
    with \n token in the end of the line.

    Args:
        filename (str): path to the file
        unique (bool): return only unique peptides

    Returns:
        peptides (list): list of peptide sequences from specified file.
        success (bool): defines whether operation was successfully completed or not.

    If 'unique=True' this list contains only unique copies.
    """
    f = open(filename, 'r')
    peptide = []
    # for line in f:
    #     if add_start_end_tokens:
    #         molecules.append('<' + line[:-1] + '>')
    #     else:
    #         molecules.append(line[:-1])
    # if unique:
    #     molecules = list(set(molecules))
    # else:
    #     molecules = list(molecules)
    for line in f:
       peptide.append(line.split())
    f.close()
    return peptide, f.closed

class GANLoss(nn.Module):
    """Reward-Refined NLLLoss Function for adversial training of Gnerator"""

    def __init__(self):
        super(GANLoss, self).__init__()

    def forward(self, prob, reward):
        """
        Args:
            prob:  torch tensor
            reward :  torch tensor
        """
        loss = prob * reward
        loss = - torch.sum(loss)
        return loss


def get_reward(sample, dis1, dis2, gen_loader):
    if len(sample) == 2:
        return 0, 0, 0
    elif sample[1:].find('<') != -1:
        return 0, 0, 0
    else:
        x_temp = sample
        return dis1.classify(gen_loader.char_tensor(x_temp)), dis2.classify(
            gen_loader.char_tensor(x_temp)), 0  # 最后一个参数是valid_score(x_temp)


class NLLLoss(nn.Module):
    """ NLLLoss Function for  Gnerator"""

    def __init__(self):
        super(NLLLoss, self).__init__()

    def forward(self, prob):
        """
        Args:
            prob:  torch tensor
        """
        loss = - torch.sum(prob)
        return loss




def standardize_peptides(peptides, min_heavy_atoms=10, max_heavy_atoms=50,
                       remove_long_side_chains=False, neutralise_charges=True):
    mol = Chem.MolFrompeptides(peptides)
    if mol and neutralise_charges:
        mol, _ = _neutraliseCharges(mol)
    if mol:
        rdmolops.Cleanup(mol)
        rdmolops.SanitizeMol(mol)
        mol = rdmolops.RemoveHs(mol, implicitOnly=False, updateExplicitCount=False, sanitize=True)
    if mol and valid_size(mol, min_heavy_atoms, max_heavy_atoms, remove_long_side_chains):
        return Chem.MolTopeptides(mol, isomericpeptides=False)
    return None


def standardize_peptides_list(peptides_list):
    """Reads a peptides list and returns a list of RDKIT peptides"""
    peptides_list = Parallel(n_jobs=-1, verbose=0)(delayed(standardize_peptides)(line) for line in peptides_list)
    peptides_list = [peptides for peptides in set(peptides_list) if peptides is not None]
    logging.debug("{} unique peptides retrieved".format(len(peptides_list)))
    return peptides_list




def save_peptides_to_file(filename, peptides, unique=True):
    """
    Takes path to file and list of peptides sequences and writes peptides to the specified file.

        Args:
            filename (str): path to the file
            peptides (list): list of peptides sequences
            unique (bool): parameter specifying whether to write only unique copies or not.

        Output:
            success (bool): defines whether operation was successfully completed or not.
       """
    if unique:
        peptides = list(set(peptides))
    else:
        peptides = list(peptides)
    f = open(filename, 'w')
    for mol in peptides:
        f.writelines([mol, '\n'])
    f.close()
    return f.closed


def read_peptides_from_file(filename, unique=True, add_start_end_tokens=False):

    f = open(filename, 'r')
    peptide = []
    for line in f:
       peptide.append(line.split())
    f.close()
    return peptide, f.closed













