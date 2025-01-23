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
    Reads peptides from file. File must contain one peptides string per line
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


def canonical_peptides(peptides):
    """
    Takes a peptides string and returns its canonical peptides.

    Parameters
    ----------
    peptides:str
         peptides strings to convert into canonical format

    Returns
    -------
    new_peptides: str
         canonical peptides and NaNs if peptides string is invalid or
        unsanitized (when sanitize is True)
    """
    try:
        return Chem.MolTopeptides(Chem.MolFrompeptides(peptides), isomericpeptides=False)
    except:
        return None




def randompeptides(mol):
    mol.SetProp("_canonicalRankingNumbers", "True")
    idxs = list(range(0, mol.GetNumAtoms()))
    random.shuffle(idxs)
    for i, v in enumerate(idxs):
        mol.GetAtomWithIdx(i).SetProp("_canonicalRankingNumber", str(v))
    return Chem.MolTopeptides(mol, isomericpeptides=False)


def smile_augmentation(smile, augmentation, max_len):
    mol = Chem.MolFrompeptides(smile)
    s = set()
    for _ in range(10000):
        peptides = randompeptides(mol)
        if len(peptides) <= max_len:
            s.add(peptides)
            if len(s) == augmentation:
                break
    return list(s)


def save_peptides_to_file(filename, peptides, unique=True):
    """
    Takes path to file and list of peptides strings and writes peptides to the specified file.

        Args:
            filename (str): path to the file
            peptides (list): list of peptides strings
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


################ For experiment ################
def fp2arr(fp):
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def fp_array_from_peptides_list(peptides, radius=2, nbits=2048):
    mols = []
    fps = []
    for smile in peptides:
        try:
            mol = Chem.MolFrompeptides(smile)
            mols.append(mol)
        except:
            pass

    for mol in mols:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol=mol, radius=radius, nBits=nbits)
        fp = fp2arr(fp)
        fps.append(fp)

    return fps


def fingerprint(peptides, radius=2, nbits=2048):
    """
    Generates fingerprint for peptides
    If peptides is invalid, returns None
    Returns fingerprint bits
    Parameters:
        peptides: peptides string
    """
    mol = Chem.MolFrompeptides(peptides)
    if mol is None:
        return None
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol=mol, radius=radius, nBits=nbits)
    return fingerprint


def scaffold(mol):
    """
    Extracts a scafold from a molecule in a form of a canonic peptides
    """
    try:
        scaffold = Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol(mol)
    except (ValueError, RuntimeError):
        return None
    scaffold_peptides = Chem.MolTopeptides(scaffold)
    if scaffold_peptides == '':
        return None
    return scaffold_peptides


def scaffolds(peptides_list):
    mol_list = [Chem.MolFrompeptides(smile) for smile in peptides_list]
    mol_list = [mol for mol in mol_list if mol is not None]

    scaffold_list = [scaffold(mol) for mol in mol_list]
    scaffolds = Counter(scaffold_list)
    if None in scaffolds:
        scaffolds.pop(None)
    return scaffolds


def fragment(mol):
    """
    fragment mol using BRICS and return peptides list
    """
    fgs = Chem.AllChem.FragmentOnBRICSBonds(mol)
    fgs_smi = Chem.MolTopeptides(fgs).split(".")
    return fgs_smi


def fragments(peptides_list):
    """
    fragment list of peptides using BRICS and return peptides list
    """
    mol_list = [Chem.MolFrompeptides(smile) for smile in peptides_list]
    mol_list = [mol for mol in mol_list if mol is not None]

    fragments = Counter()
    for mol in mol_list:
        frags = fragment(mol)
        fragments.update(frags)
    return fragments


def get_structures(peptides_list):
    fps = []
    frags = []
    scaffs = []
    for smile in peptides_list:
        mol = Chem.MolFrompeptides(smile)
        fps.append(fingerprint(smile))
        frags.append(fragment(mol))
        scaffs.append(scaffold(mol))
    return fps, frags, scaffs


def get_TanimotoSimilarity(sources_fps, target_fps, option="max"):
    maxs = []
    means = []
    for s_fp in sources_fps:
        maximum = 0
        total = 0
        for t_fp in target_fps:
            similarity = DataStructs.FingerprintSimilarity(s_fp, t_fp)
            if similarity > maximum:
                maximum = similarity
            total = total + similarity
        maxs.append(maximum)
        means.append(total / len(target_fps))
    if option == 'max':
        return maxs
    elif option == 'mean':
        return means
    else:
        return None


################ For train ################
def valid_score(peptides):
    """
    score a peptides , if  it is valid, score = 1 ; else score = 0

    Parameters
    ----------
        peptides: str
            peptides strings

    Returns
    -------
        score: int 0 or 1
    """
    mol = Chem.MolFrompeptides(peptides)
    if mol is None:
        return 0
    else:
        return 1







