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

def tokenize(smiles, tokens=None):
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
    Reads SMILES from file. File must contain one SMILES string per line
    with \n token in the end of the line.

    Args:
        filename (str): path to the file
        unique (bool): return only unique SMILES

    Returns:
        smiles (list): list of SMILES strings from specified file.
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




def standardize_smiles(smiles, min_heavy_atoms=10, max_heavy_atoms=50,
                       remove_long_side_chains=False, neutralise_charges=True):
    mol = Chem.MolFromSmiles(smiles)
    if mol and neutralise_charges:
        mol, _ = _neutraliseCharges(mol)
    if mol:
        rdmolops.Cleanup(mol)
        rdmolops.SanitizeMol(mol)
        mol = rdmolops.RemoveHs(mol, implicitOnly=False, updateExplicitCount=False, sanitize=True)
    if mol and valid_size(mol, min_heavy_atoms, max_heavy_atoms, remove_long_side_chains):
        return Chem.MolToSmiles(mol, isomericSmiles=False)
    return None


def standardize_smiles_list(smiles_list):
    """Reads a SMILES list and returns a list of RDKIT SMILES"""
    smiles_list = Parallel(n_jobs=-1, verbose=0)(delayed(standardize_smiles)(line) for line in smiles_list)
    smiles_list = [smiles for smiles in set(smiles_list) if smiles is not None]
    logging.debug("{} unique SMILES retrieved".format(len(smiles_list)))
    return smiles_list


def canonical_smiles(smiles):
    """
    Takes a SMILES string and returns its canonical SMILES.

    Parameters
    ----------
    smiles:str
         SMILES strings to convert into canonical format

    Returns
    -------
    new_smiles: str
         canonical SMILES and NaNs if SMILES string is invalid or
        unsanitized (when sanitize is True)
    """
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=False)
    except:
        return None




def randomSmiles(mol):
    mol.SetProp("_canonicalRankingNumbers", "True")
    idxs = list(range(0, mol.GetNumAtoms()))
    random.shuffle(idxs)
    for i, v in enumerate(idxs):
        mol.GetAtomWithIdx(i).SetProp("_canonicalRankingNumber", str(v))
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def smile_augmentation(smile, augmentation, max_len):
    mol = Chem.MolFromSmiles(smile)
    s = set()
    for _ in range(10000):
        smiles = randomSmiles(mol)
        if len(smiles) <= max_len:
            s.add(smiles)
            if len(s) == augmentation:
                break
    return list(s)


def save_smiles_to_file(filename, smiles, unique=True):
    """
    Takes path to file and list of SMILES strings and writes SMILES to the specified file.

        Args:
            filename (str): path to the file
            smiles (list): list of SMILES strings
            unique (bool): parameter specifying whether to write only unique copies or not.

        Output:
            success (bool): defines whether operation was successfully completed or not.
       """
    if unique:
        smiles = list(set(smiles))
    else:
        smiles = list(smiles)
    f = open(filename, 'w')
    for mol in smiles:
        f.writelines([mol, '\n'])
    f.close()
    return f.closed


def read_smiles_from_file(filename, unique=True, add_start_end_tokens=False):

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


def fp_array_from_smiles_list(smiles, radius=2, nbits=2048):
    mols = []
    fps = []
    for smile in smiles:
        try:
            mol = Chem.MolFromSmiles(smile)
            mols.append(mol)
        except:
            pass

    for mol in mols:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol=mol, radius=radius, nBits=nbits)
        fp = fp2arr(fp)
        fps.append(fp)

    return fps


def fingerprint(smiles, radius=2, nbits=2048):
    """
    Generates fingerprint for SMILES
    If smiles is invalid, returns None
    Returns fingerprint bits
    Parameters:
        smiles: SMILES string
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol=mol, radius=radius, nBits=nbits)
    return fingerprint


def scaffold(mol):
    """
    Extracts a scafold from a molecule in a form of a canonic SMILES
    """
    try:
        scaffold = Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol(mol)
    except (ValueError, RuntimeError):
        return None
    scaffold_smiles = Chem.MolToSmiles(scaffold)
    if scaffold_smiles == '':
        return None
    return scaffold_smiles


def scaffolds(smiles_list):
    mol_list = [Chem.MolFromSmiles(smile) for smile in smiles_list]
    mol_list = [mol for mol in mol_list if mol is not None]

    scaffold_list = [scaffold(mol) for mol in mol_list]
    scaffolds = Counter(scaffold_list)
    if None in scaffolds:
        scaffolds.pop(None)
    return scaffolds


def fragment(mol):
    """
    fragment mol using BRICS and return smiles list
    """
    fgs = Chem.AllChem.FragmentOnBRICSBonds(mol)
    fgs_smi = Chem.MolToSmiles(fgs).split(".")
    return fgs_smi


def fragments(smiles_list):
    """
    fragment list of smiles using BRICS and return smiles list
    """
    mol_list = [Chem.MolFromSmiles(smile) for smile in smiles_list]
    mol_list = [mol for mol in mol_list if mol is not None]

    fragments = Counter()
    for mol in mol_list:
        frags = fragment(mol)
        fragments.update(frags)
    return fragments


def get_structures(smiles_list):
    fps = []
    frags = []
    scaffs = []
    for smile in smiles_list:
        mol = Chem.MolFromSmiles(smile)
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
def valid_score(smiles):
    """
    score a smiles , if  it is valid, score = 1 ; else score = 0

    Parameters
    ----------
        smiles: str
            SMILES strings

    Returns
    -------
        score: int 0 or 1
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    else:
        return 1







