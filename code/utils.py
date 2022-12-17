import numpy as np

# constants

DNAalphabet = 'BATCG-'
AAalphabet = 'BCDSQKIPTFNGHLRWAVEYM-'

VHSE = {"A": [0.15, -1.11, -1.35, -0.92,  0.02, -0.91,  0.36, -0.48],
        "R": [-1.47,  1.45,  1.24,  1.27,  1.55,  1.47,  1.30,  0.83],
        "N": [-0.99,  0.00, -0.37,  0.69, -0.55,  0.85,  0.73, -0.80],
        "D": [-1.15,  0.67, -0.41, -0.01, -2.68,  1.31,  0.03,  0.56],
        "C": [0.18, -1.67, -0.46, -0.21,  0.00,  1.20, -1.61, -0.19],
        "Q": [-0.96,  0.12,  0.18,  0.16,  0.09,  0.42, -0.20, -0.41],
        "E": [-1.18,  0.40,  0.10,  0.36, -2.16, -0.17,  0.91,  0.02],
        "G": [-0.20, -1.53, -2.63,  2.28, -0.53, -1.18,  2.01, -1.34],
        "H": [-0.43, -0.25,  0.37,  0.19,  0.51,  1.28,  0.93,  0.65],
        "I": [1.27, -0.14,  0.30, -1.80,  0.30, -1.61, -0.16, -0.13],
        "L": [1.36,  0.07,  0.26, -0.80,  0.22, -1.37,  0.08, -0.62],
        "K": [-1.17,  0.70,  0.70,  0.80,  1.64,  0.67,  1.63,  0.13],
        "M": [1.01, -0.53,  0.43,  0.00,  0.23,  0.10, -0.86, -0.68],
        "F": [1.52,  0.61,  0.96, -0.16,  0.25,  0.28, -1.33, -0.20],
        "P": [0.22, -0.17, -0.50,  0.05, -0.01, -1.34, -0.19,  3.56],
        "S": [-0.67, -0.86, -1.07, -0.41, -0.32,  0.27, -0.64,  0.11],
        "T": [-0.34, -0.51, -0.55, -1.06, -0.06, -0.01, -0.79,  0.39],
        "W": [1.50,  2.06,  1.79,  0.75,  0.75, -0.13, -1.01, -0.85],
        "Y": [0.61,  1.60,  1.17,  0.73,  0.53,  0.25, -0.96, -0.52],
        "V": [0.76, -0.92, -0.17, -1.91,  0.22, -1.40, -0.24, -0.03],
        "B": [0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00],
        "-": [0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00]}

# functions


def chopping(data, lim=48):
    """Removes peptide sequences longer than a certain thereshold from a list of sequences

    Args:
        data (list): List of sequences as strings 
        lim (int, optional): Length thereshold. Defaults to 48.

    Returns:
        list: List of sequences shorter than limit
    """
    chopped = []
    for seq in data:
        if len(seq) <= lim:
            chopped.append(seq)

    return chopped


def padding(data, begin_token='', end_token='-', lim=48):
    """Pads all sequences in the list to a certain length with an end token

    Args:
        data (list): List of sequences as strings 
        begin_token (str, optional): Character to pad the beginning of each sequence string. Defaults to ''.
        end_token (str, optional): Character to pad the end of each sequence string to reach the length limit. Defaults to '-'.
        lim (int, optional): Length thereshold. Defaults to 48.

    Returns:
        list: List of padded sequences 
    """
    padded = []
    for seq in data:
        temp = begin_token + seq + end_token * (lim - len(seq))
        padded.append(temp)

    return padded


def onehot_encoding(data, alphabet=AAalphabet):
    """One-hot encoding of DNA or protein sequences

    Args:
        data (list): List of sequence strings to encode dim:(N, sequence length)
        alphabet (string, optional): The alphabet to use; either DNA or Amino Acid. Defaults to AAalphabet.

    Returns:
        list: List of encoded sequences dim:(N, sequence length, alphabet length)
    """
    aa2hot = {}
    for i, aa in enumerate(alphabet):
        v = [0 for j in alphabet]
        v[i] = 1
        aa2hot[aa] = v

    onehot_encoded = []
    for seq in data:
        temp = []
        for aa in seq:
            temp.append(aa2hot[aa])
        onehot_encoded.append(temp)
    return onehot_encoded


def onehot_decoding(data, alphabet=AAalphabet):
    """One-hot decoding of DNA or protein sequences

    Args:
        data (list):  List of one-hot encoded sequences dim:(N, sequence length, alphabet length)
        alphabet (_type_, optional): The alphabet to use; either DNA or Amino Acid. Defaults to AAalphabet.

    Returns:
        list: List of decoded sequences as strings. dim:(N, sequence length) 
    """
    onehot_decoded = []
    for array in data:
        temp = ''
        for i, seq in enumerate(array):
            temp += alphabet[seq.index(max(seq))]
        onehot_decoded.append(temp)

    return onehot_decoded


def vhse_encoding(data):
    """VHSE encoding of amino acid sequences

    Args:
        data (list): List of protein sequence strings 

    Returns:
        list: List of encoded sequences
    """
    vhse_encoded = []
    for seq in data:
        pep = []
        for aa in seq:
            pep.append(VHSE[aa])
        vhse_encoded.append(pep)
    return vhse_encoded


def prepare_VAE(data, csv=False):
    if csv == True:
        seq = pd.read_csv(data)
    else:
        seq = data
    seq = chopping(seq)
    seq = padding(seq)
    seq = onehot_encoding(seq)
    seq = np.asarray(seq)
    return seq


def prepare_CNN(data, csv=False):
    if csv == True:
        seq = pd.read_csv(data)
    else:
        seq = data
    seq = chopping(seq)
    seq = padding(seq)
    seq = onehot_encoding(seq)
    seq = np.asarray(seq)
    return seq
