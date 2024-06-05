import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

compounds_filepath = "./data/compounds.csv"
pairs_filepath = "./data/pairs.csv"


def _load_compounds():
    compounds = pd.read_csv(compounds_filepath)
    return compounds


def _load_pairs():
    pairs = np.genfromtxt(pairs_filepath, delimiter=",")
    return pairs


def _extract_descriptors():
    compounds = _load_compounds()
    descriptors = compounds.loc[:, "monomer_mw":]
    scaler = StandardScaler()
    norm_descriptors = scaler.fit_transform(descriptors)
    return norm_descriptors


def _extract_tril_pairs():
    pairs = _load_pairs()
    pairs_tril_indices = np.tril_indices_from(pairs, k=-1)
    pairs_tril = pairs[pairs_tril_indices]
    return pairs_tril, pairs_tril_indices


def _train_val_split(train_frac: float, seed: int = 42):
    np.random.seed(seed)
    norm_descriptors = _extract_descriptors()
    pairs_tril, pairs_tril_indices = _extract_tril_pairs()

    num_pairs = pairs_tril.size
    # Training set
    num_training_examples = int(train_frac * num_pairs)

    training_indices_flat = np.random.choice(
        num_pairs, num_training_examples, replace=False
    )

    training_indices_rows = pairs_tril_indices[0][training_indices_flat]
    training_indices_cols = pairs_tril_indices[1][training_indices_flat]

    training_indices = tuple(zip(training_indices_rows, training_indices_cols))

    # Validation set
    val_indices_flat = np.setdiff1d(np.arange(num_pairs), training_indices_flat)
    val_indices_rows = pairs_tril_indices[0][val_indices_flat]
    val_indices_cols = pairs_tril_indices[1][val_indices_flat]

    val_indices = tuple(zip(val_indices_rows, val_indices_cols))


class MisciblityData:
    def __init__(self):
        """
        Load pairwise miscibility data
        """
