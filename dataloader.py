import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

compounds_filepath = "./data/compounds.csv"
pairs_filepath = "./data/pairs.csv"


def _extract_tril(matrix: np.ndarray):
    tril_indices = np.tril_indices_from(matrix, k=-1)
    tril = matrix[tril_indices]
    return tril, tril_indices


class MisciblityData:
    def __init__(self):
        """
        Load pairwise miscibility data
        """
        self.compounds = pd.read_csv(compounds_filepath)
        self.pairs = np.genfromtxt(pairs_filepath, delimiter=",")
        self.descriptors = self.compounds.loc[:, "monomer_mw":]
        scaler = StandardScaler()
        self.norm_descriptor = scaler.fit_transform(self.descriptors)
        self.num_compounds = self.compounds.shape[0]
        self.num_descriptors = self.compounds.shape[1] - 3

    def _train_val_split(self, train_frac: float = 0.7, seed: int = 42):
        np.random.seed(seed)
        pairs_tril, pairs_tril_indices = _extract_tril(self.pairs)
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

        return training_indices, val_indices

    def load_data(self, **kwargs):
        """
        Returns the generated train and validation data.
        Each data is in the form of NDarray.
        """
        training_indices, val_indices = self._train_val_split(**kwargs)
        num_descriptors = self.num_descriptors
        norm_descriptors = self.norm_descriptor
        pairs = self.pairs

        # Make training data
        X_train_shape = (len(training_indices), 2 * num_descriptors)
        X_train = np.zeros(X_train_shape)
        Y_train = np.zeros(len(training_indices))

        for i, pair in enumerate(training_indices):
            compound1 = pair[0]
            compound2 = pair[1]
            X_train[i, :num_descriptors] = norm_descriptors[compound1, :]
            X_train[i, num_descriptors:] = norm_descriptors[compound2, :]
            Y_train[i] = pairs[pair]

        # Make validation data
        X_val_shape = (len(val_indices), 2 * num_descriptors)
        X_val = np.zeros(X_val_shape)
        Y_val = np.zeros(len(val_indices))

        for i, pair in enumerate(val_indices):
            compound1 = pair[0]
            compound2 = pair[1]
            X_val[i, :num_descriptors] = norm_descriptors[compound1, :]
            X_val[i, num_descriptors:] = norm_descriptors[compound2, :]
            Y_val[i] = pairs[pair]

        return ((X_train, Y_train), (X_val, Y_val))
