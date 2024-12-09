import sys
import numpy as np
import torch
import torch.utils.data as data
from typing import Tuple, Optional, Any, Callable

from numpy.testing import assert_array_almost_equal

set1 = np.array([
    [0, 0.75 / 3, 0.13 / 3, 0.12 / 3, 0.13 / 3, 0.12 / 3, 0.75 / 3, 0.12 / 3, 0.75 / 3, 0.13 / 3],
    [0.13 / 3, 0, 0.75 / 3, 0.13 / 3, 0.12 / 3, 0.13 / 3, 0.12 / 3, 0.75 / 3, 0.12 / 3, 0.75 / 3],
    [0.75 / 3, 0.13 / 3, 0, 0.75 / 3, 0.13 / 3, 0.12 / 3, 0.13 / 3, 0.12 / 3, 0.75 / 3, 0.12 / 3],
    [0.12 / 3, 0.75 / 3, 0.13 / 3, 0, 0.75 / 3, 0.13 / 3, 0.12 / 3, 0.13 / 3, 0.12 / 3, 0.75 / 3],
    [0.75 / 3, 0.12 / 3, 0.75 / 3, 0.13 / 3, 0, 0.75 / 3, 0.13 / 3, 0.12 / 3, 0.13 / 3, 0.12 / 3],
    [0.12 / 3, 0.75 / 3, 0.12 / 3, 0.75 / 3, 0.13 / 3, 0, 0.75 / 3, 0.13 / 3, 0.12 / 3, 0.13 / 3],
    [0.13 / 3, 0.12 / 3, 0.75 / 3, 0.12 / 3, 0.75 / 3, 0.13 / 3, 0, 0.75 / 3, 0.13 / 3, 0.12 / 3],
    [0.12 / 3, 0.13 / 3, 0.12 / 3, 0.75 / 3, 0.12 / 3, 0.75 / 3, 0.13 / 3, 0, 0.75 / 3, 0.13 / 3],
    [0.13 / 3, 0.12 / 3, 0.13 / 3, 0.12 / 3, 0.75 / 3, 0.12 / 3, 0.75 / 3, 0.13 / 3, 0, 0.75 / 3],
    [0.75 / 3, 0.13 / 3, 0.12 / 3, 0.13 / 3, 0.12 / 3, 0.75 / 3, 0.12 / 3, 0.75 / 3, 0.13 / 3, 0]])
set2 = np.array([
    [0, 0.66 / 3, 0.24 / 3, 0.1 / 3, 0.24 / 3, 0.1 / 3, 0.66 / 3, 0.1 / 3, 0.66 / 3, 0.24 / 3],
    [0.24 / 3, 0, 0.66 / 3, 0.24 / 3, 0.1 / 3, 0.24 / 3, 0.1 / 3, 0.66 / 3, 0.1 / 3, 0.66 / 3],
    [0.66 / 3, 0.24 / 3, 0, 0.66 / 3, 0.24 / 3, 0.1 / 3, 0.24 / 3, 0.1 / 3, 0.66 / 3, 0.1 / 3],
    [0.1 / 3, 0.66 / 3, 0.24 / 3, 0, 0.66 / 3, 0.24 / 3, 0.1 / 3, 0.24 / 3, 0.1 / 3, 0.66 / 3],
    [0.66 / 3, 0.1 / 3, 0.66 / 3, 0.24 / 3, 0, 0.66 / 3, 0.24 / 3, 0.1 / 3, 0.24 / 3, 0.1 / 3],
    [0.1 / 3, 0.66 / 3, 0.1 / 3, 0.66 / 3, 0.24 / 3, 0, 0.66 / 3, 0.24 / 3, 0.1 / 3, 0.24 / 3],
    [0.24 / 3, 0.1 / 3, 0.66 / 3, 0.1 / 3, 0.66 / 3, 0.24 / 3, 0, 0.66 / 3, 0.24 / 3, 0.1 / 3],
    [0.1 / 3, 0.24 / 3, 0.1 / 3, 0.66 / 3, 0.1 / 3, 0.66 / 3, 0.24 / 3, 0, 0.66 / 3, 0.24 / 3],
    [0.24 / 3, 0.1 / 3, 0.24 / 3, 0.1 / 3, 0.66 / 3, 0.1 / 3, 0.66 / 3, 0.24 / 3, 0, 0.66 / 3],
    [0.66 / 3, 0.24 / 3, 0.1 / 3, 0.24 / 3, 0.1 / 3, 0.66 / 3, 0.1 / 3, 0.66 / 3, 0.24 / 3, 0]])


def generate_complimentary_labels_random(labels: torch.Tensor) -> np.ndarray:
    """
    Generate complementary labels for given ordinary labels

    Args:
        labels: ordinary labels tensor

    Returns:
        numpy array: complementary labels
    """
    K = torch.max(labels) + 1
    candidates = np.arange(K)
    candidates = np.repeat(candidates.reshape(1, K), len(labels), 0)
    mask = np.ones((len(labels), K), dtype=bool)
    mask[range(len(labels)), labels.numpy()] = False
    candidates_ = candidates[mask].reshape(len(labels), K - 1)
    idx = np.random.randint(0, K - 1, len(labels))
    complementary_labels = candidates_[np.arange(len(labels)), np.array(idx)]
    return complementary_labels


def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    #    print (np.max(y), P.shape[0])
    y = y.numpy()
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]
    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


class LoaderWithComplementaryLabels:
    def __init__(
            self,
            train_dataset: data.Dataset,
            test_dataset: data.Dataset,
            batch_size: int,
            method: str = 'random',
            transition_matrix=None,
            seed: int = 0
    ):
        """
        Initialize the general dataset handler

        Args: train_dataset (Dataset): PyTorch training dataset test_dataset (Dataset): PyTorch test dataset
        batch_size (int): Size of batches for data loaders method (str): A method to generate complementary labels.
        By default, we assume that the labels are generated randomly seed (int): Random seed for reproducibility
        """
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.method = method
        self.transition_matrix = transition_matrix
        assert self.method in ['random', 'scarce_1', 'scarce_2'], "Method unknown"
        if self.method == 'random':
            assert transition_matrix is None, "Transition matrix not necessary for random generation"
            self.fn_generate_comp_labels = generate_complimentary_labels_random
        else:
            if self.method == 'scarce_1':
                self.transition_matrix = transition_matrix or set1
            elif self.method == 'scarce_2':
                self.transition_matrix = transition_matrix or set2
            self.fn_generate_comp_labels = lambda labels: multiclass_noisify(labels, self.transition_matrix, seed)
        self.set_seeds(seed)
        self.prepare_data()

    @staticmethod
    def set_seeds(seed: int) -> None:
        """Set random seeds for reproducibility"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @staticmethod
    def class_prior(complementary_labels: np.ndarray) -> np.ndarray:
        """
        Calculate class prior from complementary labels

        Args:
            complementary_labels: array of complementary labels

        Returns:
            numpy array: class prior probabilities
        """
        return np.bincount(complementary_labels) / len(complementary_labels)

    def prepare_data(self) -> None:
        """Prepare dataset and create data loaders"""
        self.train_loader = data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        self.test_loader = data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        self.full_train_loader = data.DataLoader(
            dataset=self.train_dataset,
            batch_size=len(self.train_dataset),
            shuffle=True
        )

        # Try to get number of classes from dataset
        self.num_classes = self._get_num_classes()

    def _get_num_classes(self) -> int:
        """
        Try different methods to get the number of classes from the dataset

        Returns:
            int: Number of classes in the dataset
        """
        # Method 1: Try to get classes attribute
        if hasattr(self.train_dataset, 'classes'):
            return len(self.train_dataset.classes)

        # Method 2: Try to get from the first batch of labels
        try:
            first_batch = next(iter(self.full_train_loader))
            labels = first_batch[1]
            return int(torch.max(labels).item()) + 1
        except:
            raise ValueError(
                "Could not determine number of classes. "
                "Please ensure your dataset's labels are zero-indexed integers."
            )

    def prepare_train_loaders(self) -> Tuple[data.DataLoader, data.DataLoader, np.ndarray, int]:
        """
        Prepare ordinary and complementary train loaders

        Returns:
            tuple: (ordinary_train_loader, complementary_train_loader, class_prior)
        """
        for i, (data_batch, labels) in enumerate(self.full_train_loader):
            complementary_labels = generate_complimentary_labels_random(labels)
            ccp = self.class_prior(complementary_labels)

            complementary_dataset = data.TensorDataset(
                data_batch,
                torch.from_numpy(complementary_labels).float()
            )

            ordinary_train_loader = data.DataLoader(
                dataset=self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True
            )

            complementary_train_loader = data.DataLoader(
                dataset=complementary_dataset,
                batch_size=self.batch_size,
                shuffle=True
            )
            dim = int(data_batch.reshape(-1).shape[0]/data.shape[0])

            return ordinary_train_loader, complementary_train_loader, ccp, dim

    def get_loaders(self) -> Tuple[data.DataLoader, data.DataLoader, data.DataLoader, np.ndarray, int]:
        """
        Get all data loaders and class prior probability in one call

        Returns:
            tuple: (ordinary_train_loader, complementary_train_loader, test_loader, class_prior)
        """
        ordinary_train_loader, complementary_train_loader, cpp, dim = self.prepare_train_loaders()
        return ordinary_train_loader, complementary_train_loader, self.test_loader, cpp, dim
