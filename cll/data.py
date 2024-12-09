import sys
import numpy as np
import torch
import torch.utils.data as data
from typing import Tuple, Optional, Any


class LoaderWithComplementaryLabels:
    def __init__(
            self,
            train_dataset: data.Dataset,
            test_dataset: data.Dataset,
            batch_size: int,
            seed: int = 0
    ):
        """
        Initialize the general dataset handler

        Args:
            train_dataset (Dataset): PyTorch training dataset
            test_dataset (Dataset): PyTorch test dataset
            batch_size (int): Size of batches for data loaders
            seed (int): Random seed for reproducibility
        """
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.set_seeds(seed)
        self.prepare_data()

    @staticmethod
    def set_seeds(seed: int) -> None:
        """Set random seeds for reproducibility"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def generate_compl_labels(self, labels: torch.Tensor) -> np.ndarray:
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

    def prepare_train_loaders(self) -> Tuple[data.DataLoader, data.DataLoader, np.ndarray]:
        """
        Prepare ordinary and complementary train loaders

        Returns:
            tuple: (ordinary_train_loader, complementary_train_loader, class_prior)
        """
        for i, (data_batch, labels) in enumerate(self.full_train_loader):
            complementary_labels = self.generate_compl_labels(labels)
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

            return ordinary_train_loader, complementary_train_loader, ccp

    def get_loaders(self) -> Tuple[data.DataLoader, data.DataLoader, data.DataLoader, np.ndarray]:
        """
        Get all data loaders and class prior probability in one call

        Returns:
            tuple: (ordinary_train_loader, complementary_train_loader, test_loader, class_prior)
        """
        ordinary_train_loader, complementary_train_loader, cpp = self.prepare_train_loaders()
        return ordinary_train_loader, complementary_train_loader, self.test_loader, cpp
