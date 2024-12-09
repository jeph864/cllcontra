import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Callable, Tuple, Any
import numpy as np


def class_prior(complementary_labels):
    return np.bincount(complementary_labels) / len(complementary_labels)


def generate_compl_labels(labels):
    # args, labels: ordinary labels
    K = torch.max(labels) + 1
    candidates = np.arange(K)
    candidates = np.repeat(candidates.reshape(1, K), len(labels), 0)
    mask = np.ones((len(labels), K), dtype=bool)
    mask[range(len(labels)), labels.numpy()] = False
    candidates_ = candidates[mask].reshape(len(labels), K - 1)  # this is the candidates without true class
    idx = np.random.randint(0, K - 1, len(labels))
    complementary_labels = candidates_[np.arange(len(labels)), np.array(idx)]
    return complementary_labels


class DatasetWithComplementaryLabels:
    def __init__(
            self,
            train_dataset: Dataset,
            test_dataset: Dataset,
            generate_complementary_fn: Callable[[torch.Tensor], torch.Tensor] = None,
            batch_size: int = 64,
            normalize_data_fn: Callable[[torch.Tensor], torch.Tensor] = None,
    ):
        """
        Generic class to prepare a dataset with ordinary and complementary labels.
        Args:
            train_dataset (Dataset): Training dataset with ordinary labels.
            test_dataset (Dataset): Test dataset with ordinary labels.
            generate_complementary_fn (Callable): Function to generate complementary labels.
            batch_size (int): Batch size for DataLoaders.
            normalize_data_fn (Callable): Function to normalize input data (optional).
        """
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.generate_complementary_fn = generate_complementary_fn or generate_compl_labels
        self.batch_size = batch_size
        self.normalize_data_fn = normalize_data_fn

        # Full train loader to access the entire dataset for complementary label generation
        self.full_train_loader = DataLoader(
            dataset=self.train_dataset, batch_size=len(self.train_dataset), shuffle=True
        )

        # Generate complementary labels and class prior probabilities
        self.complementary_labels, self.ccp = self._generate_complementary_labels()

        # Prepare data loaders
        self.ordinary_train_loader = DataLoader(
            dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.complementary_train_loader = self._create_complementary_dataloader()
        self.test_loader = DataLoader(
            dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False
        )

    def _generate_complementary_labels(self):
        """
        Generate complementary labels for the training dataset.
        Returns:
            Tuple[torch.Tensor, np.ndarray]: Complementary labels and class prior probabilities.
        """
        for _, (data, labels) in enumerate(self.full_train_loader):
            complementary_labels = self.generate_complementary_fn(labels)
            ccp = class_prior(complementary_labels)
            return complementary_labels, ccp

    def _create_complementary_dataloader(self):
        """
        Create a DataLoader for the complementary dataset.
        Returns:
            DataLoader: Complementary training DataLoader.
        """
        data = self.train_dataset.data.float()
        if self.normalize_data_fn:
            data = self.normalize_data_fn(data)
        complementary_dataset = TensorDataset(data, torch.from_numpy(self.complementary_labels).float())
        return DataLoader(dataset=complementary_dataset, batch_size=self.batch_size, shuffle=True)

    def get_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader, Any]:
        """
        Get all prepared DataLoaders.
        Returns:
            tuple: (ordinary_train_loader, complementary_train_loader, test_loader, ccp)
        """
        return self.ordinary_train_loader, self.complementary_train_loader, self.test_loader, self.ccp
