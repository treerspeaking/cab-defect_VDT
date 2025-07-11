import os
from typing import Any, Callable, Optional, Tuple, Union
import itertools
from pathlib import Path
from abc import ABC, abstractmethod
import math

import pickle
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision.datasets import CIFAR10
from torchvision.transforms.v2 import Transform
from torchvision.datasets.utils import check_integrity, download_url, verify_str_arg
import lightning as L
from lightning.pytorch.utilities.combined_loader import CombinedLoader

class BaseDataset():
    
    def __init__(self, split: str, download=True):
        self.split = split
        # self.transforms = transforms
        self.download=download
    
    # @staticmethod
    def split_labeled_unlabeled_data(self, X, y, labeled_size: Union[int, float], random_state:int = None):
        """_summary_

        Args:
            X (_type_): train_input
            y (_type_): train_output
            train_size (Union[int, float]): number of training images
            random_state (int, optional): the random training state, none mean random state

        Returns:
            _splitting : list, length=2 * len(arrays)
                List containing label-unlabels split of inputs.
        """
        # stratify ensure that it is an equal distribution split for each class
        return train_test_split(X, y, train_size=labeled_size, stratify=y, random_state=random_state)
    
class BasicLabelDataset(Dataset):
    
    def __init__(self, data, targets, transforms=None):
        super().__init__()
        self.data = data
        self.targets = targets
        self.transforms = transforms
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        
        # return two different view at once
        if self.transforms is not None:
            return self.transforms(img), target
        return img, target

class BasicUnLabelDataset(Dataset):
    def __init__(self, data, transforms=None):
        super().__init__()
        self.data = data
        self.transforms = transforms
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img = self.data[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        
        if self.transforms is not None:
            return self.transforms(img)
        
        return img

class BaseDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset: BaseDataset,
        root: str = "./data",
        train_labeled_batch_size: int = 32,
        train_unlabeled_batch_size: int = 8,
        val_batch_size: int = 32,
        num_workers: int = 4,
        train_transforms: Optional[Transform] = None,
        val_transforms: Optional[Transform] = None,
        test_transforms: Optional[Transform] = None,
        labeled_size: Union[int, float] = 0.1,
        seed: int | None = None,
        download: bool = False,
    ):
        super().__init__()
        self.root = root
        self.dataset = dataset
        self.train_labeled_batch_size = train_labeled_batch_size
        self.train_unlabeled_batch_size = train_unlabeled_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        self.labeled_size = labeled_size
        self.seed = seed
        self.download = download

    def prepare_data(self):
        if self.download == True:
            self.dataset(self.root, split="train", download=True)
            self.dataset(self.root, split="val", download=True)
            self.dataset(self.root, split="test", download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            # mnist_full = self.dataset(self.root, split="train", transform=self.transform)
            train_data = self.dataset(self.root, split="train", download=False)
            val_data = self.dataset(self.root, split="val", download=False)
            
            X_label, X_unlabeled, y_labeled, y_unlabeled = train_data.split_labeled_unlabeled_data(train_data.data, train_data.targets, self.labeled_size, self.seed)
            
            self.labeled_dataset = BasicLabelDataset(X_label, y_labeled, self.train_transforms)
            self.unlabeled_dataset = BasicUnLabelDataset(X_unlabeled, self.train_transforms)
            
            self.val_dataset = BasicLabelDataset(val_data.data, val_data.targets, self.val_transforms)
            self._steps_per_epochs = int(math.ceil(max(float(len(self.labeled_dataset)) / self.train_labeled_batch_size, float(len(self.unlabeled_dataset)) / self.train_unlabeled_batch_size)))

        if stage == "test":
            test_data = self.dataset(self.root, split="test", download=False)
            self.test_dataset = BasicLabelDataset(test_data.data, test_data.targets, self.test_transforms)

        if stage == "predict":
            test_data = self.dataset(self.root, split="test", download=False)
            self.test_dataset = BasicLabelDataset(test_data.data, test_data.targets, self.test_transforms)

    def train_dataloader(self):
        
        labeled_dataloader = DataLoader(self.labeled_dataset, self.train_labeled_batch_size, True, num_workers=self.num_workers, pin_memory=True)
        unlabeled_dataloader = DataLoader(self.unlabeled_dataset, self.train_unlabeled_batch_size, True, num_workers=self.num_workers, pin_memory=True)
        self._steps_per_epochs = max(len(labeled_dataloader), len(unlabeled_dataloader))
        return CombinedLoader([labeled_dataloader, unlabeled_dataloader], mode="max_size_cycle")

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.val_batch_size, False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.val_batch_size, False, num_workers=self.num_workers, pin_memory=True)

    def predict_dataloader(self):
        return self.test_dataloader()
    
    @property
    def steps_per_epochs(self):
        return self._steps_per_epochs
    
    @steps_per_epochs.setter
    def steps_per_epochs(self, steps_per_epochs):
        self._steps_per_epochs = steps_per_epochs


