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

from .BaseDataset import BaseDataset, BasicLabelDataset, BasicUnLabelDataset, BaseDataModule


class MTDataset(BaseDataset):
    def get_dataloader(
        self, 
        labeled_batch_size: int, 
        labeled_num_worker: int,
        shuffle: bool, 
        unlabeled: bool = False, 
        labeled_size: int = None,
        unlabeled_batch_size: Union[int, float] = None, 
        unlabeled_num_worker: int = None,
        seed: int = None, 
        ):
        """return the dataloader for the dataset
        if the use both labeled and unlabeled set the flag unlabeled to true and other unlabeled input
        else don't fill the unlabled arguments

        Args:
            labeled_batch_size (int): _description_
            labeled_num_worker (int): _description_
            shuffle (bool): _description_
            unlabeled (bool, optional): _description_. Defaults to False.
            labeled_size (int, float): number of labeled image, if float divide by percentage 
            unlabeled_batch_size (int, optional): _description_. Defaults to None.
            unlabeled_num_worker (int, optional): _description_. Defaults to None.
            seed (int, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        
        if self.train:
            if unlabeled:
                X_label, X_unlabeled, y_labeled ,y_unlabeled = self.split_labeled_unlabeled_data(X = self.data, y = self.targets, labeled_size = labeled_size, random_state = seed)
                labeled_ds = MTLabelDataset(X_label, y_labeled, self.transforms)
                unlabeled_ds = MTUnLabelDataset(X_unlabeled, self.transforms)
                unlabeled_dataloader = DataLoader(unlabeled_ds, unlabeled_batch_size, shuffle, num_workers=unlabeled_num_worker, persistent_workers=True)
                labeled_dataloader = DataLoader(labeled_ds, labeled_batch_size, shuffle, num_workers=labeled_num_worker, persistent_workers=True)
                return labeled_dataloader, unlabeled_dataloader
            
            ds = MTLabelDataset(self.data, self.targets, self.transforms)
            labeled_dataloader = DataLoader(ds, labeled_batch_size, shuffle, num_workers=labeled_num_worker, persistent_workers=True)
            return labeled_dataloader
        
        ds = BasicLabelDataset(self.data, self.targets, self.transforms)
        test_dataloader = DataLoader(ds, labeled_batch_size, shuffle, num_workers=labeled_num_worker, persistent_workers=True)
        
        return test_dataloader



class MTLabelDataset(BasicLabelDataset):
    def __init__(self, data, targets, teacher_transforms, student_transforms):
        self.data = data
        self.targets = targets
        self.teacher_transform = teacher_transforms
        self.student_transforms = student_transforms
    
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        tea_img, stu_img = img, img
        
        if self.teacher_transform is not None:
            tea_img = self.teacher_transform(img)
        
        if self.student_transforms is not None:
            stu_img = self.student_transforms(img)
        
        return tea_img, stu_img, target
    
class MTUnLabelDataset(BasicUnLabelDataset):
    def __init__(self, data, teacher_transforms, student_transforms):
        self.data = data
        self.teacher_transform = teacher_transforms
        self.student_transforms = student_transforms
    
    def __getitem__(self, index):
        img = self.data[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        tea_img, stu_img = img, img
        
        if self.teacher_transform is not None:
            tea_img = self.teacher_transform(img)
        
        if self.student_transforms is not None:
            stu_img = self.student_transforms(img)
        
        return tea_img, stu_img

class MTDataModule(BaseDataModule):
    def __init__(
        self,
        dataset: BaseDataset,
        root: str = "./data",
        train_labeled_batch_size: int = 32,
        train_unlabeled_batch_size: int = 8,
        val_batch_size: int = 32,
        num_workers: int = 4,
        student_transforms: Optional[Transform] = None,
        teacher_transforms: Optional[Transform] = None,
        val_transforms: Optional[Transform] = None,
        test_transforms: Optional[Transform] = None,
        labeled_size: Union[int, float] = 0.1,
        seed: int | None = None,
        download: bool = False,
    ):
        super().__init__(
            dataset = dataset,
            root = root,
            train_labeled_batch_size = train_labeled_batch_size,
            train_unlabeled_batch_size = train_unlabeled_batch_size,
            val_batch_size = val_batch_size,
            num_workers = num_workers,
            val_transforms = val_transforms,
            test_transforms = test_transforms,
            labeled_size = labeled_size,
            seed = seed,
            download = download,
        )
        self.student_transforms = student_transforms
        self.teacher_transforms = teacher_transforms

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        
        
        if stage == "fit":
            # mnist_full = self.dataset(self.root, split="train", transform=self.transform)
            train_data = self.dataset(self.root, split="train", download=False)
            val_data = self.dataset(self.root, split="val", download=False)
            
            X_label, X_unlabeled, y_labeled, y_unlabeled = train_data.split_labeled_unlabeled_data(train_data.data, train_data.targets, self.labeled_size, self.seed)
            
            self.labeled_dataset = MTLabelDataset(X_label, y_labeled, self.student_transforms, self.student_transforms)
            self.unlabeled_dataset = MTUnLabelDataset(X_unlabeled, self.student_transforms, self.student_transforms)
            
            self.val_dataset = BasicLabelDataset(val_data.data, val_data.targets, self.val_transforms)
            self._steps_per_epochs = int(math.ceil(max(float(len(self.labeled_dataset)) / self.train_labeled_batch_size, float(len(self.unlabeled_dataset)) / self.train_unlabeled_batch_size)))

        if stage == "test":
            super().setup(stage)

        if stage == "predict":
            super().setup(stage)