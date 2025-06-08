import os
from typing import Any, Callable, Optional, Tuple, Union, List, Dict
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

from .MTDataset import MTDataset
from .BaseDataset import BaseDataset

SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}

class MultiTaskCabDataset(BaseDataset):
    """
    Dataset for multi-task semi-supervised learning with cabinet defect images.
    
    Each sample can have multiple task labels or no label for specific tasks.
    Task labels are stored as a list of task-specific labels, with -1 indicating
    that the sample is not labeled for that task.
    """
    
    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        train_transforms: Optional[Callable] = None,
        test_transforms: Optional[Callable] = None,
        download: bool = False,
    ):
        super().__init__(split=split, download=download)
        self.root = Path(root) if isinstance(root, str) else root
        self.train_transform = train_transforms
        self.test_transform = test_transforms
        
        # Set train flag based on split
        self.train = (split == "train")
        
        # Detect all tasks and their classes
        self.tasks = self._discover_tasks()
        
        # Load images and their multi-task labels
        self.data, self.targets = self._load_data()
        
    def split_labeled_unlabeled_data(self, X, y, labeled_size: Union[int, float, None] = None, random_state:int = None):
        """_summary_

        Args:
            X (_type_): train_input
            y (_type_): train_output

        Returns:
            _splitting : list, length=2 * len(arrays)
                List containing label-unlabels split of inputs.
        """
        X_label = []
        y_label = []
        X_unlabeled = []
        y_unlabeled = []
        
        for item in zip(X, y):
            labeled = False
            # check if there is labeled in label
            for i in item[1]:
                if i >= 0:
                    labeled = True
                    break
            if labeled:
                X_label.append(item[0])
                y_label.append(item[1])
            else:
                X_unlabeled.append(item[0])
                y_unlabeled.append(item[1])
                
        return X_label, X_unlabeled, y_label, y_unlabeled
        
        
    def _get_image_files(self, directory: Path) -> List[Path]:
        """
        Get all image files in a directory with supported formats.
        
        Args:
            directory: Directory to search for images
            
        Returns:
            List of image file paths
        """
        image_files = []
        for fmt in SUPPORTED_FORMATS:
            # Use glob pattern without the dot for glob
            pattern = f"*{fmt}"
            image_files.extend(directory.glob(pattern))
        return image_files

        
    def _discover_tasks(self) -> Dict[str, List[str]]:
        """
        Discover all tasks and their possible class labels from directory structure.
        
        Returns:
            Dict mapping task names to lists of possible class values
        """
        tasks = {}
        labeled_dir = self.root / "train_data" / "labeled"
        
        if not labeled_dir.exists():
            raise RuntimeError(f"Labeled data directory not found: {labeled_dir}")
        
        # Find all task directories
        for task_dir in labeled_dir.iterdir():
            if task_dir.is_dir():
                task_name = task_dir.name
                # Find all class labels for this task
                class_labels = [d.name for d in task_dir.iterdir() if d.is_dir()]
                tasks[task_name] = class_labels
                print(f"Discovered task: {task_name} with classes: {class_labels}")
        
        return tasks
    
    def _load_data(self) -> Tuple[List[Path], List[List[int]]]:
        """
        Load all images and their multi-task labels.
        
        Returns:
            Tuple of (image_paths, task_labels)
        """
        images = []
        labels = []
        
        # Mapping from task name to index in labels list
        task_indices = {task: i for i, task in enumerate(self.tasks.keys())}
        
        if self.train:
            # Process labeled data
            labeled_dir = self.root / "train_data" / "labeled"
            
            # First, collect all images with their task-specific labels
            image_labels = {}  # Maps image path to its task labels
            
            for task_name, task_idx in task_indices.items():
                task_dir = labeled_dir / task_name
                
                # Process each class folder
                for class_dir in task_dir.iterdir():
                    if class_dir.is_dir():
                        class_label = self.tasks[task_name].index(class_dir.name)
                        
                        # Process all images in this class
                        for img_path in self._get_image_files(class_dir):
                            if img_path not in image_labels:
                                image_labels[img_path] = [-1] * len(self.tasks)
                            
                            # Set the label for this task
                            image_labels[img_path][task_idx] = class_label
            
            # Convert to list format
            for img_path, task_labels in image_labels.items():
                img = np.array(Image.open(img_path).convert('RGB'))
                images.append(img)
                labels.append(task_labels)
            
            # Process unlabeled data if in training mode
            unlabeled_dir = self.root / "train_data" / "unlabeled"
            if unlabeled_dir.exists():
                for img_path in self._get_image_files(unlabeled_dir):
                    # Add with all tasks set to -1 (unlabeled)
                    img = np.array(Image.open(img_path).convert('RGB'))
                    images.append(img)
                    labels.append([-1] * len(self.tasks))
        
        else:  # Test or validation data
            data_dir = self.root / "test_data"
            if not data_dir.exists():
                raise RuntimeError(f"Test data directory not found: {data_dir}")
                
            # Process all task directories
            image_labels = {}
            
            for task_name, task_idx in task_indices.items():
                task_dir = data_dir / task_name
                if not task_dir.exists():
                    continue
                    
                for class_dir in task_dir.iterdir():
                    if class_dir.is_dir():
                        class_label = self.tasks[task_name].index(class_dir.name)
                        
                        for img_path in self._get_image_files(class_dir):
                            if img_path not in image_labels:
                                image_labels[img_path] = [-1] * len(self.tasks)
                            
                            image_labels[img_path][task_idx] = class_label
            
            # Convert to list format
            for img_path, task_labels in image_labels.items():
                img = np.array(Image.open(img_path).convert('RGB'))
                images.append(img)
                labels.append(task_labels)
        
        print(f"Loaded {len(images)} images for {self.split} split")
        return images, labels
    
    def __len__(self):
        return len(self.data)



class MySVHNMT(BaseDataset):
    split_list = {
        "train": [
            "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
            "train_32x32.mat",
            "e26dedcc434d2e4c54c9b2d4a06d8373",
        ],
        "test": [
            "http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
            "test_32x32.mat",
            "eb5a983be6a315427106f1b164d9cef3",
        ],
        "extra": [
            "http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
            "extra_32x32.mat",
            "a93ce644f1a588dc4d68dda5feec44a7",
        ],
    }

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        transforms: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        if isinstance(root, str):
            root = os.path.expanduser(root)
        self.root = root
        self.split = verify_str_arg(split, "split", tuple(self.split_list.keys()))
        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

        self.data = loaded_mat["X"]
        # loading from the .mat file gives an np.ndarray of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.targets = loaded_mat["y"].astype(np.int64).squeeze()

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(self.targets, self.targets == 10, 0)
        # self.data = np.transpose(self.data, (3, 2, 0, 1))
        self.data = np.transpose(self.data, (3, 0, 1, 2))
        
        if split == "train" or split == "extra":
            self.train = True
        else:
            self.train = False
            
        self.transforms = transforms
        
    def _check_integrity(self) -> bool:
        root = self.root
        md5 = self.split_list[self.split][2]
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)
    
    def download(self) -> None:
        md5 = self.split_list[self.split][2]
        download_url(self.url, self.root, self.filename, md5)

class MyCifar10MT(MTDataset):
    
    base_folder = "./cifar-10-batches-py"
    meta_data = "batches.meta"
    training_list = [
        "data_batch_1",
        "data_batch_2",
        "data_batch_3",
        "data_batch_4",
        "data_batch_5",
        ]
    test_list = [
        "test_batch",
        ]
    
    def __init__(self, train: bool, transforms: Transform):
        """

        Args:
            train (Bool): if True this will read train dataset else it will read test dataset
            transforms (Transform): the transformations that will be apply to the dataset
        """
        super().__init__(train, transforms)
        
        self.data = []
        self.targets = []
        
        if train:
            self.file_list = self.training_list
        else:
            self.file_list = self.test_list
        
        for file_name in self.file_list:
            fpath = os.path.join(os.path.curdir, self.base_folder, file_name)
            with open(fpath, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry["data"])
                self.targets.extend(entry["labels"])
        
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1)) 
        
    def _load_meta(self):
        path = os.path.join(os.path.curdir, self.base_folder, self.meta_data)
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta_data["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}
        
class MyCifar10(BaseDataset):
    
    base_folder = "./cifar-10-batches-py"
    meta_data = "batches.meta"
    training_list = [
        "data_batch_1",
        "data_batch_2",
        "data_batch_3",
        "data_batch_4",
        "data_batch_5",
        ]
    test_list = [
        "test_batch",
        ]
    
    def __init__(self,root , split: str, download: bool):
        """

        Args:
            train (Bool): if True this will read train dataset else it will read test dataset
            transforms (Transform): the transformations that will be apply to the dataset
        """
        # super().__init__(split)
        
        self.data = []
        self.targets = []
        # Validate split parameter
        if split not in ["train", "test", "val"]:
            raise ValueError(f"Split '{split}' not recognized. Expected one of: 'train', 'test', 'val'")
        if split == "train":
            self.file_list = self.training_list
        else:
            self.file_list = self.test_list
        
        for file_name in self.file_list:
            fpath = os.path.join(os.path.curdir, self.base_folder, file_name)
            with open(fpath, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry["data"])
                self.targets.extend(entry["labels"])
        
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1)) 
        
    def _load_meta(self):
        path = os.path.join(os.path.curdir, self.base_folder, self.meta_data)
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta_data["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}