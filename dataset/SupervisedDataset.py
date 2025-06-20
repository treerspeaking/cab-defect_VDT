import math

from torch.utils.data import DataLoader
import lightning as L
# Simple supervised data module for labeled data only
from .BaseDataset import BasicLabelDataset, BaseDataset

class SupervisedDataModule(L.LightningDataModule):
    def __init__(self, 
                 dataset: BaseDataset, 
                 root, 
                 batch_size, 
                 val_batch_size, 
                 num_workers, 
                 train_transforms, 
                 val_transforms, 
                 test_transforms, 
                 download=False
                 ):
        super().__init__()
        self.dataset = dataset
        self.root = root
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        self.download = download
        
    def setup(self, stage: str):
        if stage == "fit":
            train_data = self.dataset(self.root, split="train", download=False)
            val_data = self.dataset(self.root, split="val", download=False)
            
            self._tasks = train_data.tasks
            
            X_label, X_unlabeled, y_label, y_unlabeled = train_data.split_labeled_unlabeled_data(train_data.data, train_data.targets)
            
            self.train_dataset = BasicLabelDataset(X_label, y_label, self.train_transforms)
            self.val_dataset = BasicLabelDataset(val_data.data, val_data.targets, self.val_transforms)
            
            self._steps_per_epochs = int(math.ceil(float(len(self.train_dataset)) / self.batch_size))
            
        if stage == "test":
            test_data = self.dataset(self.root, split="test", download=False)
            self.test_dataset = BasicLabelDataset(test_data.data, test_data.targets, self.test_transforms)

        if stage == "predict":
            test_data = self.dataset(self.root, split="test", download=False)
            self.test_dataset = BasicLabelDataset(test_data.data, test_data.targets, self.test_transforms)

            
            
            # self.train_dataset = self.dataset(
            #     root=self.root,
            #     split="train",
            #     transforms=self.train_transforms,
            #     download=self.download
            # )
            # self.val_dataset = self.dataset(
            #     root=self.root,
            #     split="val", 
            #     transforms=self.val_transforms,
            #     download=self.download
            # )
            
        # if stage == "test":
        #     self.test_dataset = self.dataset(
        #         root=self.root,
        #         split="test",
        #         transforms=self.test_transforms,
        #         download=self.download
        #     )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    @property
    def steps_per_epochs(self):
        return len(self.train_dataloader())