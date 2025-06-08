from PIL import Image

import math

from .BaseDataset import BaseDataModule, BasicLabelDataset, BasicUnLabelDataset


class FixMatchLabelDataset(BasicLabelDataset):
    def __init__(self, data, targets, weak_transforms, strong_transforms):
        super().__init__(data, targets)
        self.weak_transforms = weak_transforms
        self.strong_transforms = strong_transforms
    
    def __getitem__(self, index):
        
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        
        # Apply weak and strong augmentations to get two different views
        weak_img = self.weak_transforms(img) if self.weak_transforms is not None else img
        strong_img = self.strong_transforms(img) if self.strong_transforms is not None else img
        
        # Return both augmented views along with the target
        return weak_img, strong_img, target
    
class FixMatchUnLabelDataset(BasicUnLabelDataset):
    def __init__(self, data, weak_transforms, strong_transforms):
        super().__init__(data)
        self.weak_transforms = weak_transforms
        self.strong_transforms = strong_transforms
    
    def __getitem__(self, index):
        
        img = self.data[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        
        # Apply weak and strong augmentations to get two different views
        weak_img = self.weak_transforms(img) if self.weak_transforms is not None else img
        strong_img = self.strong_transforms(img) if self.strong_transforms is not None else img
        
        # Return both augmented views along with the target
        return weak_img, strong_img
    
    

class FixMatchDataModule(BaseDataModule):
    def __init__(
        self,
        dataset,
        root="./data",
        train_labeled_batch_size=32,
        train_unlabeled_batch_size=8,
        val_batch_size=32,
        num_workers=4,
        weak_transforms=None,
        strong_transforms=None,
        val_transforms=None,
        test_transforms=None,
        labeled_size=0.1, 
        seed=None, 
        download=False
    ):
        super().__init__(dataset, root, train_labeled_batch_size, train_unlabeled_batch_size, val_batch_size,
                         num_workers, None, val_transforms, test_transforms, labeled_size, seed, download)
        self.weak_transform = weak_transforms
        self.strong_transform = strong_transforms
    
    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            # mnist_full = self.dataset(self.root, split="train", transform=self.transform)
            train_data = self.dataset(self.root, split="train", download=False)
            val_data = self.dataset(self.root, split="val", download=False)
            
            # self._tasks = train_data.tasks
            
            X_label, X_unlabeled, y_labeled, y_unlabeled = train_data.split_labeled_unlabeled_data(train_data.data, train_data.targets, self.labeled_size, self.seed)
            
            self.labeled_dataset = FixMatchLabelDataset(X_label, y_labeled, self.weak_transform, self.strong_transform)
            self.unlabeled_dataset = FixMatchUnLabelDataset(X_unlabeled, self.weak_transform, self.strong_transform)
            
            self.val_dataset = BasicLabelDataset(val_data.data, val_data.targets, self.val_transforms)
            self._steps_per_epochs = int(math.ceil(max(float(len(self.labeled_dataset)) / self.train_labeled_batch_size, float(len(self.unlabeled_dataset)) / self.train_unlabeled_batch_size)))

        if stage == "test":
            super().setup(stage)

        if stage == "predict":
            super().setup(stage)
    
    # @property
    # def tasks(self):
    #     return self._tasks
