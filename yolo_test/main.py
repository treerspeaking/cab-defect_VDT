from typing import Any, Dict, List, Tuple

import torch
import torchvision.transforms as T
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.data.dataset import ClassificationDataset
from ultralytics.models.yolo.classify import ClassificationTrainer
from ultralytics.nn.tasks import ClassificationModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK

class FocalLoss(torch.nn.Module):
    """
    Focal Loss implementation for multi-class classification.
    
    Args:
        alpha (float or tensor): Weighting factor for rare class (default: 1.0)
        gamma (float): Focusing parameter (default: 2.0)
        reduction (str): Specifies the reduction to apply to the output
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        # print("created focal loss")
    
    def forward(self, inputs, targets):
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        # print("passing through the focal loss")
        # Compute probabilities
        pt = torch.exp(-ce_loss)
        
        # Compute focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CustomizedDataset(ClassificationDataset):
    def init(self, root: str, args, augment: bool = False, prefix: str = ""):
        super().init(root, args, augment, prefix)
        train_transforms = T.Compose(
            [
                T.Resize((args.imgsz, args.imgsz)),
                T.RandomHorizontalFlip(p=args.fliplr),
                T.RandomVerticalFlip(p=args.flipud),
                T.RandomAffine(args.degrees, [args.translate, args.translate], shear=args.shear),
                T.RandAugment(interpolation=T.InterpolationMode.BILINEAR),
                T.ColorJitter(brightness=args.hsv_v, contrast=args.hsv_v, saturation=args.hsv_s, hue=args.hsv_h),
                T.ToTensor(),
                T.Normalize(mean=torch.tensor(0), std=torch.tensor(1)),
                T.RandomErasing(p=args.erasing, scale=[0.02, 0.05], ratio=[], inplace=True),
            ]
        )
        val_transforms = T.Compose(
            [
                T.Resize((args.imgsz, args.imgsz)),
                T.ToTensor(),
                T.Normalize(mean=torch.tensor(0), std=torch.tensor(1)),
            ]
        )
        self.torch_transforms = train_transforms if augment else val_transforms
        
class Myv8ClassificationLoss:
    """Criterion class for computing training losses for classification."""
    def __init__(self):
        self.loss_func = FocalLoss(1, 1, reduction="mean")
        
    def __call__(self, preds: Any, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the classification loss between predictions and true labels."""
        # print("calculated focal loss")
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = self.loss_func(preds, batch["cls"])
        return loss, loss.detach()
        
class MyClassificationModel(ClassificationModel):
    def init_criterion(self):
        """Initialize the loss criterion for the ClassificationModel."""
        return Myv8ClassificationLoss()

class CustomizedTrainer(ClassificationTrainer):
    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        return CustomizedDataset(root=img_path, args=self.args, augment=mode == "train", prefix=mode)
    
    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        """
        Return a modified PyTorch model configured for training YOLO classification.

        Args:
            cfg (Any, optional): Model configuration.
            weights (Any, optional): Pre-trained model weights.
            verbose (bool, optional): Whether to display model information.

        Returns:
            (ClassificationModel): Configured PyTorch model for classification.
        """
        model = MyClassificationModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        for m in model.modules():
            if not self.args.pretrained and hasattr(m, "reset_parameters"):
                m.reset_parameters()
            if isinstance(m, torch.nn.Dropout) and self.args.dropout:
                m.p = self.args.dropout  # set dropout
        for p in model.parameters():
            p.requires_grad = True  # for training
        return model

model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)

results = model.train(
    data="/home/treerspeaking/src/python/cabdefect/yolo_test/yolo_data", 
    trainer=CustomizedTrainer, 
    epochs=1000, 
    imgsz=512,
    cfg="/home/treerspeaking/src/python/cabdefect/yolo_test/aug.yaml"
    )
