import argparse
from typing import List
import lightning as L
from lightning.pytorch.profilers import AdvancedProfiler
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from lightning.pytorch.cli import SaveConfigCallback, LightningArgumentParser
from torchmetrics import Accuracy, Precision, Recall, F1Score
import torch
from torch.optim.lr_scheduler import LambdaLR
from torchvision.transforms import v2
import torch.nn.functional as F
import numpy as np
from types import SimpleNamespace
import yaml
from lightning.pytorch.cli import LightningCLI
# from lightning.pytorch.trainer

from dataset.FixMatchDataset import FixMatchDataModule
from dataset.Dataset import MultiTaskCabDataset
from networks.net_factory import net_factory


from utils.ramps import cosine_ramp_down

NO_LABEL = -1
# parser = argparse.ArgumentParser()
# parser.add_argument('--config', type=str, required=True)

# args = parser.parse_args()
# cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
# cfg = yaml.load(open("/home/treerspeaking/src/python/cabdefect/configs/fixmatch.yaml", "r"), Loader=yaml.Loader)
cfg = yaml.load(open("/home/treerspeaking/src/python/cabdefect/configs/fixmatch.yaml", "r"), Loader=yaml.Loader)




weak_aug = v2.Compose([
    v2.ToTensor(),
    v2.Resize([224, 224]),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    v2.RandomAffine(degrees=0, translate=(0.126, 0.126)),
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
])

strong_aug = v2.Compose([
    v2.ToTensor(),
    v2.Resize([224, 224]),
    v2.RandAugment(),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

                                  
test_aug = transforms=v2.Compose([
    v2.ToTensor(),
    v2.Resize([224, 224]),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', ignore_index=-1):
        """
        Focal Loss implementation
        
        Args:
            alpha: Weighting factor for rare class (default: 1)
            gamma: Focusing parameter (default: 2)
            reduction: Specifies the reduction to apply to the output
            ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(self, inputs, targets):
        # Create mask for valid targets (not equal to ignore_index)
        valid_mask = targets != self.ignore_index
        
        # If no valid targets, return zero loss
        if not valid_mask.any():
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        
        # Compute cross entropy with ignore_index
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        
        # Apply mask to get only valid losses
        ce_loss = ce_loss[valid_mask]
        
        # Compute p_t for valid targets only
        pt = torch.exp(-ce_loss)
        
        # Compute focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


data = FixMatchDataModule(
    dataset=MultiTaskCabDataset, 
    root="./data", 
    train_labeled_batch_size=cfg["labeled_batch_size"], 
    train_unlabeled_batch_size=cfg["unlabeled_batch_size"], 
    val_batch_size=cfg["val_batch_size"], 
    num_workers=4, 
    weak_transforms=weak_aug,
    strong_transforms=strong_aug,
    val_transforms=test_aug, 
    test_transforms=test_aug, 
    # labeled_size=cfg["labeled_sample"], 
    download=False
    )

class TaskSpecificHead(torch.nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.in_channels = in_features
        self.num_classes = num_classes

        self.classifier = torch.nn.Linear(in_features, num_classes, bias=False)
        
    def forward(self, input):
        return self.classifier(input)

class FixMatch(L.LightningModule):
    def __init__(self, task: dict, cfg: dict, method="fixmatch"):
        super().__init__()
        self.backbone = net_factory("MobileNetV3Feature")
        self.in_features = 576
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=NO_LABEL)
        # self.ce_loss = torch.nn
        self.accuracy_metrics = torch.nn.ModuleList()
        # self.acc = Accuracy('multiclass', num_classes=cfg["num_classes"])
        # self.ramp_down_length = cfg["ramp_down_length"] * steps_per_epochs
        self.cfg = cfg
        self.threshold=0.95
        # self.threshold = self.cfg["threshold"]
        self.lr = torch.tensor(self.cfg["lr"])
        self.momentum=torch.tensor(self.cfg["momentum"])
        self.weight_decay=torch.tensor(self.cfg["weight_decay"])
        self.nesterov=torch.tensor(self.cfg["nesterov"])
        self.task_specific_heads= torch.nn.ModuleList()
        self.task = task  # Store the task names
        self.task_names = list(task.keys())
        
        for k, v in self.task.items():
            num_classes = len(v)
            self.task_specific_heads.append(TaskSpecificHead(self.in_features, num_classes))
            # Add an accuracy metric for this task
            self.accuracy_metrics.append([
                Accuracy(task="multiclass", num_classes=num_classes),
                ])
    
        
        self.save_hyperparameters() 
    
    
    def training_step(self, batch, batch_idx):
    
        weak_augmented_labeled_data, strong_augmented_labeled_data, labels = batch[0]
        weak_augmented_unlabeled_data, strong_augmented_unlabeled_data = batch[1]
        labeled_data_len = len(weak_augmented_labeled_data)
        unlabeled_data_len = len(weak_augmented_unlabeled_data)
        
        # Concatenate all data and run backbone only once
        all_data = torch.concat([
            weak_augmented_labeled_data,
            weak_augmented_unlabeled_data,
            strong_augmented_unlabeled_data,
            strong_augmented_labeled_data
        ], dim=0)

        all_features = self.backbone(all_data)

        # Split the features back into their respective parts
        weak_labeled_features, weak_unlabeled_features, strong_unlabeled_features, strong_labeled_features = torch.split(
            all_features, 
            [labeled_data_len, unlabeled_data_len, unlabeled_data_len, labeled_data_len],
            dim=0
        )
        
        total_loss = 0
        total_labeled_loss = 0
        total_unlabeled_loss = 0
        metrics = {}
        
        # Process each task
        for task_idx, (head, task_labels) in enumerate(zip(self.task_specific_heads, labels)):
            # Get predictions for weakly augmented labeled data
            task_name = self.task_names[task_idx] 
            weak_labeled_logits = head(weak_labeled_features)
            
            # Get labeled indices and filter out NO_LABEL entries
            labeled_indices = task_labels >= 0
            if labeled_indices.sum() > 0:
                # Supervised loss on labeled data
                sup_loss = self.ce_loss(
                    weak_labeled_logits[labeled_indices], 
                    task_labels[labeled_indices]
                )
                
                # Calculate accuracy for labeled data
                task_acc = self.accuracy_metrics[task_idx](
                    weak_labeled_logits[labeled_indices], 
                    task_labels[labeled_indices]
                )
            else:
                sup_loss = torch.tensor(0.0, device=self.device)
                task_acc = torch.tensor(0.0, device=self.device)
            
            # Get pseudo-labels from weak augmentation of unlabeled data
            with torch.no_grad():
                weak_unlabeled_logits = head(weak_unlabeled_features)
                weak_unlabled_labeled_logits = weak_labeled_logits[~labeled_indices]
                weak_logits = torch.concat((weak_unlabled_labeled_logits, weak_unlabeled_logits), dim=0)
                pseudo_probs = F.softmax(weak_logits, dim=1)
                max_probs, pseudo_labels = torch.max(pseudo_probs, dim=1)
                # Create mask for confident predictions
                mask = max_probs >= self.threshold
            
            # Get predictions for strongly augmented unlabeled data
            strong_features = torch.concat((strong_labeled_features[~labeled_indices], strong_unlabeled_features), dim=0)
            strong_unlabeled_logits = head(strong_features)
            
            # Unsupervised loss with pseudo-labels (only for confident predictions)
            if mask.sum() > 0:
                unsup_loss = self.ce_loss(
                    strong_unlabeled_logits[mask], 
                    pseudo_labels[mask],
                )
            else:
                unsup_loss = torch.tensor(0.0, device=self.device)
            
            # Combine losses for this task
            task_loss = sup_loss + unsup_loss
            total_loss += task_loss
            total_labeled_loss += sup_loss
            total_unlabeled_loss += unsup_loss
            
            # Log task-specific metrics
            metrics[f"train/{task_name}_acc"] = task_acc
            # metrics[f"train/task_{task_name}_sup_loss"] = sup_loss
            # metrics[f"train/task_{task_name}_unsup_loss"] = unsup_loss
            metrics[f"train/{task_name}_use_sample %"] = mask.float().mean()
        
        # Average losses across tasks
        num_tasks = len(self.task_specific_heads)
        total_loss = total_loss / num_tasks
        total_labeled_loss = total_labeled_loss / num_tasks
        total_unlabeled_loss = total_unlabeled_loss / num_tasks
        
        # Log global metrics
        metrics["train/learning_rate"] = self.trainer.optimizers[0].param_groups[0]['lr']
        metrics["train/labeled_loss"] = total_labeled_loss
        metrics["train/unlabeled_loss"] = total_unlabeled_loss
        metrics["train/total_loss"] = total_loss
        
        self.log_dict(metrics)
        return total_loss
    
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step for multi-task FixMatch model.
        Only processes labeled validation data since we don't have pseudo-labels during validation.
        """
        
        # Unpack validation batch - should contain labeled data and labels for each task
        labeled_data, labels = batch
        
        # Extract features using the backbone
        features = self.backbone(labeled_data)
        
        total_loss = 0
        metrics = {}
        
        # Process each task
        for task_idx, (head, task_labels) in enumerate(zip(self.task_specific_heads, labels)):
            # Get predictions for this task
            task_name = self.task_names[task_idx]
            logits = head(features)
            
            # Get labeled indices (filter out NO_LABEL entries)
            labeled_indices = task_labels >= 0
            
            if labeled_indices.sum() > 0:
                # Calculate validation loss only on labeled data
                val_loss = self.ce_loss(
                    logits[labeled_indices], 
                    task_labels[labeled_indices]
                )
                
                # Calculate accuracy for this task
                task_acc = self.accuracy_metrics[task_idx](
                    logits[labeled_indices], 
                    task_labels[labeled_indices]
                )
                
                # Calculate additional metrics
                with torch.no_grad():
                    probs = F.softmax(logits[labeled_indices], dim=1)
                    max_probs, predictions = torch.max(probs, dim=1)
                    confidence = max_probs.mean()
            else:
                # No labeled data for this task in this batch
                val_loss = torch.tensor(0.0, device=self.device)
                task_acc = torch.tensor(0.0, device=self.device)
                confidence = torch.tensor(0.0, device=self.device)
            
            total_loss += val_loss
            
            # Log task-specific validation metrics
            metrics[f"val/{task_name}_loss"] = val_loss
            metrics[f"val/{task_name}_acc"] = task_acc
            metrics[f"val/{task_name}_confidence"] = confidence
            metrics[f"val/{task_name}_labeled_samples"] = labeled_indices.sum().float()
        
        # Average loss across tasks
        num_tasks = len(self.task_specific_heads)
        total_loss = total_loss / num_tasks
        
        # Log global validation metrics
        metrics["val/total_loss"] = total_loss
        metrics["val/avg_acc"] = torch.stack([metrics[f"val/{task_name}_acc"] for task_name in self.task_names]).mean()
        metrics["val/avg_confidence"] = torch.stack([metrics[f"val/{task_name}_confidence"] for task_name in self.task_names]).mean()
        
        # Log all metrics
        self.log_dict(metrics, on_epoch=True, on_step=False, prog_bar=True)
        
        return total_loss

    
    def ramp_lr(self, steps):
        return cosine_ramp_down(steps, self.ramp_down_length, out_multiplier=0.5, in_multiplier=0.4375)

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=cfg["lr"], betas=(cfg["adam_beta_1"], cfg["adam_beta_2_during_ramp_up"]), eps=1e-8)
        optimizer = torch.optim.SGD(self.parameters(), self.lr,
                                momentum=self.momentum,
                                weight_decay=self.weight_decay,
                                nesterov=self.nesterov)
        # optimizer = torch.optim.Adam(self.parameters(), lr=3e-3, betas=(0.9, 0.99), eps=1e-8)
        # self.ramp_down_length = cfg["ramp_down_length"] * len(self.trainer.train_dataloader)
        self.ramp_down_length =  self.cfg["ramp_down_length"] * self.trainer.datamodule.steps_per_epochs
        scheduler = LambdaLR(optimizer, self.ramp_lr)
        
        lr_scheduler_config = {
            # REQUIRED: The scheduler instance
            "scheduler": scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "step",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            "name": None,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
    
    def forward(self, input):
        features_maps = self.backbone.backbone.features(input) # B C H W
        global_avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        flatten = torch.nn.Flatten()
        pooled_feat = flatten(global_avg_pool(features_maps))
        batch_size, channels, height, width = features_maps.shape
        
        # Calculate total number of CAMs (one per class across all tasks)
        total_classes = sum(len(class_list) for class_list in self.task.values())
        cam_map = torch.zeros((batch_size, total_classes, height, width), dtype=torch.float32, device=input.device)
        
        cam_idx = 0
        task_names = list(self.task.keys())
        out = []
        
        for head_idx, head in enumerate(self.task_specific_heads):
            task_name = task_names[head_idx]
            class_values = self.task[task_name]
            
            # Get the weight matrix from the classifier
            weight = head.classifier.weight  # [num_classes, in_features]
            
            # Get predictions for this task
            logits = head(pooled_feat)
            probs = F.softmax(logits, dim=1)
            max_probs, pred_classes = torch.max(probs, dim=1)
            
            # Store prediction results
            task_result = {
                'task_name': task_name,
                'probabilities': probs,
                'predicted_class_names': [class_values[idx.item()] for idx in pred_classes]
            }
            out.append(task_result)
            
            for class_idx in range(len(class_values)):
                # Get the weights for this specific class
                class_weight = weight[class_idx]  # [in_features]
                
                # Reshape to match the feature maps for multiplication
                class_weight = class_weight.view(1, -1, 1, 1)
                
                # Multiply feature maps with class weights and sum along the channel dimension
                class_cam = (features_maps * class_weight).sum(dim=1)
                
                # Store in the output tensor
                cam_map[:, cam_idx] = class_cam
                cam_idx += 1
        
        return cam_map, out
    


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    global cfg
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    

    trainer = L.Trainer(
        max_steps=cfg["steps"], 
        max_epochs=cfg["epochs"],
        enable_checkpointing=True, 
        log_every_n_steps=cfg["val_check_interval"], 
        check_val_every_n_epoch=None, 
        val_check_interval=cfg["val_check_interval"],
        # accumulate_grad_batches = 1
        # callbacks=[SaveConfigCallback(parser=LightningArgumentParser(parser_mode="yaml", default_env=True, skip_validation=True), config=cfg)],
        benchmark=True,
        )


    trainer.fit(model=FixMatch({
        'co_dinh_cap': ['Ảnh sai', 'Ảnh đúng'],
    #  'bo_chia': ['Ảnh đúng', 'Ảnh lỗi'],
    'han_box': ['han_open', 'han_close']}, cfg), datamodule=data)

if __name__ == "__main__":
    main()
