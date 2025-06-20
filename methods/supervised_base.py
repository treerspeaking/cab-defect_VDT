import argparse
from typing import List
import lightning as L
from lightning.pytorch.profilers import AdvancedProfiler
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import SaveConfigCallback, LightningArgumentParser
from torchmetrics import Accuracy
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
from dataset.SupervisedDataset import SupervisedDataModule
from dataset.Dataset import MultiTaskCabDataset
from networks.net_factory import net_factory


from utils.ramps import cosine_ramp_down, sigmoid_ramp_up

NO_LABEL = -1
# parser = argparse.ArgumentParser()
# parser.add_argument('--config', type=str, required=True)

# args = parser.parse_args()
# cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
# cfg = yaml.load(open("/home/treerspeaking/src/python/cabdefect/configs/fixmatch.yaml", "r"), Loader=yaml.Loader)
cfg = yaml.load(open("/home/treerspeaking/src/python/cabdefect/configs/fixmatch.yaml", "r"), Loader=yaml.Loader)



# train_aug = v2.Compose([
#     v2.ToTensor(),
#     v2.Resize([224, 224]),
#     v2.RandomHorizontalFlip(0.5),
#     v2.RandomVerticalFlip(0.5),
#     v2.RandomAffine(0, [0.1, 0.1], shear=0),
#     v2.RandAugment(interpolation=v2.InterpolationMode.BILINEAR),
#     v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.7, hue=0.015),
#     # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     v2.Normalize(mean=[0, 0 , 0], std=[1, 1, 1]),
#     v2.RandomErasing(p=0.3, inplace=True),
# ])

train_aug = v2.Compose(
            [
                v2.Resize([224, 224]),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomAffine(0, [0.1, 0.1], shear=0),
                v2.RandAugment(interpolation=v2.InterpolationMode.BILINEAR),
                v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.7, hue=0.015),
                v2.ToTensor(),
                v2.Normalize(mean=[0, 0 , 0], std=[1, 1, 1]),
                v2.RandomErasing(p=0.3, inplace=True),
            ]
        )

                                  
test_aug = transforms=v2.Compose([
    v2.ToTensor(),
    v2.Resize([224, 224]),
    # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    v2.Normalize(mean=[0, 0 , 0], std=[1, 1, 1]),
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



class TaskSpecificHead(torch.nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.in_channels = in_features
        self.num_classes = num_classes

        self.classifier = torch.nn.Linear(in_features, num_classes, bias=False)
        
    def forward(self, input):
        return self.classifier(input)

class FixMatchBase(L.LightningModule):
    def __init__(self, task: dict, cfg: dict, method="fixmatch_base"):
        super().__init__()
        self.num_classes = 2
        self.backbone = net_factory(cfg["network"], num_classes=2, pretrained=True)
        self.in_features = 960
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=NO_LABEL, reduction="mean")
        # self.ce_loss = FocalLoss(2, 2)
        self.accuracy_metrics = torch.nn.ModuleList()
        self.cfg = cfg
        self.threshold = self.cfg["threshold"]
        self.lr = torch.tensor(self.cfg["lr"])
        self.momentum=torch.tensor(self.cfg["momentum"])
        self.weight_decay=torch.tensor(self.cfg["weight_decay"])
        self.nesterov=torch.tensor(self.cfg["nesterov"])
        self.task = task  # Store the task names
        self.task_names = list(task.keys())
        self.ramp_up_length = cfg["ramp_up_length"]
    
        
        self.save_hyperparameters() 
    
    
    def training_step(self, batch, batch_idx):
    
        labeled_data, labels = batch
        # sorry future Viet Tung but im 2 lazy to fix the dataloader
        labels = labels[0]
        
        # Concatenate all data and run backbone only once
        weak_data = labeled_data

        weak_data_logits = self.backbone(weak_data)
        
        weak_data_prob = F.softmax(weak_data_logits, dim=1)
        
        weak_data_confidence, weak_data_pseudo = torch.max(weak_data_prob, dim=1)
        
        # sup loss
        sup_loss = self.ce_loss(weak_data_logits, labels)
        
        
        total_loss = sup_loss 
        total_labeled_loss = sup_loss
        metrics = {}
        
        # Log global metrics
        metrics["train/learning_rate"] = self.trainer.optimizers[0].param_groups[0]['lr']
        metrics["train/labeled_loss"] = total_labeled_loss
        metrics["train/total_loss"] = total_loss
        metrics["train/confidence"] = weak_data_confidence.mean()
        
        self.log_dict(metrics)
        return total_loss
    
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        labels = labels[0]
        
        logits = self.backbone(inputs)
        val_loss = self.ce_loss(logits, labels)
        
        probs = F.softmax(logits, dim=1)
        _, predicted = torch.max(probs, dim=1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        accuracy = correct / total if total > 0 else 0
        
        # Calculate precision, recall, and accuracy for each class
        precisions = []
        recalls = []
        class_accuracies = []
        
        # For micro F1 calculation
        total_true_positives = 0
        total_false_positives = 0
        total_false_negatives = 0
        
        for class_idx in range(self.num_classes):
            # True positives: predicted class_idx and actual class_idx
            true_positives = ((predicted == class_idx) & (labels == class_idx)).sum().item()
            
            # False positives: predicted class_idx but not actual class_idx
            false_positives = ((predicted == class_idx) & (labels != class_idx)).sum().item()
            
            # False negatives: not predicted class_idx but actual class_idx
            false_negatives = ((predicted != class_idx) & (labels == class_idx)).sum().item()
            
            # Accumulate for micro F1
            total_true_positives += true_positives
            total_false_positives += false_positives
            total_false_negatives += false_negatives
            
            # Total samples of this class
            class_total = (labels == class_idx).sum().item()
            
            # Calculate precision and recall (handling division by zero)
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            
            # Calculate per-class accuracy (same as recall, but logged separately for clarity)
            class_accuracy = true_positives / class_total if class_total > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
            class_accuracies.append(class_accuracy)
            
            # Log per-class metrics
            self.log(f"val/precision_class_{class_idx}", precision, on_step=False, on_epoch=True)
            self.log(f"val/recall_class_{class_idx}", recall, on_step=False, on_epoch=True)
            self.log(f"val/accuracy_class_{class_idx}", class_accuracy, on_step=False, on_epoch=True)
        
        # Calculate micro precision, recall, and F1
        micro_precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
        micro_recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
        
        # Calculate and log average precision and recall (macro averaging)
        avg_precision = sum(precisions) / len(precisions) if precisions else 0
        avg_recall = sum(recalls) / len(recalls) if recalls else 0
        avg_class_accuracy = sum(class_accuracies) / len(class_accuracies) if class_accuracies else 0
        
        metrics = {
            "val/loss": val_loss,
            "val/accuracy": accuracy,
            "val/precision": avg_precision,
            "val/recall": avg_recall,
            "val/avg_class_accuracy": avg_class_accuracy,
            "val/micro_precision": micro_precision,
            "val/micro_recall": micro_recall,
            "val/micro_f1": micro_f1
        }
        
        self.log_dict(metrics)
        return val_loss

    def ramp_up(self, steps):
        return sigmoid_ramp_up(steps, self.ramp_up_length)
    
    def ramp_lr(self, steps):
        return cosine_ramp_down(steps, self.ramp_down_length, out_multiplier=0.5, in_multiplier=0.4375)

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=cfg["lr"], betas=(cfg["adam_beta_1"], cfg["adam_beta_2_during_ramp_up"]), eps=1e-8)
        # optimizer = torch.optim.SGD(self.parameters(), self.lr,
        #                         momentum=self.momentum,
        #                         weight_decay=self.weight_decay,
        #                         nesterov=self.nesterov)
        optimizer = torch.optim.AdamW(self.parameters(), 0.001667, [0.9, 0.999], weight_decay=0.0005)
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
    
    data = SupervisedDataModule(
        dataset=MultiTaskCabDataset, 
        root="./data_more_label", 
        batch_size=cfg.get("batch_size", cfg.get("labeled_batch_size", 32)),
        val_batch_size=cfg.get("val_batch_size", 32), 
        num_workers=2, 
        train_transforms=train_aug,
        val_transforms=test_aug, 
        test_transforms=test_aug, 
        download=False
    )

    
    checkpoint_callback = ModelCheckpoint(
        monitor="val/accuracy",  # Monitor validation accuracy
        save_top_k=5,           # Save the top 3 models
        mode="max",             # "max" because we want to maximize accuracy
        filename="supervised_base-{epoch:02d}-{val/accuracy:.4f}",  # Filename pattern
        save_last=True,         # Additionally save the last model
    )
    
    trainer = L.Trainer(
        max_steps=cfg["steps"], 
        max_epochs=cfg["epochs"],
        enable_checkpointing=True, 
        # log_every_n_steps=cfg["val_check_interval"], 
        check_val_every_n_epoch=1, 
        # val_check_interval=cfg["val_check_interval"],
        callbacks=[checkpoint_callback],
        # accumulate_grad_batches = 1
        # callbacks=[SaveConfigCallback(parser=LightningArgumentParser(parser_mode="yaml", default_env=True, skip_validation=True), config=cfg)],
        benchmark=True,
        )


    trainer.fit(model=FixMatchBase({
    # 'han_box': ['open', 'close']
    'co_dinh_cap': ["co_dinh_dung", "co_dinh_sai"]
    }, cfg), datamodule=data)

if __name__ == "__main__":
    main()
