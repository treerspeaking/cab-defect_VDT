import argparse
from typing import List

import lightning as L
from lightning.pytorch.profilers import AdvancedProfiler
from lightning.pytorch.utilities.combined_loader import CombinedLoader
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
from lightning.pytorch.callbacks import ModelCheckpoint

from dataset.Dataset import MultiTaskCabDataset
from networks.net_factory import net_factory
from utils.ramps import cosine_ramp_down
from dataset.SupervisedDataset import SupervisedDataModule

NO_LABEL = -1

# Define augmentations for training and validation
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

                                  
val_aug = v2.Compose([
    v2.ToTensor(),
    v2.Resize([224, 224]),
    # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    v2.Normalize(mean=[0, 0 , 0], std=[1, 1, 1]),
])

class TaskSpecificHead(torch.nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.in_channels = in_features
        self.num_classes = num_classes

        self.classifier = torch.nn.Linear(in_features, num_classes, bias=False)
        
    def forward(self, input):
        return self.classifier(input)

class SupervisedModel(L.LightningModule):
    def __init__(self, task: dict, cfg: dict, method="supervised_base"):
        super().__init__()
        self.in_features = 960
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=NO_LABEL)
        self.accuracy_metrics = torch.nn.ModuleList()
        self.cfg = cfg
        self.lr = self.cfg["lr"]
        self.momentum = self.cfg["momentum"]
        self.weight_decay = self.cfg["weight_decay"]
        self.nesterov = self.cfg["nesterov"]
        self.task_specific_heads = torch.nn.ModuleList()
        self.task = task  # Store the task names
        self.task_names = list(task.keys())
        # this is temporary fix
        self.num_classes = 2
        self.backbone = net_factory("MobileNetV3Feature", pretrained=True)
        
        for k, v in self.task.items():
            num_classes = len(v)
            self.task_specific_heads.append(TaskSpecificHead(self.in_features, num_classes))
            # Add an accuracy metric for this task
            self.accuracy_metrics.append(Accuracy(task="multiclass", num_classes=num_classes))
        
        self.save_hyperparameters() 
        
    
    def training_step(self, batch, batch_idx):
        # For supervised learning, we only need labeled data
        # Assuming batch format: (data, labels) where labels is a list of task labels
        data, labels = batch
        
        # Extract features using the backbone
        features = self.backbone(data)
        
        total_loss = None  # Start with None instead of 0
        metrics = {}
        valid_tasks = 0
        
        # Process each task
        for task_idx, (head, task_labels) in enumerate(zip(self.task_specific_heads, labels)):
            task_name = self.task_names[task_idx] 
            logits = head(features)
            
            # Get labeled indices and filter out NO_LABEL entries
            labeled_indices = task_labels >= 0
            
            if labeled_indices.sum() > 0:
                # Supervised loss on labeled data
                sup_loss = self.ce_loss(
                    logits[labeled_indices], 
                    task_labels[labeled_indices]
                )
                
                # Calculate accuracy for labeled data
                task_acc = self.accuracy_metrics[task_idx](
                    logits[labeled_indices], 
                    task_labels[labeled_indices]
                )
                
                # Add to total loss only if we have labeled data
                if total_loss is None:
                    total_loss = sup_loss
                else:
                    total_loss = total_loss + sup_loss
                    
                valid_tasks += 1
            else:
                # No labeled data for this task
                sup_loss = torch.tensor(0.0, device=self.device)
                task_acc = torch.tensor(0.0, device=self.device)
            
            # Log task-specific metrics
            metrics[f"train/{task_name}_acc"] = task_acc
            metrics[f"train/{task_name}_loss"] = sup_loss
        
        # If no tasks had any labeled data, create a dummy loss that's connected to the model
        if total_loss is None:
            # Create a loss connected to the model by getting a small value from a parameter
            dummy_param = next(iter(self.parameters()))
            total_loss = 0.0 * dummy_param.sum()
        elif valid_tasks > 1:
            # Average losses across tasks that had labeled data
            total_loss = total_loss / valid_tasks
        
        # Log global metrics
        metrics["train/learning_rate"] = self.trainer.optimizers[0].param_groups[0]['lr']
        metrics["train/total_loss"] = total_loss
        
        self.log_dict(metrics)
        return total_loss
    
    # def validation_step(self, batch, batch_idx):
    #     inputs, labels = batch
    #     labels = labels[0]
        
    #     logits = self.backbone(inputs)
    #     val_loss = self.ce_loss(logits, labels)
        
    #     probs = F.softmax(logits, dim=1)
    #     _, predicted = torch.max(probs, dim=1)
    #     correct = (predicted == labels).sum().item()
    #     total = labels.size(0)
    #     accuracy = correct / total if total > 0 else 0
        
    #     # Calculate precision, recall, and accuracy for each class
    #     precisions = []
    #     recalls = []
    #     class_accuracies = []
        
    #     # For micro F1 calculation
    #     total_true_positives = 0
    #     total_false_positives = 0
    #     total_false_negatives = 0
        
    #     for class_idx in range(self.num_classes):
    #         # True positives: predicted class_idx and actual class_idx
    #         true_positives = ((predicted == class_idx) & (labels == class_idx)).sum().item()
            
    #         # False positives: predicted class_idx but not actual class_idx
    #         false_positives = ((predicted == class_idx) & (labels != class_idx)).sum().item()
            
    #         # False negatives: not predicted class_idx but actual class_idx
    #         false_negatives = ((predicted != class_idx) & (labels == class_idx)).sum().item()
            
    #         # Accumulate for micro F1
    #         total_true_positives += true_positives
    #         total_false_positives += false_positives
    #         total_false_negatives += false_negatives
            
    #         # Total samples of this class
    #         class_total = (labels == class_idx).sum().item()
            
    #         # Calculate precision and recall (handling division by zero)
    #         precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    #         recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            
    #         # Calculate per-class accuracy (same as recall, but logged separately for clarity)
    #         class_accuracy = true_positives / class_total if class_total > 0 else 0
            
    #         precisions.append(precision)
    #         recalls.append(recall)
    #         class_accuracies.append(class_accuracy)
            
    #         # Log per-class metrics
    #         self.log(f"val/precision_class_{class_idx}", precision, on_step=False, on_epoch=True)
    #         self.log(f"val/recall_class_{class_idx}", recall, on_step=False, on_epoch=True)
    #         self.log(f"val/accuracy_class_{class_idx}", class_accuracy, on_step=False, on_epoch=True)
        
    #     # Calculate micro precision, recall, and F1
    #     micro_precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
    #     micro_recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0
    #     micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
        
    #     # Calculate and log average precision and recall (macro averaging)
    #     avg_precision = sum(precisions) / len(precisions) if precisions else 0
    #     avg_recall = sum(recalls) / len(recalls) if recalls else 0
    #     avg_class_accuracy = sum(class_accuracies) / len(class_accuracies) if class_accuracies else 0
        
    #     metrics = {
    #         "val/loss": val_loss,
    #         "val/accuracy": accuracy,
    #         "val/precision": avg_precision,
    #         "val/recall": avg_recall,
    #         "val/avg_class_accuracy": avg_class_accuracy,
    #         "val/micro_precision": micro_precision,
    #         "val/micro_recall": micro_recall,
    #         "val/micro_f1": micro_f1
    #     }
        
    #     self.log_dict(metrics)
    #     return val_loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        
        # Extract features using the backbone (same as in training_step)
        features = self.backbone(inputs)
        
        total_loss = None
        metrics = {}
        valid_tasks = 0
        
        # Process each task
        for task_idx, (head, task_labels) in enumerate(zip(self.task_specific_heads, labels)):
            task_name = self.task_names[task_idx]
            num_classes = head.num_classes
            
            # Get predictions for this task
            logits = head(features)
            
            # Get labeled indices and filter out NO_LABEL entries
            labeled_indices = task_labels >= 0
            
            if labeled_indices.sum() > 0:
                # Validation loss on labeled data
                val_loss = self.ce_loss(
                    logits[labeled_indices], 
                    task_labels[labeled_indices]
                )
                
                # Calculate task-specific metrics for labeled data only
                task_logits = logits[labeled_indices]
                task_labels_filtered = task_labels[labeled_indices]
                
                probs = F.softmax(task_logits, dim=1)
                _, predicted = torch.max(probs, dim=1)
                
                # Calculate precision, recall, and accuracy for each class in this task
                precisions = []
                recalls = []
                class_accuracies = []
                
                # For micro F1 calculation
                total_true_positives = 0
                total_false_positives = 0
                total_false_negatives = 0
                
                for class_idx in range(num_classes):
                    # True positives: predicted class_idx and actual class_idx
                    true_positives = ((predicted == class_idx) & (task_labels_filtered == class_idx)).sum().item()
                    
                    # False positives: predicted class_idx but not actual class_idx
                    false_positives = ((predicted == class_idx) & (task_labels_filtered != class_idx)).sum().item()
                    
                    # False negatives: not predicted class_idx but actual class_idx
                    false_negatives = ((predicted != class_idx) & (task_labels_filtered == class_idx)).sum().item()
                    
                    # Accumulate for micro F1
                    total_true_positives += true_positives
                    total_false_positives += false_positives
                    total_false_negatives += false_negatives
                    
                    # Total samples of this class
                    class_total = (task_labels_filtered == class_idx).sum().item()
                    
                    # Calculate precision and recall (handling division by zero)
                    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                    
                    # Calculate per-class accuracy
                    class_accuracy = true_positives / class_total if class_total > 0 else 0
                    
                    precisions.append(precision)
                    recalls.append(recall)
                    class_accuracies.append(class_accuracy)
                    
                    # Log per-class metrics
                    self.log(f"val/{task_name}_precision_class_{class_idx}", precision, on_step=False, on_epoch=True)
                    self.log(f"val/{task_name}_recall_class_{class_idx}", recall, on_step=False, on_epoch=True)
                    self.log(f"val/{task_name}_accuracy_class_{class_idx}", class_accuracy, on_step=False, on_epoch=True)
                
                # Calculate micro precision, recall, and F1 for this task
                micro_precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
                micro_recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0
                micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
                
                # Calculate and log average precision and recall (macro averaging)
                avg_precision = sum(precisions) / len(precisions) if precisions else 0
                avg_recall = sum(recalls) / len(recalls) if recalls else 0
                avg_class_accuracy = sum(class_accuracies) / len(class_accuracies) if class_accuracies else 0
                
                # Overall accuracy for this task
                correct = (predicted == task_labels_filtered).sum().item()
                total = task_labels_filtered.size(0)
                accuracy = correct / total if total > 0 else 0
                
                # # Log task-specific metrics
                # metrics[f"val/{task_name}_loss"] = val_loss
                # metrics[f"val/{task_name}_accuracy"] = accuracy
                # metrics[f"val/{task_name}_precision"] = avg_precision
                # metrics[f"val/{task_name}_recall"] = avg_recall
                # metrics[f"val/{task_name}_avg_class_accuracy"] = avg_class_accuracy
                # metrics[f"val/{task_name}_micro_f1"] = micro_f1
                
                # for now
                metrics[f"val/loss"] = val_loss
                metrics[f"val/accuracy"] = accuracy
                metrics[f"val/precision"] = avg_precision
                metrics[f"val/ecall"] = avg_recall
                metrics[f"val/avg_class_accuracy"] = avg_class_accuracy
                metrics[f"val/micro_f1"] = micro_f1
                
                # Add to total loss
                if total_loss is None:
                    total_loss = val_loss
                else:
                    total_loss = total_loss + val_loss
                    
                valid_tasks += 1
                
        # Average the total loss if we have multiple valid tasks
        if total_loss is None:
            # Create a dummy loss connected to the model
            dummy_param = next(iter(self.parameters()))
            total_loss = 0.0 * dummy_param.sum()
        elif valid_tasks > 1:
            total_loss = total_loss / valid_tasks
        
        # Log the overall loss
        metrics["val/total_loss"] = total_loss
        
        self.log_dict(metrics)
        return total_loss

    def ramp_lr(self, steps):
        return cosine_ramp_down(steps, self.ramp_down_length, out_multiplier=0.5, in_multiplier=0.4375)

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(self.parameters(), self.lr,
        #                         momentum=self.momentum,
        #                         weight_decay=self.weight_decay,
        #                         nesterov=self.nesterov)
        optimizer = torch.optim.AdamW(self.parameters(), 0.001667, [0.9, 0.999], weight_decay=0.0005)
        # Set up learning rate scheduler if needed
        if hasattr(self, 'trainer') and hasattr(self.trainer, 'datamodule'):
            self.ramp_down_length = self.cfg["ramp_down_length"] * self.trainer.datamodule.steps_per_epochs
            scheduler = LambdaLR(optimizer, self.ramp_lr)
            
            lr_scheduler_config = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": None,
            }
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
        else:
            return optimizer
    
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
    
    with open(args.config, "r") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    # Create supervised data module (only labeled data)
    data = SupervisedDataModule(
        dataset=MultiTaskCabDataset, 
        root="./data_more_label", 
        batch_size=cfg.get("batch_size", cfg.get("labeled_batch_size", 32)),
        val_batch_size=cfg.get("val_batch_size", 32), 
        num_workers=4, 
        train_transforms=train_aug,
        val_transforms=val_aug, 
        test_transforms=val_aug, 
        download=False
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val/accuracy",  # Monitor validation accuracy
        save_top_k=5,           # Save the top 3 models
        mode="max",             # "max" because we want to maximize accuracy
        filename="supervised{epoch:02d}-{val/accuracy:.4f}",  # Filename pattern
        save_last=True,         # Additionally save the last model
    )

    trainer = L.Trainer(
        max_steps=cfg.get("steps", -1), 
        max_epochs=cfg.get("epochs", 100),
        enable_checkpointing=True, 
        callbacks=[checkpoint_callback],
        # log_every_n_steps=cfg.get("val_check_interval", 50), 
        check_val_every_n_epoch=cfg.get("check_val_every_n_epoch", 1), 
        # val_check_interval=cfg.get("val_check_interval", None),
        benchmark=True,
    )

    # Create supervised model with the same tasks
    model = SupervisedModel({
        # 'co_dinh_cap': ['Ảnh sai', 'Ảnh đúng'],
        'han_box': ['close', 'open']
    }, cfg)

    trainer.fit(model=model, datamodule=data)

if __name__ == "__main__":
    main()