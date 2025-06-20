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
from lightning.pytorch.callbacks import ModelCheckpoint
# from lightning.pytorch.trainer

from dataset.FixMatchDataset import FixMatchDataModule
from dataset.Dataset import MultiTaskCabDataset
from networks.net_factory import net_factory

from utils.loss import FocalLossV2
from utils.ramps import cosine_ramp_down, sigmoid_ramp_up

NO_LABEL = -1
# parser = argparse.ArgumentParser()
# parser.add_argument('--config', type=str, required=True)

# args = parser.parse_args()
# cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
# cfg = yaml.load(open("/home/treerspeaking/src/python/cabdefect/configs/fixmatch.yaml", "r"), Loader=yaml.Loader)
cfg = yaml.load(open("/home/treerspeaking/src/python/cabdefect/configs/fixmatch.yaml", "r"), Loader=yaml.Loader)


weak_aug = v2.Compose([
    v2.ToTensor(),
    v2.Resize([672, 672]),
    v2.CenterCrop([512, 512]),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    v2.RandomAffine(degrees=0, translate=(0.0625, 0.0625)),
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
])

strong_aug = v2.Compose([
    v2.ToTensor(),
    v2.Resize([672, 672]),
    v2.CenterCrop([512, 512]),
    v2.RandAugment(),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

                                  
test_aug = transforms=v2.Compose([
    v2.ToTensor(),
    v2.Resize([672, 672]),
    v2.CenterCrop([512, 512]),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

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
        self.in_features = 960
        # self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=NO_LABEL)
        self.ce_loss = FocalLossV2(alpha=1, gamma=0.25)
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
        
        # this is temporary fix
        self.num_classes = 2
        self.backbone = net_factory(self.cfg["network"], pretrained=True)
        
        for k, v in self.task.items():
            num_classes = len(v)
            self.task_specific_heads.append(TaskSpecificHead(self.in_features, num_classes))
            # Add an accuracy metric for this task
            self.accuracy_metrics.append(Accuracy(task="multiclass", num_classes=num_classes))
        
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
        inputs, labels = batch
        
        # Extract features using the backbone (same as in training_step)
        features = self.backbone(inputs)
        
        total_loss = None
        metrics = {}
        valid_tasks = 0
        
        # For overall accuracy
        all_preds = []
        all_labels = []
        
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
                
                # For overall accuracy
                all_preds.append(predicted)
                all_labels.append(task_labels_filtered)
                
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
                metrics[f"val/{task_name}_loss"] = val_loss
                metrics[f"val/{task_name}_accuracy"] = accuracy
                metrics[f"val/{task_name}_precision"] = avg_precision
                metrics[f"val/{task_name}_recall"] = avg_recall
                metrics[f"val/{task_name}_avg_class_accuracy"] = avg_class_accuracy
                metrics[f"val/{task_name}_micro_f1"] = micro_f1
                
                # Add to total loss
                if total_loss is None:
                    total_loss = val_loss
                else:
                    total_loss = total_loss + val_loss
                    
                valid_tasks += 1
                
        # After all tasks, compute overall accuracy
        if all_preds and all_labels:
            all_preds_cat = torch.cat(all_preds)
            all_labels_cat = torch.cat(all_labels)
            overall_correct = (all_preds_cat == all_labels_cat).sum().item()
            overall_total = all_labels_cat.size(0)
            overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0
            metrics["val/accuracy"] = overall_accuracy
        
        # Average the total loss if we have multiple valid tasks
        if total_loss is None:
            # Create a dummy loss connected to the model
            dummy_param = next(iter(self.parameters()))
            total_loss = 0.0 * dummy_param.sum()
        elif valid_tasks > 1:
            total_loss = total_loss / valid_tasks
        
        # Log the overall loss
        metrics["val/total_loss"] = total_loss
        
        self.log_dict(metrics, on_epoch=True)
        return total_loss

    
    def ramp_lr(self, steps):
        if (steps < self.ramp_up_length):
            return sigmoid_ramp_up(steps, self.ramp_up_length)
        else:
            return cosine_ramp_down(steps, self.ramp_down_length, out_multiplier=0.5, in_multiplier=0.4375)

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=cfg["lr"], betas=(cfg["adam_beta_1"], cfg["adam_beta_2_during_ramp_up"]), eps=1e-8)
        # optimizer = torch.optim.SGD(self.parameters(), self.lr,
        #                         momentum=self.momentum,
        #                         weight_decay=self.weight_decay,
        #                         nesterov=self.nesterov)
        optimizer = torch.optim.AdamW(self.parameters(), self.lr, [0.9, 0.999], weight_decay=self.weight_decay)
        # optimizer = torch.optim.Adam(self.parameters(), lr=3e-3, betas=(0.9, 0.99), eps=1e-8)
        # self.ramp_down_length = cfg["ramp_down_length"] * len(self.trainer.train_dataloader)
        self.ramp_down_length = self.cfg["ramp_down_length"] * self.trainer.datamodule.steps_per_epochs
        self.ramp_up_length = self.cfg["ramp_up_length"] * self.trainer.datamodule.steps_per_epochs
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
    
    def forward(self, inputs):
        # lore help me one time
        inputs, labels = inputs
        features_maps = self.backbone.features(inputs) # B C H W
        # why does using features compare to average pool cause the diff
        global_avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        flatten = torch.nn.Flatten()
        relu = torch.nn.ReLU()
        pooled_feat = flatten(global_avg_pool(features_maps))
        batch_size, channels, height, width = features_maps.shape
        
        # Calculate total number of CAMs (one per class across all tasks)
        total_classes = sum(len(class_list) for class_list in self.task.values())
        cam_map = torch.zeros((batch_size, total_classes, height, width), dtype=torch.float32, device=inputs.device)
        
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
                cam_map[:, cam_idx] = relu(class_cam)
                cam_idx += 1
        
        return cam_map, out
    


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    

    checkpoint_callback = ModelCheckpoint(
        monitor="val/total_loss",  # Monitor validation accuracy
        save_top_k=100,           # Save the top 3 models
        every_n_epochs=25,
        mode="min",             # "max" because we want to maximize accuracy
        filename="supervised{epoch:02d}-{val/accuracy:.4f}-{val/total_loss}",  # Filename pattern
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
    
    
    data = FixMatchDataModule(
        dataset=MultiTaskCabDataset, 
        root="./data_more_label", 
        train_labeled_batch_size=cfg["labeled_batch_size"], 
        train_unlabeled_batch_size=cfg["unlabeled_batch_size"], 
        val_batch_size=cfg["val_batch_size"], 
        num_workers=2, 
        weak_transforms=weak_aug,
        strong_transforms=strong_aug,
        val_transforms=test_aug, 
        test_transforms=test_aug, 
        # labeled_size=cfg["labeled_sample"], 
        download=False
    )


    data.setup("fit")
    tasks = data._tasks

    # Create supervised model with the same tasks
    model = FixMatch(tasks, cfg)

    trainer.fit(model=model, datamodule=data)

if __name__ == "__main__":
    main()
