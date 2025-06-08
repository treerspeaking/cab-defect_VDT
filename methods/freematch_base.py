import argparse
from typing import List
import lightning as L
import torch
import torch.nn.functional as F
import numpy as np
import yaml
from torch.optim.lr_scheduler import LambdaLR
from lightning.pytorch.callbacks import ModelCheckpoint
from torchvision.transforms import v2
from torchmetrics import Accuracy, Precision, Recall

from dataset.FixMatchDataset import FixMatchDataModule
from dataset.Dataset import MultiTaskCabDataset
from networks.net_factory import net_factory
from utils.ramps import cosine_ramp_down, sigmoid_ramp_up

NO_LABEL = -1
cfg = yaml.load(open("/home/treerspeaking/src/python/cabdefect/configs/fixmatch.yaml", "r"), Loader=yaml.Loader)

# Same augmentations as your FixMatch implementation
weak_aug = v2.Compose([
    v2.ToTensor(),
    v2.Resize([224, 224]),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    v2.RandomAffine(degrees=0, translate=(0.126, 0.126)),
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
])

strong_aug = v2.Compose([
    v2.ToTensor(),
    v2.Resize([224, 224]),
    v2.RandAugment(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_aug = v2.Compose([
    v2.ToTensor(),
    v2.Resize([224, 224]),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create the data module
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
    download=False
)

class FreeMatch(L.LightningModule):
    def __init__(self, task: dict, cfg: dict, method="freematch"):
        super().__init__()
        self.backbone = net_factory(cfg["network"], num_classes=2)
        self.in_features = 576
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=NO_LABEL)
        self.accuracy_metrics = torch.nn.ModuleList()
        self.cfg = cfg
        
        # FreeMatch specific parameters
        self.threshold_warmup_epochs = cfg.get("threshold_warmup_epochs", 10)
        self.max_threshold = cfg.get("max_threshold", 0.95)
        self.min_threshold = cfg.get("min_threshold", 0.5)
        self.ema_decay = cfg.get("ema_decay", 0.999)
        
        # Initialize adaptive threshold - starts at min and grows to max
        self.register_buffer("threshold", torch.tensor(self.min_threshold))
        
        # Class distribution statistics
        self.num_classes = 2  # Binary classification for 'han_open' and 'han_close'
        self.register_buffer("prob_sum", torch.zeros(self.num_classes))
        self.register_buffer("prob_count", torch.zeros(1))
        
        # Standard parameters from FixMatch
        self.lr = torch.tensor(self.cfg["lr"])
        self.momentum = torch.tensor(self.cfg["momentum"])
        self.weight_decay = torch.tensor(self.cfg["weight_decay"])
        self.nesterov = torch.tensor(self.cfg["nesterov"])
        self.task = task
        self.task_names = list(task.keys())
        self.ramp_up_length = cfg["ramp_up_length"]
        
        self.save_hyperparameters()
    
    def training_step(self, batch, batch_idx):
        weak_augmented_labeled_data, strong_augmented_labeled_data, labels = batch[0]
        labels = labels[0]  # Using your convention
        weak_augmented_unlabeled_data, strong_augmented_unlabeled_data = batch[1]
        
        # Concatenate all data and run backbone only once
        weak_data = torch.concat([
            weak_augmented_labeled_data,
            weak_augmented_unlabeled_data,
        ], dim=0)

        weak_data_logits = self.backbone(weak_data)
        
        weak_data_label_logits, weak_data_unlabeled_logits = torch.split(
            weak_data_logits, 
            [len(weak_augmented_labeled_data), len(weak_augmented_unlabeled_data)],
            dim=0
        )

        # Supervised loss
        sup_loss = self.ce_loss(weak_data_label_logits, labels)
        
        # Process unlabeled data with FreeMatch algorithm
        weak_data_prob = F.softmax(weak_data_unlabeled_logits, dim=1)
        weak_data_confidence, weak_data_pseudo = torch.max(weak_data_prob, dim=1)
        
        # Update class distribution estimation
        with torch.no_grad():
            # Average probabilities across batch
            prob_avg = weak_data_prob.mean(0)
            self.prob_sum = self.ema_decay * self.prob_sum + (1 - self.ema_decay) * prob_avg * len(weak_data_prob)
            self.prob_count = self.ema_decay * self.prob_count + (1 - self.ema_decay) * len(weak_data_prob)
            
            # Update adaptive threshold based on training progress
            if self.current_epoch < self.threshold_warmup_epochs:
                # Linear warmup
                warmup_factor = self.current_epoch / self.threshold_warmup_epochs
                self.threshold = torch.tensor(self.min_threshold + (self.max_threshold - self.min_threshold) * warmup_factor).to(self.device)
            
            # Distribution alignment (DA) for balanced pseudo-labeling
            if self.prob_count > 0:
                estimated_distribution = self.prob_sum / self.prob_count
                # Avoid division by zero
                estimated_distribution = torch.clamp(estimated_distribution, min=1e-6)
                # Normalize to ensure it sums to 1
                estimated_distribution = estimated_distribution / estimated_distribution.sum()
                
                # Apply distribution alignment to confidence scores
                # This adjusts thresholds per class based on their frequency
                class_thresholds = self.threshold / estimated_distribution
                class_thresholds = torch.clamp(class_thresholds, max=0.95)  # Cap at 0.95
                
                # Create mask using class-specific thresholds
                mask = torch.zeros_like(weak_data_confidence, dtype=torch.bool)
                for c in range(self.num_classes):
                    c_mask = (weak_data_pseudo == c)
                    mask[c_mask] = weak_data_confidence[c_mask] > class_thresholds[c]
            else:
                # Fallback to standard threshold when no stats available
                mask = weak_data_confidence > self.threshold
        
        # Get logits for strongly augmented unlabeled samples that passed the threshold
        strong_above_threshold_logits = self.backbone(strong_augmented_unlabeled_data[mask])
        
        # Unsupervised loss - only if we have samples above threshold
        if mask.sum() > 0:
            unsup_loss = self.ce_loss(strong_above_threshold_logits, weak_data_pseudo[mask])
        else:
            unsup_loss = torch.tensor(0.0, device=self.device)
        
        # Total loss with ramp-up factor
        unsup_weight = self.ramp_up(self.current_epoch)
        # unsup_weight = 1
        total_loss = sup_loss + unsup_loss * unsup_weight
        
        # Log metrics
        metrics = {
            "train/learning_rate": self.trainer.optimizers[0].param_groups[0]['lr'],
            "train/labeled_loss": sup_loss,
            "train/unlabeled_loss": unsup_loss,
            "train/total_loss": total_loss,
            "train/confidence": weak_data_confidence.mean(),
            "train/threshold": self.threshold,
            "train/sample_use %": mask.float().mean(),
        }
        
        # Log class distribution info
        if self.prob_count > 0:
            for c in range(self.num_classes):
                metrics[f"train/class_{c}_distribution"] = (self.prob_sum / self.prob_count)[c]
        
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
        optimizer = torch.optim.SGD(self.parameters(), self.lr,
                            momentum=self.momentum,
                            weight_decay=self.weight_decay,
                            nesterov=self.nesterov)
        
        self.ramp_down_length = self.cfg["ramp_down_length"] * self.trainer.datamodule.steps_per_epochs
        scheduler = LambdaLR(optimizer, self.ramp_lr)
        
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "name": None,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
    
    def forward(self, input):
        # Forward pass same as in FixMatch
        features_maps = self.backbone.backbone.features(input)
        global_avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        flatten = torch.nn.Flatten()
        pooled_feat = flatten(global_avg_pool(features_maps))
        
        # Get the final logits from the backbone
        logits = self.backbone.classifier(pooled_feat)
        
        return logits

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    global cfg
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val/accuracy",
        save_top_k=3,
        mode="max",
        filename="freematch-{epoch:02d}-{val/accuracy:.4f}",
        save_last=True,
    )
    
    trainer = L.Trainer(
        max_steps=cfg["steps"], 
        max_epochs=cfg["epochs"],
        enable_checkpointing=True, 
        log_every_n_steps=cfg["val_check_interval"], 
        check_val_every_n_epoch=None, 
        val_check_interval=cfg["val_check_interval"],
        callbacks=[checkpoint_callback],
        benchmark=True,
    )

    trainer.fit(model=FreeMatch({
        'han_box': ['open', 'close']}, cfg), datamodule=data)

if __name__ == "__main__":
    main()