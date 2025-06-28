import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from methods.supervised import SupervisedModel
from dataset.Dataset import MultiTaskCabDataset
from dataset.BaseDataset import BasicLabelDataset

def parse_args():
    parser = argparse.ArgumentParser(description='Generate and save CAM visualizations')
    parser.add_argument('--model_path', type=str, default="/home/treerspeaking/src/python/cabdefect/lightning_logs/version_288/checkpoints/supervisedepoch=411-step=26780-val/accuracy=0.8675-val/total_loss=0.7918075323104858.ckpt",
                        help='Path to the model checkpoint')
    parser.add_argument('--data_dir', type=str, default="/home/treerspeaking/src/python/cabdefect/data_more_label",
                        help='Directory containing the dataset')
    parser.add_argument('--output_dir', type=str, default="./cam_output",
                        help='Directory to save visualizations')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to process')
    return parser.parse_args()

def open_image(img_path):
    return np.array(Image.open(img_path).convert("RGB"))

def create_data_loader(data_dir, transform, batch_size):
    data = MultiTaskCabDataset(data_dir, split="test", test_transforms=transform)
    label_data = BasicLabelDataset(data.data, data.targets, transform)
    return DataLoader(label_data, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

def generate_and_save_cam(imgs, labels, model, output_dir, batch_idx, task_dict):
    os.makedirs(output_dir, exist_ok=True)
    
    cams, out_preds = model((imgs.to("cuda"), labels))
    batch_size = imgs.shape[0]
    num_tasks = cams.shape[1] // 2  # Assuming 2 CAMs per task

    for idx in range(batch_size):
        img = imgs[idx]
        cam = cams[idx]  # shape: [num_tasks*2, H, W]
        
        # Undo normalization for display
        original_img = img.permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        original_img = std * original_img + mean
        original_img = np.clip(original_img, 0, 1)

        # Create a visualization figure
        fig, ax = plt.subplots(num_tasks, 4, figsize=(20, 5 * (num_tasks + 1)))
        if num_tasks == 1:
            ax = np.expand_dims(ax, 0)  # Ensure ax is 2D

        sample_filename = f"sample_{batch_idx}_{idx}"

        for task_idx, (k, v) in enumerate(task_dict.items()):
            # First column: original image
            ax[task_idx, 0].imshow(original_img)
            ax[task_idx, 0].set_title('Original Image')
            ax[task_idx, 0].axis('off')

            # Next columns: CAM overlays
            for class_idx in range(2):
                cam_map = cam[task_idx * 2 + class_idx].detach().cpu().numpy()
                cam_map = cv2.resize(cam_map, dsize=(512, 512))
                cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min() + 1e-8)
                ax[task_idx, class_idx + 1].imshow(original_img)
                ax[task_idx, class_idx + 1].imshow(cam_map, cmap='jet', alpha=0.5)
                ax[task_idx, class_idx + 1].set_title(f'Task {k} - CAM Overlay - Class {v[class_idx]}')
                ax[task_idx, class_idx + 1].axis('off')
            
            cam1 = cam[task_idx * 2 + 0].detach().cpu().numpy()
            cam2 = cam[task_idx * 2 + 1].detach().cpu().numpy()
            decision_cam_map = cam1 + cam2

            cam_map = cv2.resize(decision_cam_map, dsize=(512, 512))
            cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min() + 1e-8)
            ax[task_idx, 3].imshow(original_img)
            ax[task_idx, 3].imshow(cam_map, cmap='jet', alpha=0.5)
            ax[task_idx, 3].set_title(f'Combine activation')
            ax[task_idx, 3].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{sample_filename}_visualization.png"), dpi=300)
        plt.close(fig)

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model_used = SupervisedModel.load_from_checkpoint(args.model_path)
    model_used.eval()
    model_used.to("cuda")
    
    # Define transforms
    trans = v2.Compose([
        v2.ToTensor(),
        v2.Resize([512, 512]),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Define task dictionary
    task_dict = {
        'co_dinh_cap': ['co_dinh_dung', 'co_dinh_sai'], 
        'bo_chia': ['bo_chia_loi', 'bo_chia_dung'], 
        'han_box': ['close', 'open']
    }
    
    # Create data loader
    data_loader = create_data_loader(args.data_dir, trans, args.batch_size)
    
    # Process samples
    print(f"Processing {args.num_samples} samples and saving to {args.output_dir}")
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= args.num_samples:
                break
                
            imgs, labels = batch
            generate_and_save_cam(imgs, labels, model_used, args.output_dir, batch_idx, task_dict)
            print(f"Processed batch {batch_idx+1}/{args.num_samples}")

if __name__ == "__main__":
    main()