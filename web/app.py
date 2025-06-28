import gradio as gr
import torch
import numpy as np
from PIL import Image

def open_image(img_path):
    return np.array(Image.open(img_path).convert("RGB"))

from methods.supervised import SupervisedModel

# Load three different models, one for each task
# model_co_dinh_cap = SupervisedModel.load_from_checkpoint("/home/treerspeaking/src/python/cabdefect/lightning_logs/version_288/checkpoints/supervisedepoch=411-step=26780-val/accuracy=0.8675-val/total_loss=0.7918075323104858.ckpt")
# model_bo_chia = SupervisedModel.load_from_checkpoint("/home/treerspeaking/src/python/cabdefect/lightning_logs/version_288/checkpoints/supervisedepoch=411-step=26780-val/accuracy=0.8675-val/total_loss=0.7918075323104858.ckpt")
model_han_box = SupervisedModel.load_from_checkpoint("/home/treerspeaking/src/python/cabdefect/lightning_logs/version_288/checkpoints/supervisedepoch=411-step=26780-val/accuracy=0.8675-val/total_loss=0.7918075323104858.ckpt")
model_co_dinh_cap = SupervisedModel.load_from_checkpoint("/home/treerspeaking/src/python/cabdefect/lightning_logs/goat_co_dinh/checkpoints/supervisedepoch=496-val/accuracy=0.8000.ckpt")
model_bo_chia = SupervisedModel.load_from_checkpoint("/home/treerspeaking/src/python/cabdefect/lightning_logs/goat_bo_chia_2/checkpoints/supervisedepoch=349-step=6300-val/accuracy=1.0000-val/total_loss=0.014687540009617805.ckpt")
# model_han_box = SupervisedModel.load_from_checkpoint("/home/treerspeaking/src/python/cabdefect/lightning_logs/goat_han/checkpoints/supervisedepoch=347-val/accuracy=0.9200.ckpt")

# Map tasks to models
task_models = {
    'co_dinh_cap': model_co_dinh_cap,
    'bo_chia': model_bo_chia,
    'han_box': model_han_box
}

from torchvision.transforms import v2
trans = v2.Compose([
    v2.ToTensor(),
    v2.Resize([512, 512]),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

task_map = {
    'co_dinh_cap': 0,
    'bo_chia': 0,
    'han_box': 2,
}

task_dict = {
    'han_box': ['close', 'open'],
    'co_dinh_cap': ['co_dinh_dung', 'co_dinh_sai'], 
    'bo_chia': ['bo_chia_loi', 'bo_chia_dung'], 
    }
print(task_dict)

import matplotlib.pyplot as plt
import cv2

def predict_and_plot_cam(image):
    # Convert PIL image to tensor and preprocess
    img_tensor = trans(image).unsqueeze(0)  # Add batch dimension
    label_dummy = torch.zeros((1, 3), dtype=torch.long)  # Dummy label
    img_tensor = img_tensor.to("cuda")
    label_dummy = label_dummy.to("cuda")
    
    # Results containers
    all_cams = []
    all_preds = []
    
    # Process each task with its dedicated model
    for task_idx, (task_name, _) in enumerate(task_dict.items()):
        model = task_models[task_name]
        cams, out_preds = model((img_tensor, label_dummy))
        all_cams.append(cams)
        all_preds.append(out_preds)
    
    # Prepare for visualization
    num_tasks = len(task_dict)
    img = img_tensor[0]
    
    original_img = img.permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    original_img = std * original_img + mean
    original_img = np.clip(original_img, 0, 1)
    
    fig, ax = plt.subplots(num_tasks, 4, figsize=(20, 5 * (num_tasks + 1)))
    if num_tasks == 1:
        ax = np.expand_dims(ax, 0)
    
    for task_idx, (task_name, classes) in enumerate(task_dict.items()):
        cam = all_cams[task_idx][0]  # Get CAM for this task
        out_pred = all_preds[task_idx][task_map[task_name]]  # Get prediction for this task
        
        ax[task_idx, 0].imshow(original_img)
        ax[task_idx, 0].set_title(f'Original Image - Task: {task_name}')
        ax[task_idx, 0].axis('off')
        
        for class_idx in range(2):
            cam_map = cam[task_map[task_name] * 2 + class_idx].detach().cpu().numpy()
            cam_map = cv2.resize(cam_map, dsize=(512, 512))
            cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min() + 1e-8)
            
            ax[task_idx, class_idx + 1].imshow(original_img)
            ax[task_idx, class_idx + 1].imshow(cam_map, cmap='jet', alpha=0.5)
            ax[task_idx, class_idx + 1].set_title(f'Task {task_name} - Class {classes[class_idx]} - prob {out_pred["probabilities"][0][class_idx]:.2f}')
            ax[task_idx, class_idx + 1].axis('off')
        
        cam1 = cam[task_map[task_name] * 2].detach().cpu().numpy()
        cam2 = cam[task_map[task_name] * 2 + 1].detach().cpu().numpy()
        decision_cam_map = cam1 + cam2
        cam_map = cv2.resize(decision_cam_map, dsize=(512, 512))
        cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min() + 1e-8)
        
        ax[task_idx, 3].imshow(original_img)
        ax[task_idx, 3].imshow(cam_map, cmap='jet', alpha=0.5)
        ax[task_idx, 3].set_title(f'Combine activation - {task_name}')
        ax[task_idx, 3].axis('off')
    
    plt.tight_layout()
    # Save figure to a buffer
    import io
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

# Gradio interface with custom layout
with gr.Blocks(title="CabDefect CAM Visualizer") as iface:
    gr.Markdown("# CabDefect CAM Visualizer")
    gr.Markdown("Upload an image to see the CAM map predictions from three different models.")
    
    with gr.Column():
        input_image = gr.Image(type="pil", label="Upload Image")
        submit_btn = gr.Button("Analyze Image", variant="primary")
        output_image = gr.Image(type="pil", label="CAM Visualization")
    
    submit_btn.click(
        fn=predict_and_plot_cam,
        inputs=input_image,
        outputs=output_image
    )

if __name__ == "__main__":
    iface.launch()