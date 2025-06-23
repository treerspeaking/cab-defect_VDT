import gradio as gr
import torch
import numpy as np
from PIL import Image

def open_image(img_path):
    return np.array(Image.open(img_path).convert("RGB"))

from methods.supervised import SupervisedModel

# model_used = SupervisedModel.load_from_checkpoint("/home/treerspeaking/src/python/cabdefect/lightning_logs/goat_2_supervised/checkpoints/supervisedepoch=674-step=43875-val/accuracy=0.8313-val/total_loss=0.49993109703063965.ckpt")
model_used = SupervisedModel.load_from_checkpoint("/home/treerspeaking/src/python/cabdefect/lightning_logs/version_288/checkpoints/supervisedepoch=411-step=26780-val/accuracy=0.8675-val/total_loss=0.7918075323104858.ckpt")
# model_used = SupervisedModel.load_from_checkpoint("/home/treerspeaking/src/python/cabdefect/lightning_logs/version_291/checkpoints/supervisedepoch=634-step=41275-val/accuracy=0.8675-val/total_loss=1.2008382081985474.ckpt")
# ye this is just it gg


from torchvision.transforms import v2
trans = v2.Compose([
    v2.ToTensor(),
    # v2.Resize([640, 720]),
    v2.Resize([512, 512]),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# trans_view = v2.Compose([
#     v2.ToTensor(),
#     v2.Resize([512, 512]),
# ])

# from dataset.Dataset import MultiTaskCabDataset
# from dataset.BaseDataset import BasicLabelDataset

# data = MultiTaskCabDataset("/home/treerspeaking/src/python/cabdefect/data_more_label", split="test", test_transforms=trans)

task_dict = {'co_dinh_cap': ['co_dinh_dung', 'co_dinh_sai'], 'bo_chia': ['bo_chia_loi', 'bo_chia_dung'], 'han_box': ['close', 'open']}
print(task_dict)
# label_data = BasicLabelDataset(data.data, data.targets, trans)

from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.data import DataLoader

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

import cv2

def predict_and_plot_cam(image):
    # Convert PIL image to tensor and preprocess
    img_tensor = trans(image).unsqueeze(0)  # Add batch dimension
    label_dummy = torch.zeros((1, 3), dtype=torch.long)  # Dummy label, adjust shape as needed
    img_tensor = img_tensor.to("cuda")
    label_dummy = label_dummy.to("cuda")
    # Get CAMs and predictions
    cams, out_preds = model_used((img_tensor, label_dummy))
    # Plot and return the figure as an image
    batch_size = img_tensor.shape[0]
    num_tasks = cams.shape[1] // 2
    img = img_tensor[0]
    cam = cams[0]
    original_img = img.permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    original_img = std * original_img + mean
    original_img = np.clip(original_img, 0, 1)
    fig, ax = plt.subplots(num_tasks, 4, figsize=(20, 5 * (num_tasks + 1)))
    if num_tasks == 1:
        ax = np.expand_dims(ax, 0)
    for task_idx, (k, v) in enumerate(task_dict.items()):
        ax[task_idx, 0].imshow(original_img)
        ax[task_idx, 0].set_title('Original Image')
        ax[task_idx, 0].axis('off')
        for class_idx in range(2):
            cam_map = cam[task_idx * 2 + class_idx].detach().cpu().numpy()
            cam_map = cv2.resize(cam_map, dsize=(512, 512))
            cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min() + 1e-8)
            ax[task_idx, class_idx + 1].imshow(original_img)
            ax[task_idx, class_idx + 1].imshow(cam_map, cmap='jet', alpha=0.5)
            ax[task_idx, class_idx + 1].set_title(f'Task {k} - Class {v[class_idx]} - prob {out_preds[task_idx]["probabilities"][0][class_idx]:.2f}')
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
    gr.Markdown("Upload an image to see the CAM map predictions.")
    
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