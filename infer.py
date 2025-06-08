
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

import cv2
import numpy as np
from methods.fixmatch import FixMatch
from dataset.Dataset import MultiTaskCabDataset
from torchvision.transforms import v2

trans = v2.Compose([
    v2.ToTensor(),
    v2.Resize([224, 224]),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

trans_view = v2.Compose([
    v2.ToTensor(),
    v2.Resize([224, 224]),
])

def open_image(img_path):
    return np.array(Image.open(img_path).convert('RGB'))

def open_and_plot_cam(img_path):
    # img = trans_view(data.data[pic_num])
    img = trans_view(open_image(img_path))
    input = trans(img).unsqueeze(0)
    cam, out_pred = a(input.to("cuda"))
    

    # Create a figure with 2 rows
    fig, ax = plt.subplots(3, 3, figsize=(12, 10))

    # First row - Original visualization
    # Original image
    ax[0, 0].imshow(img.permute(1, 2, 0))
    ax[0, 0].set_title('Original Image')
    ax[0, 0].axis('off')

    # CAM visualization for class 0 and 1
    ax[0, 1].imshow(cam[0, 0].detach().cpu())
    ax[0, 1].set_title('CAM - Class 0')
    ax[0, 1].axis('off')

    ax[0, 2].imshow(cam[0, 1].detach().cpu())
    ax[0, 2].set_title('CAM - Class 1')
    ax[0, 2].axis('off')

    # Second row - Overlay visualizations
    # Create normalized CAMs for overlays

    cam_0 = cv2.resize(cam[0, 0].detach().cpu().numpy(), dsize=[224, 224])
    cam_1 = cv2.resize( cam[0, 1].detach().cpu().numpy(), dsize=(224, 224))

    # Normalize CAMs to 0-1 range for visualization
    cam_0 = (cam_0 - cam_0.min()) / (cam_0.max() - cam_0.min())
    cam_1 = (cam_1 - cam_1.min()) / (cam_1.max() - cam_1.min())

    # Overlay CAM on original image
    original_img = img.permute(1, 2, 0).numpy()
    ax[1, 1].imshow(original_img)
    cam_0_heatmap = ax[1, 1].imshow(cam_0, cmap='jet', alpha=0.5)
    ax[1, 1].set_title('Class 0 CAM Overlay task 0')
    ax[1, 1].axis('off')

    ax[1, 2].imshow(original_img)
    cam_1_heatmap = ax[1, 2].imshow(cam_1, cmap='jet', alpha=0.5)
    ax[1, 2].set_title('Class 1 CAM Overlay task 1')
    ax[1, 2].axis('off')

    cam_2 = cv2.resize(cam[0, 2].detach().cpu().numpy(), dsize=[224, 224])
    cam_3 = cv2.resize( cam[0, 3].detach().cpu().numpy(), dsize=(224, 224))

    # Normalize CAMs to 0-1 range for visualization
    cam_2 = (cam_2 - cam_2.min()) / (cam_2.max() - cam_2.min())
    cam_3 = (cam_3 - cam_3.min()) / (cam_3.max() - cam_3.min())

    # Overlay CAM on original image
    original_img = img.permute(1, 2, 0).numpy()
    ax[2, 1].imshow(original_img)
    cam_2_heatmap = ax[2, 1].imshow(cam_2, cmap='jet', alpha=0.5)
    ax[2, 1].set_title('Class 0 CAM Overlay task 1')
    ax[2, 1].axis('off')

    ax[2, 2].imshow(original_img)
    cam_3_heatmap = ax[2, 2].imshow(cam_3, cmap='jet', alpha=0.5)
    ax[2, 2].set_title('Class 1 CAM Overlay task 1')
    ax[2, 2].axis('off')

    plt.tight_layout()
    plt.show()
    print(f"out_pred: {out_pred}")


a = FixMatch.load_from_checkpoint("/home/treerspeaking/src/python/cabdefect/lightning_logs/version_30/checkpoints/epoch=299-step=7500.ckpt")


data = MultiTaskCabDataset("data", split="test")

open_and_plot_cam("/home/treerspeaking/src/python/cabdefect/data/test_data/co_dinh_cap/Ảnh sai/53.jpg")
open_and_plot_cam("/home/treerspeaking/src/python/cabdefect/data/test_data/co_dinh_cap/Ảnh sai/55.jpg")
open_and_plot_cam("/home/treerspeaking/src/python/cabdefect/data/test_data/co_dinh_cap/Ảnh sai/56.jpg")
open_and_plot_cam("/home/treerspeaking/src/python/cabdefect/data/test_data/co_dinh_cap/Ảnh sai/57.jpg")
open_and_plot_cam("/home/treerspeaking/src/python/cabdefect/data/test_data/co_dinh_cap/Ảnh sai/58.jpg")
open_and_plot_cam("/home/treerspeaking/src/python/cabdefect/data/test_data/co_dinh_cap/Ảnh sai/59.jpg")
open_and_plot_cam("/home/treerspeaking/src/python/cabdefect/data/test_data/co_dinh_cap/Ảnh sai/60.jpg")
open_and_plot_cam("/home/treerspeaking/src/python/cabdefect/data/test_data/co_dinh_cap/Ảnh sai/61.jpg")
open_and_plot_cam("/home/treerspeaking/src/python/cabdefect/data/test_data/co_dinh_cap/Ảnh sai/62.jpg")