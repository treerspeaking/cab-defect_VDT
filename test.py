

import numpy as np
from PIL import Image

def open_image(img_path):
    return np.array(Image.open(img_path).convert("RGB"))

from methods.supervised import SupervisedModel

# model_used = SupervisedModel.load_from_checkpoint("/home/treerspeaking/src/python/cabdefect/lightning_logs/version_183/checkpoints/supervisedepoch=108-val/accuracy=1.0000-val/total_loss=0.023907367140054703.ckpt")
model_used = SupervisedModel.load_from_checkpoint("/home/treerspeaking/src/python/cabdefect/lightning_logs/version_248/checkpoints/supervisedepoch=99-val/accuracy=1.0000-val/total_loss=0.01311619020998478.ckpt")

from torchvision.transforms import v2
trans = v2.Compose([
    v2.ToTensor(),
    v2.Resize([224, 224]),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# trans_view = v2.Compose([
#     v2.ToTensor(),
#     v2.Resize([512, 512]),
# ])

from dataset.Dataset import MultiTaskCabDataset
from dataset.BaseDataset import BasicLabelDataset

data = MultiTaskCabDataset("/home/treerspeaking/src/python/cabdefect/data_more_label", split="test", test_transforms=trans)
label_data = BasicLabelDataset(data.data, data.targets, trans)

from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.data import DataLoader

data_loader = DataLoader(
            label_data,
            batch_size=5,
            shuffle=False,
            num_workers=1,
            pin_memory=True
        )

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

import cv2

def open_and_plot_cam(inputs, model):
    # # img = trans_view(data.data[pic_num])
    img, labels = inputs
    # input = img.unsqueeze(0)
    # cam, out_pred = model(input.to("cuda"))
    
    cam, out_pred = model((img.to("cuda"), labels))
    

    # Create a figure with 2 rows
    fig, ax = plt.subplots(2, 3, figsize=(12, 10))

    # First row - Original visualization
    # Original image
    # ax[0, 0].imshow(img.permute(1, 2, 0))
    # ax[0, 0].set_title('Original Image')
    # ax[0, 0].axis('off')

    # # CAM visualization for class 0 and 1
    # ax[0, 1].imshow(cam[0, 0].detach().cpu())
    # ax[0, 1].set_title('CAM - Class 0')
    # ax[0, 1].axis('off')

    # ax[0, 2].imshow(cam[0, 1].detach().cpu())
    # ax[0, 2].set_title('CAM - Class 1')
    # ax[0, 2].axis('off')

    # # Second row - Overlay visualizations
    # # Create normalized CAMs for overlays

    # cam_0 = cv2.resize(cam[0, 0].detach().cpu().numpy(), dsize=[512, 512])
    # cam_1 = cv2.resize( cam[0, 1].detach().cpu().numpy(), dsize=(512, 512))

    # # Normalize CAMs to 0-1 range for visualization
    # cam_0 = (cam_0 - cam_0.min()) / (cam_0.max() - cam_0.min())
    # cam_1 = (cam_1 - cam_1.min()) / (cam_1.max() - cam_1.min())

    # # Overlay CAM on original image
    # original_img = img.permute(1, 2, 0).numpy()
    # ax[1, 1].imshow(original_img)
    # cam_0_heatmap = ax[1, 1].imshow(cam_0, cmap='jet', alpha=0.5)
    # ax[1, 1].set_title('Class 0 CAM Overlay task 0')
    # ax[1, 1].axis('off')

    # ax[1, 2].imshow(original_img)
    # cam_1_heatmap = ax[1, 2].imshow(cam_1, cmap='jet', alpha=0.5)
    # ax[1, 2].set_title('Class 1 CAM Overlay task 1')
    # ax[1, 2].axis('off')

    # cam_2 = cv2.resize(cam[0, 2].detach().cpu().numpy(), dsize=[224, 224])
    # cam_3 = cv2.resize( cam[0, 3].detach().cpu().numpy(), dsize=(224, 224))

    # # Normalize CAMs to 0-1 range for visualization
    # cam_2 = (cam_2 - cam_2.min()) / (cam_2.max() - cam_2.min())
    # cam_3 = (cam_3 - cam_3.min()) / (cam_3.max() - cam_3.min())

    # # Overlay CAM on original image
    # original_img = img.permute(1, 2, 0).numpy()
    # ax[2, 1].imshow(original_img)
    # cam_2_heatmap = ax[2, 1].imshow(cam_2, cmap='jet', alpha=0.5)
    # ax[2, 1].set_title('Class 0 CAM Overlay task 1')
    # ax[2, 1].axis('off')

    # ax[2, 2].imshow(original_img)
    # cam_3_heatmap = ax[2, 2].imshow(cam_3, cmap='jet', alpha=0.5)
    # ax[2, 2].set_title('Class 1 CAM Overlay task 1')
    # ax[2, 2].axis('off')

    # plt.tight_layout()
    # plt.show()
    print(f"out_pred: {out_pred}")
    # for i in out_pred[0].items():
    #     print(i["out_pred"])
    
for i in data_loader:
    # inputs, labels = i
    open_and_plot_cam(i, model_used)