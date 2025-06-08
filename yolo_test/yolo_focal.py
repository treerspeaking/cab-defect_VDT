import torch
import torchvision.transforms as T

from ultralytics import YOLO
from ultralytics.data.dataset import ClassificationDataset
from ultralytics.models.yolo.classify import ClassificationTrainer

class CustomizedDataset(ClassificationDataset):
    def init(self, root: str, args, augment: bool = False, prefix: str = ""):
        super().init(root, args, augment, prefix)
        train_transforms = T.Compose(
            [
                T.Resize((args.imgsz, args.imgsz)),
                T.RandomHorizontalFlip(p=args.fliplr),
                T.RandomVerticalFlip(p=args.flipud),
                T.RandomAffine(args.degrees, [args.translate, args.translate], shear=args.shear),
                T.RandAugment(interpolation=T.InterpolationMode.BILINEAR),
                T.ColorJitter(brightness=args.hsv_v, contrast=args.hsv_v, saturation=args.hsv_s, hue=args.hsv_h),
                T.ToTensor(),
                T.Normalize(mean=torch.tensor(0), std=torch.tensor(1)),
                T.RandomErasing(p=args.erasing, scale=[0.02, 0.05],inplace=True),
            ]
        )
        val_transforms = T.Compose(
            [
                T.Resize((args.imgsz, args.imgsz)),
                T.ToTensor(),
                T.Normalize(mean=torch.tensor(0), std=torch.tensor(1)),
            ]
        )
        self.torch_transforms = train_transforms if augment else val_transforms

class CustomizedTrainer(ClassificationTrainer):
    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        return CustomizedDataset(root=img_path, args=self.args, augment=mode == "train", prefix=mode)

model = YOLO("yolo11l-cls.pt")  # load a pretrained model (recommended for training)

results = model.train(
    data="/home/treerspeaking/src/python/cabdefect/yolo_test/yolo_data", 
    trainer=CustomizedTrainer, 
    epochs=1000, 
    imgsz=512,
    cfg="/home/treerspeaking/src/python/cabdefect/yolo_test/aug.yaml"
    )
