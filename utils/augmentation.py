from torchvision.transforms import v2
import torch

class MyRandAugment(v2.RandAugment):
    _AUGMENTATION_SPACE = {
        "Identity": (lambda num_bins, height, width: None, False),
        "ShearX": (lambda num_bins, height, width: torch.linspace(0.0, 0.3, num_bins), True),
        "ShearY": (lambda num_bins, height, width: torch.linspace(0.0, 0.3, num_bins), True),
        "TranslateX": (
            lambda num_bins, height, width: torch.linspace(0.0, 150.0 / 331.0 * width, num_bins),
            True,
        ),
        "TranslateY": (
            lambda num_bins, height, width: torch.linspace(0.0, 150.0 / 331.0 * height, num_bins),
            True,
        ),
        # "Rotate": (lambda num_bins, height, width: torch.linspace(0.0, 30.0, num_bins), True),
        # "Brightness": (lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins), True),
        # "Color": (lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins), True),
        # "Contrast": (lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins), True),
        # "Sharpness": (lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins), True),
        # "Posterize": (
        #     lambda num_bins, height, width: (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4))).round().int(),
        #     False,
        # ),
        # "Solarize": (lambda num_bins, height, width: torch.linspace(1.0, 0.0, num_bins), False),
        # "AutoContrast": (lambda num_bins, height, width: None, False),
        # "Equalize": (lambda num_bins, height, width: None, False),
    }
