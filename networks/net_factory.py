import torch

from .MobileNetV3 import ModifiedMobileNetV3
from .ResNet50 import ModifiedResNet50

def net_factory(network, in_channels=None, num_classes=None, pretrained=None):
    if network == "MobileNetV3Feature":
        return ModifiedMobileNetV3(model_type="large", pretrained=pretrained, remove_layers=["classifier"])
    
    if network == "ResNet50Feature":
        return ModifiedResNet50(pretrained=pretrained, remove_layers=["fc"])
    
    if network == "MobileNetV3":
        # return ModifiedMobileNetV3(model_type="large", pretrained=pretrained, num_classes=num_classes)
        return ModifiedMobileNetV3(model_type="large", pretrained=pretrained, remove_layers=["classifier"])
        # im gonna regret this later
    
    if network == "ResNet50":
        return ModifiedResNet50(pretrained=pretrained, num_classes=num_classes)
    
