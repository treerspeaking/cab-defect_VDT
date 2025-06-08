import torch
import torch.nn as nn
from torchvision import models
from collections import OrderedDict

class ModifiedMobileNetV3(nn.Module):
    """
    Modified MobileNetV3 with customizable layer removal
    """
    def __init__(self, model_type='large', pretrained=True, remove_layers=None, num_classes = 1000):
        super().__init__()
        
        # Load pretrained MobileNetV3
        if model_type.lower() == 'large':
            self.backbone = models.mobilenet_v3_large(pretrained=pretrained)
            if num_classes != 1000:
                self.backbone.classifier[3] = torch.nn.Linear(1280, num_classes)
        elif model_type.lower() == 'small':
            self.backbone = models.mobilenet_v3_small(pretrained=pretrained)
            if num_classes != 1000:
                self.backbone.classifier[3] = torch.nn.Linear(1024, num_classes)
        else:
            raise ValueError("model_type must be 'large' or 'small'")
        
        
        # Remove specified layers
        if remove_layers:
            self._remove_layers(remove_layers)
    
    def _remove_layers(self, layers_to_remove):
        """
        Remove specified layers from the model
        layers_to_remove: list of layer names or 'classifier' for final layers
        """
        if 'classifier' in layers_to_remove:
            # Remove the final classifier
            self.backbone.classifier = nn.Identity()
        
        if 'avgpool' in layers_to_remove:
            # Remove global average pooling
            self.backbone.avgpool = nn.Identity()
    
    def forward(self, x):
        return self.backbone(x)

# Example usage functions
def example_basic_loading():
    """Example: Basic loading with pretrained weights"""
    # Load pretrained MobileNetV3-Large without classifier
    model = ModifiedMobileNetV3(
        model_type='large',
        # remove_layers=['classifier', 'avgpool']
    )
    
    # Test forward pass
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        features = model(x)
    print(f"Output shape: {features.shape}")
    for name, module in model.backbone.named_modules():
        print(f"{name}: {module}")
        
    return model


if __name__ == "__main__":
    # Example usage
    print("Loading MobileNetV3 with modifications...")
    
    # Basic example
    model = example_basic_loading()