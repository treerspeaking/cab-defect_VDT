import torch
import torch.nn as nn
from torchvision import models

class ModifiedResNet50(nn.Module):
    """
    Modified ResNet50 with customizable layer removal
    """
    def __init__(self, pretrained=True, remove_layers=None, num_classes=1000):
        super().__init__()
        
        # Load pretrained ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        if num_classes != 1000:
                self.backbone.fc = torch.nn.Linear(2048, num_classes)
        
        # Remove specified layers
        if remove_layers:
            self._remove_layers(remove_layers)
    
    def _remove_layers(self, layers_to_remove):
        """
        Remove specified layers from the model
        layers_to_remove: list of layer names or 'fc' for final fully connected layer
        """
        if 'fc' in layers_to_remove:
            # Remove the final fully connected layer
            self.backbone.fc = nn.Identity()
        
        if 'avgpool' in layers_to_remove:
            # Remove global average pooling
            self.backbone.avgpool = nn.Identity()
    
    def forward(self, x):
        return self.backbone(x)
    
    def features(self, x):
        # Get features before the final fully connected layer
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        # x = self.backbone.avgpool(x)
        # x = torch.flatten(x, 1)
        
        return x

# Example usage functions
def example_basic_loading():
    """Example: Basic loading with pretrained weights"""
    # Load pretrained ResNet50 without fc layer
    model = ModifiedResNet50(
        pretrained=True,
        # remove_layers=['fc', 'avgpool']
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
    print("Loading ResNet50 with modifications...")
    
    # Basic example
    model = example_basic_loading()