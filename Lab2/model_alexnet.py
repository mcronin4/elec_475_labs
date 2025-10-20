import torch
import torch.nn as nn
import torchvision.models as models


class AlexNetNose(nn.Module):
    """
    AlexNet architecture adapted for pet nose localization.
    Uses pretrained ImageNet weights and replaces final layer for regression.
    
    Input: (batch_size, 3, 227, 227)
    Output: (batch_size, 2) - (u, v) coordinates of nose location
    """
    def __init__(self, pretrained=True):
        super(AlexNetNose, self).__init__()
        
        # Load pretrained AlexNet
        alexnet = models.alexnet(pretrained=pretrained)
        
        # Extract features and classifier
        self.features = alexnet.features
        self.avgpool = alexnet.avgpool
        self.classifier = nn.Sequential(*list(alexnet.classifier.children())[:-1])
        
        # Replace final layer: was Linear(4096, 1000) for ImageNet classification
        # Now Linear(4096, 2) for nose coordinate regression
        self.regressor = nn.Linear(4096, 2)
        
        # Initialize final layer with small weights and bias at image center (113.5, 113.5)
        nn.init.kaiming_normal_(self.regressor.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.regressor.bias, 113.5)

        # Store input shape for reference
        self.input_shape = (3, 227, 227)
        
        if pretrained:
            print("Loaded pretrained AlexNet weights from ImageNet")
            print("Replaced final classification layer with regression head (4096 -> 2)")
    
    def forward(self, x):
        # Feature extraction
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Classifier layers (all but final)
        x = self.classifier(x)
        
        # Regression head
        x = self.regressor(x)
        
        return x


if __name__ == "__main__":
    # Test the model
    print("Testing AlexNetNose...")
    
    # Create model
    model = AlexNetNose(pretrained=False)  # Don't download weights for test
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel parameters: {num_params:,}")
    print(f"Trainable parameters: {num_trainable:,}")
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 227, 227)
    output = model(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: (4, 2)")
    
    assert output.shape == (4, 2), f"Output shape mismatch: {output.shape}"
    print("\n✓ Model test passed!")



