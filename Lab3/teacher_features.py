#########################################################################################################
#
#   ELEC 475 - Lab 3: Semantic Segmentation
#   Fall 2025
#
#   teacher_features.py - Extract intermediate features from FCN-ResNet50 teacher model
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.segmentation as segmentation_models


def extract_teacher_features(teacher_model, x):
    """
    Extract intermediate features from FCN-ResNet50 teacher model.
    
    Args:
        teacher_model: FCN-ResNet50 model
        x: [B, 3, H, W] input images
        
    Returns:
        dict: Dictionary with features at different stride levels
            - 'stride4': [B, 256, H/4, W/4] from layer1
            - 'stride8': [B, 512, H/8, W/8] from layer2
            - 'stride16': [B, 1024, H/16, W/16] from layer3
    """
    features = {}
    
    # Get backbone
    backbone = teacher_model.backbone
    
    # Forward through initial layers
    x = backbone.conv1(x)
    x = backbone.bn1(x)
    x = backbone.relu(x)
    x = backbone.maxpool(x)  # Now at stride 4
    
    # Extract features from ResNet stages
    # Layer1: stride 4
    x = backbone.layer1(x)
    features['stride4'] = x  # [B, 256, H/4, W/4]
    
    # Layer2: stride 8
    x = backbone.layer2(x)
    features['stride8'] = x  # [B, 512, H/8, W/8]
    
    # Layer3: stride 16
    x = backbone.layer3(x)
    features['stride16'] = x  # [B, 1024, H/16, W/16]
    
    return features


def load_fcn_resnet50_with_features(pretrained=True, device='cpu'):
    """
    Load FCN-ResNet50 model for feature extraction.
    
    Args:
        pretrained (bool): Whether to load pretrained weights
        device (str): Device to load model on
        
    Returns:
        nn.Module: FCN-ResNet50 model
    """
    model = segmentation_models.fcn_resnet50(pretrained=pretrained)
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    return model


if __name__ == "__main__":
    """Test feature extraction."""
    print("=" * 80)
    print("Testing Teacher Feature Extraction")
    print("=" * 80)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher = load_fcn_resnet50_with_features(pretrained=False, device=device)
    
    # Test input
    batch_size = 2
    height, width = 320, 320
    dummy_input = torch.randn(batch_size, 3, height, width).to(device)
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Extract features
    with torch.no_grad():
        features = extract_teacher_features(teacher, dummy_input)
    
    print(f"\nExtracted features:")
    for stride, feat in features.items():
        print(f"  {stride}: {feat.shape}")
    
    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)

