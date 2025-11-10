#########################################################################################################
#
#   ELEC 475 - Lab 3: Semantic Segmentation
#   Fall 2025
#
#   augmentation.py - Data augmentation transforms using Albumentations
#

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


def get_training_augmentation(input_size=(320, 320)):
    """
    Get training augmentation pipeline.
    
    Includes:
        - Geometric augmentations: flip, scale, crop, rotate
        - Photometric augmentations: color jitter, blur
        - Normalization with ImageNet stats
    
    Args:
        input_size (tuple): Target (height, width) for training
        
    Returns:
        albumentations.Compose: Augmentation pipeline
    """
    height, width = input_size
    
    transform = A.Compose([
        # Geometric augmentations
        A.HorizontalFlip(p=0.5),
        
        # Random scale then crop to fixed size
        # Scale limits: 0.5x to 2x
        A.RandomScale(scale_limit=(-0.5, 1.0), p=0.8),
        A.PadIfNeeded(
            min_height=height,
            min_width=width,
            border_mode=0,  # cv2.BORDER_CONSTANT
            value=0,
            mask_value=255,  # Pad masks with ignore_index
            p=1.0
        ),
        A.RandomCrop(height=height, width=width, p=1.0),
        
        # Random rotation
        A.Rotate(
            limit=10,  # -10 to +10 degrees
            border_mode=0,  # cv2.BORDER_CONSTANT
            value=0,
            mask_value=255,  # Use ignore_index for padded areas
            p=0.3
        ),
        
        # Photometric augmentations
        A.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1,
            p=0.5
        ),
        
        A.GaussianBlur(
            blur_limit=(3, 7),
            p=0.1
        ),
        
        # Normalize with ImageNet statistics
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        
        # Convert to PyTorch tensors
        ToTensorV2()
    ])
    
    return transform


def get_validation_augmentation(input_size=(320, 320)):
    """
    Get validation augmentation pipeline.
    
    Only resizes and normalizes (no data augmentation).
    
    Args:
        input_size (tuple): Target (height, width) for validation
        
    Returns:
        albumentations.Compose: Augmentation pipeline
    """
    height, width = input_size
    
    transform = A.Compose([
        # Resize to fixed size
        A.Resize(height=height, width=width, p=1.0),
        
        # Normalize with ImageNet statistics
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        
        # Convert to PyTorch tensors
        ToTensorV2()
    ])
    
    return transform


def get_test_augmentation():
    """
    Get test augmentation pipeline.
    
    Only normalizes (preserves original image size).
    
    Returns:
        albumentations.Compose: Augmentation pipeline
    """
    transform = A.Compose([
        # Normalize with ImageNet statistics
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        
        # Convert to PyTorch tensors
        ToTensorV2()
    ])
    
    return transform


if __name__ == "__main__":
    """Test augmentation pipelines."""
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    
    print("=" * 80)
    print("Testing Augmentation Pipelines")
    print("=" * 80)
    
    # Create dummy image and mask
    height, width = 400, 500
    dummy_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    dummy_mask = np.random.randint(0, 21, (height, width), dtype=np.uint8)
    
    print(f"\nOriginal image shape: {dummy_image.shape}")
    print(f"Original mask shape: {dummy_mask.shape}")
    
    # Test training augmentation
    print("\n" + "-" * 80)
    print("Testing Training Augmentation")
    print("-" * 80)
    
    train_transform = get_training_augmentation(input_size=(320, 320))
    
    for i in range(3):
        augmented = train_transform(image=dummy_image, mask=dummy_mask)
        aug_image = augmented['image']
        aug_mask = augmented['mask']
        
        print(f"\nAugmentation {i+1}:")
        print(f"  Image shape: {aug_image.shape} (should be [3, 320, 320])")
        print(f"  Mask shape: {aug_mask.shape} (should be [320, 320])")
        print(f"  Image dtype: {aug_image.dtype} (should be torch.float32)")
        print(f"  Mask dtype: {aug_mask.dtype} (should be torch.int64)")
        print(f"  Image range: [{aug_image.min():.3f}, {aug_image.max():.3f}]")
        print(f"  Unique mask values: {len(aug_mask.unique())} classes")
    
    # Test validation augmentation
    print("\n" + "-" * 80)
    print("Testing Validation Augmentation")
    print("-" * 80)
    
    val_transform = get_validation_augmentation(input_size=(320, 320))
    augmented = val_transform(image=dummy_image, mask=dummy_mask)
    aug_image = augmented['image']
    aug_mask = augmented['mask']
    
    print(f"\nValidation transform:")
    print(f"  Image shape: {aug_image.shape} (should be [3, 320, 320])")
    print(f"  Mask shape: {aug_mask.shape} (should be [320, 320])")
    
    # Test test augmentation (preserves size)
    print("\n" + "-" * 80)
    print("Testing Test Augmentation")
    print("-" * 80)
    
    test_transform = get_test_augmentation()
    augmented = test_transform(image=dummy_image, mask=dummy_mask)
    aug_image = augmented['image']
    aug_mask = augmented['mask']
    
    print(f"\nTest transform:")
    print(f"  Image shape: {aug_image.shape} (should be [3, {height}, {width}])")
    print(f"  Mask shape: {aug_mask.shape} (should be [{height}, {width}])")
    
    print("\n" + "=" * 80)
    print("Augmentation test complete!")
    print("=" * 80)




