#########################################################################################################
#
#   ELEC 475 - Lab 3: Semantic Segmentation
#   Fall 2025
#
#   dataset.py - PASCAL VOC 2012 dataset handling for semantic segmentation
#

import torch
from torch.utils.data import Dataset, Subset
import torchvision.transforms as transforms
from torchvision.datasets import VOCSegmentation
from PIL import Image
import numpy as np


# PASCAL VOC 2012 class names (21 classes: background + 20 object categories)
VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
    'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
    'sofa', 'train', 'tvmonitor'
]

# PASCAL VOC colormap for visualization
VOC_COLORMAP = [
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    [0, 64, 128]
]


def get_voc_transforms(augment=False):
    """
    Get transform pipelines for PASCAL VOC segmentation.
    
    For FCN-ResNet50, we need to normalize images with ImageNet statistics.
    Note: We don't resize images - FCN can handle variable input sizes.
    
    Args:
        augment (bool): Whether to apply data augmentation (not used for evaluation)
        
    Returns:
        tuple: (image_transform, target_transform)
    """
    # Image transforms - normalize with ImageNet stats
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Target (mask) transform - convert PIL Image to tensor
    target_transform = transforms.Lambda(lambda x: torch.as_tensor(np.array(x), dtype=torch.long))
    
    return image_transform, target_transform


class VOCSegmentationDataset(Dataset):
    """
    Wrapper around torchvision's VOCSegmentation dataset.
    
    This wrapper provides:
    - Automatic dataset download
    - Proper transforms for FCN-ResNet50
    - Easy access to class information
    """
    
    def __init__(self, root='./data', year='2012', image_set='val', 
                 download=True, transform=None, target_transform=None):
        """
        Initialize PASCAL VOC segmentation dataset.
        
        Args:
            root (str): Root directory for dataset
            year (str): Dataset year ('2012', '2011', '2010', '2009', '2008', '2007')
            image_set (str): 'train', 'val', or 'trainval'
            download (bool): Whether to download dataset if not found
            transform (callable): Transform for images
            target_transform (callable): Transform for segmentation masks
        """
        # If no transforms provided, use default ones
        if transform is None or target_transform is None:
            default_img_transform, default_target_transform = get_voc_transforms()
            if transform is None:
                transform = default_img_transform
            if target_transform is None:
                target_transform = default_target_transform
        
        self.dataset = VOCSegmentation(
            root=root,
            year=year,
            image_set=image_set,
            download=download,
            transform=transform,
            target_transform=target_transform
        )
        
        self.num_classes = 21  # PASCAL VOC has 21 classes
        self.ignore_index = 255  # Boundary pixels marked as 255
        self.classes = VOC_CLASSES
        self.colormap = VOC_COLORMAP
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Returns:
            tuple: (image_tensor, mask_tensor)
                - image_tensor: [3, H, W] float tensor, normalized
                - mask_tensor: [H, W] long tensor, values in [0-20, 255]
        """
        return self.dataset[idx]
    
    def get_class_name(self, class_idx):
        """Get class name from class index."""
        if 0 <= class_idx < len(self.classes):
            return self.classes[class_idx]
        elif class_idx == self.ignore_index:
            return 'ignore'
        else:
            return 'unknown'
    
    def get_subset(self, num_samples):
        """
        Get a subset of the dataset.
        
        Args:
            num_samples (int): Number of samples to include in subset
            
        Returns:
            Subset: PyTorch Subset of this dataset
        """
        if num_samples is None or num_samples >= len(self):
            return self
        
        indices = list(range(min(num_samples, len(self))))
        return Subset(self, indices)


def load_voc_dataset(root='./data', image_set='val', download=True):
    """
    Convenience function to load PASCAL VOC 2012 segmentation dataset.
    
    Args:
        root (str): Root directory for dataset
        image_set (str): 'train', 'val', or 'trainval'
        download (bool): Whether to download if not found
        
    Returns:
        VOCSegmentationDataset: Dataset ready for evaluation
    """
    print(f"Loading PASCAL VOC 2012 {image_set} dataset...")
    dataset = VOCSegmentationDataset(
        root=root,
        year='2012',
        image_set=image_set,
        download=download
    )
    print(f"Loaded {len(dataset)} images")
    return dataset


def decode_segmentation_mask(mask, colormap=None):
    """
    Convert segmentation mask to RGB image for visualization.
    
    Args:
        mask (torch.Tensor or np.ndarray): [H, W] segmentation mask with class indices
        colormap (list): List of [R, G, B] colors for each class
        
    Returns:
        np.ndarray: [H, W, 3] RGB image
    """
    if colormap is None:
        colormap = VOC_COLORMAP
    
    # Convert to numpy if tensor
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    
    # Create RGB image
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_idx in range(len(colormap)):
        rgb[mask == class_idx] = colormap[class_idx]
    
    # Handle ignore index (255) - make it white
    rgb[mask == 255] = [255, 255, 255]
    
    return rgb


if __name__ == "__main__":
    """Test dataset loading."""
    print("Testing PASCAL VOC 2012 Dataset Loading...")
    print("=" * 60)
    
    # Load dataset
    dataset = load_voc_dataset(root='./data', image_set='val', download=True)
    
    print(f"\nDataset Information:")
    print(f"  Number of samples: {len(dataset)}")
    print(f"  Number of classes: {dataset.num_classes}")
    print(f"  Ignore index: {dataset.ignore_index}")
    print(f"  Classes: {', '.join(dataset.classes)}")
    
    # Test loading a sample
    print(f"\nLoading first sample...")
    image, mask = dataset[0]
    
    print(f"  Image shape: {image.shape}")
    print(f"  Image dtype: {image.dtype}")
    print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"  Mask shape: {mask.shape}")
    print(f"  Mask dtype: {mask.dtype}")
    print(f"  Unique classes in mask: {torch.unique(mask).tolist()}")
    
    # Test subset functionality
    print(f"\nTesting subset functionality...")
    subset = dataset.get_subset(10)
    print(f"  Subset size: {len(subset)}")
    
    print("\n" + "=" * 60)
    print("Dataset loading test complete!")

