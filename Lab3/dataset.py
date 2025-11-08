#########################################################################################################
#
#   ELEC 475 - Lab 3: Semantic Segmentation
#   Fall 2025
#
#   dataset.py - PASCAL VOC 2012 dataset handling for semantic segmentation
#

from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Subset
import torchvision.transforms as transforms


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
    PASCAL VOC 2012 segmentation dataset loader for the local Kaggle layout.
    
    This implementation assumes the dataset has been manually placed inside
    `root/` with the following directories:
        - VOC2012_train_val
        - VOC2012_test
    
    Both directories should contain the original PASCAL file structure
    (`Annotations`, `ImageSets/Segmentation`, `JPEGImages`, `SegmentationClass`, ...).
    """
    
    _SPLIT_DIRECTORY_MAP = {
        'train': 'VOC2012_train_val',
        'val': 'VOC2012_train_val',
        'trainval': 'VOC2012_train_val',
        'test': 'VOC2012_test',
    }
    
    def __init__(self, root='./data', image_set='val', transform=None, target_transform=None, use_albumentations=False):
        """
        Initialize PASCAL VOC segmentation dataset.
        
        Args:
            root (str): Root directory for dataset (containing VOC2012_* folders)
            image_set (str): 'train', 'val', 'trainval', or 'test'
            transform (callable): Transform for images (torchvision or albumentations)
            target_transform (callable): Transform for segmentation masks (only used if not using albumentations)
            use_albumentations (bool): Whether to use Albumentations transforms
        """
        self.root = Path(root).expanduser()
        self.image_set = image_set.lower()
        self.use_albumentations = use_albumentations
        
        if self.image_set not in self._SPLIT_DIRECTORY_MAP:
            valid_sets = ', '.join(sorted(self._SPLIT_DIRECTORY_MAP.keys()))
            raise ValueError(f"Invalid image_set '{image_set}'. Supported sets: {valid_sets}")
        
        split_dir = self.root / self._SPLIT_DIRECTORY_MAP[self.image_set]
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Expected dataset directory '{split_dir}' not found.\n"
                "Ensure you extracted the Kaggle archive so that the directory structure is:\n"
                f"  {self.root}/VOC2012_train_val/... and {self.root}/VOC2012_test/..."
            )
        
        # Determine file lists and required subdirectories
        self.images_dir = split_dir / 'JPEGImages'
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Missing JPEGImages directory: {self.images_dir}")
        
        self.masks_dir = split_dir / 'SegmentationClass'
        if self.image_set == 'test':
            if not self.masks_dir.exists():
                raise ValueError(
                    "The PASCAL VOC test split does not include ground-truth segmentation masks.\n"
                    "Use the 'val', 'train', or 'trainval' splits for supervised evaluation."
                )
        else:
            if not self.masks_dir.exists():
                raise FileNotFoundError(
                    f"Missing SegmentationClass directory for split '{self.image_set}': {self.masks_dir}"
                )
        
        split_list_file = split_dir / 'ImageSets' / 'Segmentation' / f'{self.image_set}.txt'
        if not split_list_file.exists():
            raise FileNotFoundError(
                f"Image set file not found: {split_list_file}\n"
                "Verify the Kaggle dataset contains the segmentation split lists."
            )
        
        self.ids: List[str] = self._load_ids(split_list_file)
        if len(self.ids) == 0:
            raise RuntimeError(f"No image ids found in {split_list_file}")
        
        # Set up transforms (use defaults if none provided)
        if transform is None or target_transform is None:
            default_img_transform, default_target_transform = get_voc_transforms()
            if transform is None:
                transform = default_img_transform
            if target_transform is None:
                target_transform = default_target_transform
        
        self.transform = transform
        self.target_transform = target_transform
        
        self.num_classes = 21  # PASCAL VOC has 21 classes
        self.ignore_index = 255  # Boundary pixels marked as 255
        self.classes = VOC_CLASSES
        self.colormap = VOC_COLORMAP
    
    @staticmethod
    def _load_ids(list_file: Path) -> List[str]:
        with list_file.open('r') as f:
            return [line.strip() for line in f if line.strip()]
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Returns:
            tuple: (image_tensor, mask_tensor)
                - image_tensor: [3, H, W] float tensor, normalized
                - mask_tensor: [H, W] long tensor, values in [0-20, 255]
        """
        image_id = self.ids[idx]
        
        image_path = self.images_dir / f'{image_id}.jpg'
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        mask_path = self.masks_dir / f'{image_id}.png'
        if not mask_path.exists():
            raise FileNotFoundError(
                f"Mask not found for image '{image_id}': {mask_path}\n"
                "Ensure you are using a split that includes segmentation masks."
            )
        
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path)
        
        if self.use_albumentations:
            # Convert PIL images to numpy arrays for Albumentations
            image_np = np.array(image)
            mask_np = np.array(mask, dtype=np.int64)
            
            # Apply Albumentations transform (handles both image and mask)
            if self.transform is not None:
                augmented = self.transform(image=image_np, mask=mask_np)
                image = augmented['image']
                mask = augmented['mask']
                
                # Ensure mask is Long (int64) - Albumentations may return int32
                if not isinstance(mask, torch.Tensor):
                    mask = torch.from_numpy(mask).long()
                else:
                    mask = mask.long()
            else:
                # Convert to tensors manually if no transform provided
                image = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
                mask = torch.from_numpy(mask_np).long()
        else:
            # Use torchvision transforms (original behavior)
            if self.transform is not None:
                image = self.transform(image)
            
            if self.target_transform is not None:
                mask = self.target_transform(mask)
        
        return image, mask
    
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


def load_voc_dataset(root='./data', image_set='val'):
    """
    Convenience function to load PASCAL VOC 2012 segmentation dataset.
    
    Args:
        root (str): Root directory for dataset
        image_set (str): 'train', 'val', or 'trainval'
        
    Returns:
        VOCSegmentationDataset: Dataset ready for evaluation
    """
    print(f"Loading PASCAL VOC 2012 {image_set} dataset...")
    dataset = VOCSegmentationDataset(
        root=root,
        image_set=image_set,
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
    dataset = load_voc_dataset(root='./data', image_set='val')
    
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

