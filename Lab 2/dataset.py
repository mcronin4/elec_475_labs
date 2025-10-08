import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import re
import matplotlib.pyplot as plt
import numpy as np


class PetNoseDataset(Dataset):
    """
    Custom Dataset for oxford-iiit-pet-noses dataset.
    
    Handles loading images and nose coordinate labels from the dataset.
    Supports data augmentation transforms.
    """
    
    def __init__(self, images_dir, labels_file, transform=None):
        """
        Initialize the dataset.
        
        Args:
            images_dir (str): Path to directory containing images
            labels_file (str): Path to text file with image names and coordinates
            transform (callable, optional): Optional transform to be applied on samples
        """
        self.images_dir = images_dir
        self.transform = transform
        self.samples = []
        
        # Parse labels file
        self._load_labels(labels_file)
        
    def _load_labels(self, labels_file):
        """
        Load image filenames and nose coordinates from labels file.
        
        Format: filename.jpg,"(x, y)"
        """
        with open(labels_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Find the comma that separates filename from coordinates
                    # Look for the pattern: filename,"(x, y)"
                    comma_quote_idx = line.find(',"')
                    if comma_quote_idx != -1:
                        filename = line[:comma_quote_idx].strip()
                        coords_str = line[comma_quote_idx+1:].strip()
                        
                        # Extract coordinates using regex: "(x, y)" - remove quotes first
                        coords_str_clean = coords_str.strip('"')
                        coords_match = re.search(r'\((\d+),\s*(\d+)\)', coords_str_clean)
                        if coords_match:
                            x = int(coords_match.group(1))
                            y = int(coords_match.group(2))
                            
                            # Check if image file exists
                            image_path = os.path.join(self.images_dir, filename)
                            if os.path.exists(image_path):
                                self.samples.append({
                                    'image_path': image_path,
                                    'filename': filename,
                                    'coordinates': (x, y)
                                })
                            else:
                                print(f"Warning: Image not found: {image_path}")
                        else:
                            print(f"Warning: Could not parse coordinates from: {coords_str}")
                    else:
                        print(f"Warning: Invalid line format: {line}")
        
        print(f"Loaded {len(self.samples)} valid samples from {labels_file}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Returns:
            tuple: (image_tensor, coordinates_tensor)
        """
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        original_size = image.size  # (width, height)
        
        # Get coordinates
        coordinates = sample['coordinates']
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        # Scale coordinates to match resized image (227x227)
        # Calculate scale factors
        scale_x = 227.0 / original_size[0]  # width
        scale_y = 227.0 / original_size[1]  # height
        
        # Scale coordinates
        scaled_coordinates = (
            coordinates[0] * scale_x,
            coordinates[1] * scale_y
        )
        
        # Convert coordinates to tensor
        coordinates_tensor = torch.tensor(scaled_coordinates, dtype=torch.float32)
        
        return image, coordinates_tensor
    
    def get_sample_info(self, idx):
        """
        Get sample information without loading the image.
        Useful for debugging and visualization.
        """
        sample = self.samples[idx]
        return {
            'filename': sample['filename'],
            'coordinates': sample['coordinates'],
            'image_path': sample['image_path']
        }


def get_transforms(resize_size=227, augmentation=False):
    """
    Get transform pipelines for training and testing.
    
    Args:
        resize_size (int): Target size for image resizing (227 for SnoutNet)
        augmentation (bool): Whether to apply data augmentation
        
    Returns:
        transforms.Compose: Transform pipeline
    """
    if augmentation:
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=10),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Basic transforms without augmentation
        transform = transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def reality_check_dataset(dataset, num_samples=5, save_plots=True):
    """
    Reality check routine to verify dataset loading.
    
    Args:
        dataset (PetNoseDataset): Dataset to check
        num_samples (int): Number of samples to visualize
        save_plots (bool): Whether to save plots to files
    """
    print(f"\n=== Reality Check for Dataset ===")
    print(f"Total samples: {len(dataset)}")
    
    # Check first few samples
    for i in range(min(num_samples, len(dataset))):
        sample_info = dataset.get_sample_info(i)
        print(f"\nSample {i}:")
        print(f"  Filename: {sample_info['filename']}")
        print(f"  Coordinates: {sample_info['coordinates']}")
        print(f"  Image path: {sample_info['image_path']}")
        
        # Load image and coordinates
        image, coordinates = dataset[i]
        
        print(f"  Image tensor shape: {image.shape}")
        print(f"  Image tensor dtype: {image.dtype}")
        print(f"  Image tensor range: [{image.min():.3f}, {image.max():.3f}]")
        print(f"  Coordinates tensor: {coordinates}")
        print(f"  Coordinates tensor dtype: {coordinates.dtype}")
        
        # Verify image dimensions
        expected_shape = (3, 227, 227)
        if image.shape != expected_shape:
            print(f"  WARNING: Expected shape {expected_shape}, got {image.shape}")
        
        # Verify coordinates are reasonable (within image bounds)
        if coordinates[0] < 0 or coordinates[0] > 227 or coordinates[1] < 0 or coordinates[1] > 227:
            print(f"  WARNING: Coordinates {coordinates} may be outside image bounds")
    
    # Visualize samples
    if save_plots:
        visualize_samples(dataset, num_samples=num_samples)


def visualize_samples(dataset, num_samples=5, save_path="reality_check_samples.png"):
    """
    Visualize dataset samples with nose coordinates marked.
    
    Args:
        dataset (PetNoseDataset): Dataset to visualize
        num_samples (int): Number of samples to show
        save_path (str): Path to save the visualization
    """
    fig, axes = plt.subplots(1, min(num_samples, len(dataset)), figsize=(15, 3))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(min(num_samples, len(dataset))):
        # Get sample
        image, coordinates = dataset[i]
        
        # Convert tensor to numpy for visualization
        # Denormalize image
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_denorm = image * std + mean
        image_denorm = torch.clamp(image_denorm, 0, 1)
        
        # Convert to numpy and transpose for matplotlib
        image_np = image_denorm.permute(1, 2, 0).numpy()
        
        # Plot
        axes[i].imshow(image_np)
        axes[i].scatter(coordinates[0].item(), coordinates[1].item(), 
                       c='red', s=100, marker='x', linewidths=3)
        axes[i].set_title(f"{dataset.get_sample_info(i)['filename']}\nCoords: {coordinates.tolist()}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    # Test the dataset
    print("Testing PetNoseDataset...")
    
    # Dataset paths
    images_dir = "oxford-iiit-pet-noses/images-original/images"
    train_labels = "oxford-iiit-pet-noses/train_noses.txt"
    test_labels = "oxford-iiit-pet-noses/test_noses.txt"
    
    # Test basic transforms
    print("\n=== Testing Basic Transforms ===")
    basic_transform = get_transforms(augmentation=False)
    train_dataset = PetNoseDataset(images_dir, train_labels, transform=basic_transform)
    test_dataset = PetNoseDataset(images_dir, test_labels, transform=basic_transform)
    
    # Reality check
    reality_check_dataset(train_dataset, num_samples=3)
    
    # Test augmentation transforms
    print("\n=== Testing Augmentation Transforms ===")
    aug_transform = get_transforms(augmentation=True)
    train_dataset_aug = PetNoseDataset(images_dir, train_labels, transform=aug_transform)
    reality_check_dataset(train_dataset_aug, num_samples=2)
