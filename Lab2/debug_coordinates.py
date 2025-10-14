import torch
from PIL import Image
import os
from dataset import PetNoseDataset, get_transforms

# Test with a specific image to debug coordinate issues
images_dir = "oxford-iiit-pet-noses/images-original/images"
train_labels = "oxford-iiit-pet-noses/train_noses.txt"

# Load the first sample without any transforms
dataset_no_transform = PetNoseDataset(images_dir, train_labels, transform=None)

print("=== Debugging Coordinate Issues ===")
print(f"First sample info: {dataset_no_transform.get_sample_info(0)}")

# Load the original image
sample_info = dataset_no_transform.get_sample_info(0)
original_image = Image.open(sample_info['image_path']).convert('RGB')

print(f"Original image size: {original_image.size}")  # (width, height)
print(f"Original coordinates: {sample_info['coordinates']}")  # (x, y)

# Now load with transforms
basic_transform = get_transforms(augmentation=False)
dataset_with_transform = PetNoseDataset(images_dir, train_labels, transform=basic_transform)

image_tensor, coordinates_tensor = dataset_with_transform[0]
print(f"Transformed image tensor shape: {image_tensor.shape}")  # (3, 227, 227)
print(f"Coordinates tensor: {coordinates_tensor}")  # (x, y)

# Check if coordinates are being scaled properly
original_coords = sample_info['coordinates']
print(f"Original coords: {original_coords}")
print(f"Tensor coords: {coordinates_tensor.tolist()}")

# The coordinates should be scaled from original image size to 227x227
original_width, original_height = original_image.size
print(f"Original image: {original_width} x {original_height}")
print(f"Target image: 227 x 227")

# Calculate expected scaling
scale_x = 227 / original_width
scale_y = 227 / original_height
print(f"Expected scale factors: x={scale_x:.3f}, y={scale_y:.3f}")

# Check if coordinates are being scaled
expected_scaled_x = original_coords[0] * scale_x
expected_scaled_y = original_coords[1] * scale_y
print(f"Expected scaled coordinates: ({expected_scaled_x:.1f}, {expected_scaled_y:.1f})")
print(f"Actual coordinates in dataset: {coordinates_tensor.tolist()}")
