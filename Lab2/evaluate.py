import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import argparse

from model import SnoutNet
from model_alexnet import AlexNetNose
from model_vgg16 import VGG16Nose
from dataset import PetNoseDataset, get_transforms


def detect_model_type(model_path):
    """
    Detect model type from filename.
    
    Args:
        model_path: Path to model file
        
    Returns:
        str: Model type ('alexnet', 'vgg16', or 'snoutnet')
    """
    model_path_lower = model_path.lower()
    
    if 'alexnet' in model_path_lower:
        return 'alexnet'
    elif 'vgg16' in model_path_lower or 'vgg' in model_path_lower:
        return 'vgg16'
    else:
        return 'snoutnet'


def evaluate_model(model_path, batch_size=32, visualize=True, save_path="evaluation_samples/evaluation_samples.png"):
    """
    Evaluate the trained model on the test dataset.
    Supports SnoutNet, AlexNet, and VGG16 architectures.
    
    Args:
        model_path: Path to saved model weights (required)
        batch_size: Batch size for evaluation
        visualize: Whether to create visualizations
        save_path: Path to save visualization output
        
    Returns:
        dict: Dictionary containing statistics and results
    """
    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    
    # Set device - support CUDA, MPS (Apple Metal), and CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\nUsing device: CUDA GPU")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"\nUsing device: Apple Metal (MPS)")
    else:
        device = torch.device("cpu")
        print(f"\nUsing device: CPU")
    
    print("\n" + "-" * 60)
    print("Loading Model...")
    print("-" * 60)
    
    # Detect model type from filename
    model_type = detect_model_type(model_path)
    print(f"Detected model type: {model_type.upper()}")
    
    # Initialize appropriate model architecture
    if model_type == 'alexnet':
        model = AlexNetNose(pretrained=False)  # Don't need pretrained weights, loading from file
    elif model_type == 'vgg16':
        model = VGG16Nose(pretrained=False)  # Don't need pretrained weights, loading from file
    else:  # snoutnet
        model = SnoutNet()
    
    # Load trained weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    print(f"Loaded model from: {model_path}")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    print("\n" + "-" * 60)
    print("Loading Test Dataset...")
    print("-" * 60)
    
    # Dataset paths
    images_dir = "oxford-iiit-pet-noses/images-original/images"
    test_labels = "oxford-iiit-pet-noses/test_noses.txt"
    
    # Get transforms (no augmentation)
    transform = get_transforms(resize_size=227, color_aug=False, blur_aug=False)
    
    # Create test dataset
    test_dataset = PetNoseDataset(images_dir, test_labels, transform=transform)
    print(f"Test samples: {len(test_dataset)}")
    
    # Create data loader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=2, pin_memory=True)
    
    print("\n" + "-" * 60)
    print("Running Inference...")
    print("-" * 60)
    
    # Storage for predictions and ground truth
    all_predictions = []
    all_ground_truth = []
    
    # Inference loop
    with torch.no_grad():
        for images, coordinates in tqdm(test_loader, desc="Evaluating"):
            # Move data to device
            images = images.to(device)
            coordinates = coordinates.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Store results (move to CPU for processing)
            all_predictions.append(outputs.cpu())
            all_ground_truth.append(coordinates.cpu())
    
    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=0)  # Shape: (num_samples, 2)
    all_ground_truth = torch.cat(all_ground_truth, dim=0)  # Shape: (num_samples, 2)
    
    print(f"Predictions shape: {all_predictions.shape}")
    print(f"Ground truth shape: {all_ground_truth.shape}")
    
    print("\n" + "-" * 60)
    print("Computing Euclidean Distances...")
    print("-" * 60)
    
    # Calculate Euclidean distances
    # distance = sqrt((pred_u - true_u)^2 + (pred_v - true_v)^2)
    distances = torch.sqrt(
        (all_predictions[:, 0] - all_ground_truth[:, 0])**2 + 
        (all_predictions[:, 1] - all_ground_truth[:, 1])**2
    )
    
    # Convert to numpy for statistics
    distances_np = distances.numpy()
    
    # Compute statistics
    min_distance = np.min(distances_np)
    mean_distance = np.mean(distances_np)
    max_distance = np.max(distances_np)
    std_distance = np.std(distances_np)
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nTest samples: {len(test_dataset)}")
    print(f"\nLocalization Accuracy (Euclidean Distance in pixels):")
    print(f"  Minimum:   {min_distance:.2f} pixels")
    print(f"  Mean:      {mean_distance:.2f} pixels")
    print(f"  Maximum:   {max_distance:.2f} pixels")
    print(f"  Std Dev:   {std_distance:.2f} pixels")
    
    # Prepare results dictionary
    results = {
        'model_path': model_path,
        'num_samples': len(test_dataset),
        'predictions': all_predictions.numpy(),
        'ground_truth': all_ground_truth.numpy(),
        'distances': distances_np,
        'statistics': {
            'min': min_distance,
            'mean': mean_distance,
            'max': max_distance,
            'std': std_distance
        }
    }
    
    # Visualizations
    if visualize:
        print("\n" + "-" * 60)
        print("Creating Visualizations...")
        print("-" * 60)
        
        # Visualize best, worst, and average predictions
        visualize_predictions(test_dataset, all_predictions.numpy(), 
                            all_ground_truth.numpy(), distances_np,
                            mean_distance,
                            save_path=save_path)
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    
    return results


def visualize_predictions(dataset, predictions, ground_truth, distances, mean_distance,
                         num_samples=9, save_path="evaluation_samples/evaluation_samples.png"):
    """
    Visualize sample predictions with best, worst, and average cases.
    
    Args:
        dataset: PetNoseDataset
        predictions: Array of predicted coordinates
        ground_truth: Array of ground truth coordinates
        distances: Array of Euclidean distances
        mean_distance: Mean distance value
        num_samples: Number of samples to show (will show 3 each of best, worst, average)
        save_path: Path to save the visualization
    """
    # Find best, worst, and most average predictions
    sorted_indices = np.argsort(distances)
    num_per_category = num_samples // 3
    
    # Best predictions (smallest distances)
    best_indices = sorted_indices[:num_per_category]
    
    # Worst predictions (largest distances)
    worst_indices = sorted_indices[-num_per_category:]
    
    # Most average predictions (closest to mean distance)
    distance_from_mean = np.abs(distances - mean_distance)
    average_indices = np.argsort(distance_from_mean)[:num_per_category]
    
    # Combine indices in order: best, average, worst
    sample_indices = np.concatenate([best_indices, average_indices, worst_indices])
    
    # Create figure with 3 rows
    fig, axes = plt.subplots(3, num_per_category, figsize=(15, 9))
    axes = axes.flatten()
    
    for i, idx in enumerate(sample_indices):
        # Get sample
        image, _ = dataset[idx]
        
        # Denormalize image for display
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_denorm = image * std + mean
        image_denorm = torch.clamp(image_denorm, 0, 1)
        
        # Convert to numpy
        image_np = image_denorm.permute(1, 2, 0).numpy()
        
        # Get coordinates
        pred_coords = predictions[idx]
        true_coords = ground_truth[idx]
        dist = distances[idx]
        
        # Plot
        axes[i].imshow(image_np)
        axes[i].scatter(true_coords[0], true_coords[1], 
                       c='green', s=100, marker='x', linewidths=3, label='Ground Truth')
        axes[i].scatter(pred_coords[0], pred_coords[1], 
                       c='red', s=100, marker='o', linewidths=2, 
                       facecolors='none', label='Prediction')
        
        # Draw line between prediction and ground truth
        axes[i].plot([true_coords[0], pred_coords[0]], 
                    [true_coords[1], pred_coords[1]], 
                    'y--', linewidth=1.5, alpha=0.7)
        
        # Add title with distance
        sample_info = dataset.get_sample_info(idx)
        title = f"{sample_info['filename']}\nError: {dist:.2f}px"
        axes[i].set_title(title, fontsize=8)
        axes[i].axis('off')
        
        # Add legend only to first subplot
        if i == 0:
            axes[i].legend(loc='upper right', fontsize=6)
    
    # Add overall title and row labels
    fig.suptitle(f'Best {num_per_category}, Average {num_per_category}, and Worst {num_per_category} Predictions', 
                fontsize=14, fontweight='bold', y=0.99)
    
    # Add text labels for each row
    fig.text(0.02, 0.83, 'Best', fontsize=12, fontweight='bold', rotation=90, 
             verticalalignment='center')
    fig.text(0.02, 0.50, 'Average', fontsize=12, fontweight='bold', rotation=90, 
             verticalalignment='center')
    fig.text(0.02, 0.17, 'Worst', fontsize=12, fontweight='bold', rotation=90, 
             verticalalignment='center')
    
    plt.tight_layout(rect=[0.03, 0, 1, 0.98])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Sample predictions saved to: {save_path}")
    plt.close()


def main():
    """Main function to handle command line arguments and start evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate model on test dataset (SnoutNet, AlexNet, or VGG16)')
    parser.add_argument('-m', '--model', type=str, required=True,
                       help='Path to model file (e.g., best_snoutnet.pth, best_alexnet_aug.pth, best_vgg16.pth)')
    parser.add_argument('-b', '--batch-size', type=int, default=32,
                       help='Batch size for evaluation (default: 32)')
    
    args = parser.parse_args()
    
    # Extract model type and suffix from model filename for output files
    # e.g., "best_alexnet_aug.pth" -> "evaluation_samples_alexnet_aug.png"
    model_basename = os.path.basename(args.model)
    model_type = detect_model_type(args.model)
    
    # include model type in filename
    suffix = "_aug" if "_aug" in model_basename else ""
    output_path = f"evaluation_samples_{model_type}{suffix}.png"
    
    output_path = "evaluation_samples/" + output_path

    print("\n" + "=" * 60)
    print("Evaluation Configuration:")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output file: {output_path}")
    
    # Evaluate the trained model
    results = evaluate_model(
        model_path=args.model,
        batch_size=args.batch_size,
        visualize=True,
        save_path=output_path
    )
    
    print("\n" + "=" * 60)
    print("Evaluation script finished successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

