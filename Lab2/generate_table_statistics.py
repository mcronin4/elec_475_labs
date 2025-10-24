"""
Generate comprehensive statistics for all models to fill out the results table.
Evaluates SnoutNet, AlexNet, VGG16, and Ensemble (all with both augmentations).
"""
import torch
from torch.utils.data import DataLoader
import numpy as np
import csv
from tqdm import tqdm
import os

from model import SnoutNet
from model_alexnet import AlexNetNose
from model_vgg16 import VGG16Nose
from model_ensemble import EnsembleModel
from dataset import PetNoseDataset, get_transforms


def evaluate_model_detailed(model, test_loader, device, model_name):
    """
    Evaluate a model and compute detailed statistics.
    
    Args:
        model: The model to evaluate
        test_loader: DataLoader for test set
        device: Device to run on
        model_name: Name of the model for display
        
    Returns:
        dict: Dictionary with all statistics
    """
    print(f"\nEvaluating {model_name}...")
    
    model.eval()
    all_predictions = []
    all_ground_truth = []
    
    # Inference loop
    with torch.no_grad():
        for images, coordinates in tqdm(test_loader, desc=f"{model_name}", leave=False):
            images = images.to(device)
            coordinates = coordinates.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Store results
            all_predictions.append(outputs.cpu())
            all_ground_truth.append(coordinates.cpu())
    
    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=0)
    all_ground_truth = torch.cat(all_ground_truth, dim=0)
    
    # Calculate Euclidean distances
    distances = torch.sqrt(
        (all_predictions[:, 0] - all_ground_truth[:, 0])**2 + 
        (all_predictions[:, 1] - all_ground_truth[:, 1])**2
    ).numpy()
    
    # Overall statistics (all test samples)
    overall_stats = {
        'min': np.min(distances),
        'max': np.max(distances),
        'mean': np.mean(distances),
        'stdev': np.std(distances)
    }
    
    # Get indices for best and worst predictions
    sorted_indices = np.argsort(distances)
    
    # 4 Best predictions (smallest distances)
    best_4_indices = sorted_indices[:4]
    best_4_distances = distances[best_4_indices]
    best_4_stats = {
        'min': np.min(best_4_distances),
        'max': np.max(best_4_distances),
        'mean': np.mean(best_4_distances),
        'stdev': np.std(best_4_distances)
    }
    
    # 4 Worst predictions (largest distances)
    worst_4_indices = sorted_indices[-4:]
    worst_4_distances = distances[worst_4_indices]
    worst_4_stats = {
        'min': np.min(worst_4_distances),
        'max': np.max(worst_4_distances),
        'mean': np.mean(worst_4_distances),
        'stdev': np.std(worst_4_distances)
    }
    
    return {
        'model_name': model_name,
        'overall': overall_stats,
        'best_4': best_4_stats,
        'worst_4': worst_4_stats,
        'num_samples': len(distances)
    }


def load_model(model_type, device, use_augmentation=True):
    """
    Load a specific model with or without augmentation weights.
    
    Args:
        model_type: 'snoutnet', 'alexnet', 'vgg16', or 'ensemble'
        device: Device to load on
        use_augmentation: If True, load aug_both models; if False, load non-aug models
        
    Returns:
        Loaded model
    """
    suffix = "_aug_both" if use_augmentation else ""
    
    if model_type == 'snoutnet':
        model_path = f"model_weights/snoutnet/best_snoutnet{suffix}.pth"
        model = SnoutNet()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        
    elif model_type == 'alexnet':
        model_path = f"model_weights/alexnet/best_alexnet{suffix}.pth"
        model = AlexNetNose(pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        
    elif model_type == 'vgg16':
        model_path = f"model_weights/vgg16/best_vgg16{suffix}.pth"
        model = VGG16Nose(pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        
    elif model_type == 'ensemble':
        snoutnet_path = f"model_weights/snoutnet/best_snoutnet{suffix}.pth"
        alexnet_path = f"model_weights/alexnet/best_alexnet{suffix}.pth"
        vgg16_path = f"model_weights/vgg16/best_vgg16{suffix}.pth"
        model = EnsembleModel(snoutnet_path, alexnet_path, vgg16_path, device=device)
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.eval()
    return model


def generate_all_statistics(batch_size=32):
    """
    Generate statistics for all four models.
    
    Args:
        batch_size: Batch size for evaluation
        
    Returns:
        list: List of result dictionaries
    """
    print("=" * 70)
    print("GENERATING TABLE STATISTICS FOR ALL MODELS")
    print("=" * 70)
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\nUsing device: CUDA GPU - {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"\nUsing device: Apple Metal (MPS)")
    else:
        device = torch.device("cpu")
        print(f"\nUsing device: CPU")
    
    # Load test dataset
    print("\nLoading test dataset...")
    images_dir = "oxford-iiit-pet-noses/images-original/images"
    test_labels = "oxford-iiit-pet-noses/test_noses.txt"
    transform = get_transforms(resize_size=227, color_aug=False, blur_aug=False)
    test_dataset = PetNoseDataset(images_dir, test_labels, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=2, pin_memory=True)
    print(f"Test samples: {len(test_dataset)}")
    
    # Models to evaluate (both with and without augmentation)
    models_config = [
        ('snoutnet', 'SnoutNet'),
        ('alexnet', 'SnoutNet-A'),
        ('vgg16', 'SnoutNet-V'),
        ('ensemble', 'SnoutNet-Ensemble')
    ]
    
    augmentation_configs = [
        (False, 'No Augmentation', ''),
        (True, 'With Augmentation', 'x')
    ]
    
    results = []
    
    # Evaluate each model with both augmentation settings
    for model_type, display_name in models_config:
        for use_aug, aug_desc, aug_marker in augmentation_configs:
            try:
                print(f"\n{'=' * 70}")
                full_name = f"{display_name} ({aug_desc})"
                print(f"Loading {full_name}...")
                
                model = load_model(model_type, device, use_augmentation=use_aug)
                
                # Evaluate
                result = evaluate_model_detailed(model, test_loader, device, display_name)
                
                # Add augmentation marker to result
                result['augmentation'] = aug_marker
                results.append(result)
                
                # Print immediate results
                print(f"\n{full_name} Results:")
                print(f"  Overall: mean={result['overall']['mean']:.4f}, stdev={result['overall']['stdev']:.4f}")
                print(f"  Best 4:  mean={result['best_4']['mean']:.4f}, stdev={result['best_4']['stdev']:.4f}")
                print(f"  Worst 4: mean={result['worst_4']['mean']:.4f}, stdev={result['worst_4']['stdev']:.4f}")
                
            except Exception as e:
                print(f"\n❌ Error evaluating {full_name}: {e}")
                continue
    
    return results


def save_results_to_csv(results, output_path="table_statistics.csv"):
    """
    Save results to CSV file in table format.
    
    Args:
        results: List of result dictionaries
        output_path: Path to save CSV file
    """
    print(f"\n{'=' * 70}")
    print("Saving results to CSV...")
    
    # Prepare CSV data
    headers = [
        'Model', 'Augmentation',
        'min', 'max', 'mean', 'stdev',  # Overall
        'min', 'max', 'mean', 'stdev',  # Best 4
        'min', 'max', 'mean', 'stdev'   # Worst 4
    ]
    
    rows = []
    for result in results:
        row = [
            result['model_name'],
            result['augmentation'],  # '' for no aug, 'x' for aug
            # Overall stats
            f"{result['overall']['min']:.4f}",
            f"{result['overall']['max']:.4f}",
            f"{result['overall']['mean']:.4f}",
            f"{result['overall']['stdev']:.4f}",
            # Best 4 stats
            f"{result['best_4']['min']:.4f}",
            f"{result['best_4']['max']:.4f}",
            f"{result['best_4']['mean']:.4f}",
            f"{result['best_4']['stdev']:.4f}",
            # Worst 4 stats
            f"{result['worst_4']['min']:.4f}",
            f"{result['worst_4']['max']:.4f}",
            f"{result['worst_4']['mean']:.4f}",
            f"{result['worst_4']['stdev']:.4f}"
        ]
        rows.append(row)
    
    # Write to CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    
    print(f"✓ Results saved to: {output_path}")
    
    # Also print to console in formatted table
    print_formatted_table(results)


def print_formatted_table(results):
    """
    Print results in a nicely formatted table.
    
    Args:
        results: List of result dictionaries
    """
    print(f"\n{'=' * 70}")
    print("FORMATTED TABLE RESULTS")
    print("=" * 70)
    
    # Print header
    print(f"\n{'Model':<20} {'Aug':<5} | {'Localization Error (Overall)':<35} | {'Best 4':<35} | {'Worst 4':<35}")
    print(f"{'':<20} {'':<5} | {'min':>7} {'max':>7} {'mean':>8} {'stdev':>8} | {'min':>7} {'max':>7} {'mean':>8} {'stdev':>8} | {'min':>7} {'max':>7} {'mean':>8} {'stdev':>8}")
    print("-" * 160)
    
    # Print each model
    for result in results:
        overall = result['overall']
        best_4 = result['best_4']
        worst_4 = result['worst_4']
        aug_marker = result['augmentation'] if result['augmentation'] else ''
        
        print(f"{result['model_name']:<20} {aug_marker:<5} | "
              f"{overall['min']:>9.4f} {overall['max']:>9.4f} {overall['mean']:>10.4f} {overall['stdev']:>10.4f} | "
              f"{best_4['min']:>9.4f} {best_4['max']:>9.4f} {best_4['mean']:>10.4f} {best_4['stdev']:>10.4f} | "
              f"{worst_4['min']:>9.4f} {worst_4['max']:>9.4f} {worst_4['mean']:>10.4f} {worst_4['stdev']:>10.4f}")
    
    print("=" * 70)


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate table statistics for all models')
    parser.add_argument('-b', '--batch-size', type=int, default=32,
                       help='Batch size for evaluation (default: 32)')
    parser.add_argument('-o', '--output', type=str, default='table_statistics.csv',
                       help='Output CSV file path (default: table_statistics.csv)')
    
    args = parser.parse_args()
    
    # Generate statistics
    results = generate_all_statistics(batch_size=args.batch_size)
    
    if results:
        # Save to CSV
        save_results_to_csv(results, output_path=args.output)
        
        print(f"\n{'=' * 70}")
        print("✓ ALL STATISTICS GENERATED SUCCESSFULLY!")
        print(f"{'=' * 70}")
        print(f"\nYou can now copy the values from '{args.output}' into your table.")
    else:
        print("\n❌ No results generated. Please check that all model weights exist.")


if __name__ == "__main__":
    main()

