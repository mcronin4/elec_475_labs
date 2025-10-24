"""
Script to evaluate all three ensemble configurations and create comparison plots.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

from evaluate_ensemble import evaluate_ensemble


def evaluate_all_ensembles(batch_size=32):
    """
    Evaluate all three ensemble configurations (no aug, color aug, both augs).
    
    Args:
        batch_size: Batch size for evaluation
        
    Returns:
        dict: Results for all three ensembles
    """
    print("=" * 70)
    print(" " * 20 + "ENSEMBLE EVALUATION")
    print("=" * 70)
    
    results = {}
    
    # Configuration for three ensemble versions
    configs = [
        {
            'name': 'No Augmentation',
            'suffix': '',
            'output_suffix': '_no_aug',
            'color': '#1f77b4'
        },
        {
            'name': 'Color Augmentation',
            'suffix': '_aug_color',
            'output_suffix': '_aug_color',
            'color': '#ff7f0e'
        },
        {
            'name': 'Both Augmentations',
            'suffix': '_aug_both',
            'output_suffix': '_aug_both',
            'color': '#2ca02c'
        }
    ]
    
    # Evaluate each configuration
    for config in configs:
        print(f"\n{'=' * 70}")
        print(f"Evaluating Ensemble: {config['name']}")
        print(f"{'=' * 70}")
        
        # Construct model paths
        snoutnet_path = f"model_weights/snoutnet/best_snoutnet{config['suffix']}.pth"
        alexnet_path = f"model_weights/alexnet/best_alexnet{config['suffix']}.pth"
        vgg16_path = f"model_weights/vgg16/best_vgg16{config['suffix']}.pth"
        
        # Check if all models exist
        missing_models = []
        for path in [snoutnet_path, alexnet_path, vgg16_path]:
            if not os.path.exists(path):
                missing_models.append(path)
        
        if missing_models:
            print(f"\n⚠ WARNING: Missing models for {config['name']}:")
            for path in missing_models:
                print(f"  - {path}")
            print("Skipping this ensemble configuration.\n")
            continue
        
        # Output path
        output_path = f"evaluation_samples/evaluation_samples_ensemble{config['output_suffix']}.png"
        
        try:
            # Evaluate ensemble
            result = evaluate_ensemble(
                snoutnet_path=snoutnet_path,
                alexnet_path=alexnet_path,
                vgg16_path=vgg16_path,
                batch_size=batch_size,
                visualize=True,
                save_path=output_path
            )
            
            # Store results
            results[config['name']] = result
            
        except Exception as e:
            print(f"\n❌ Error evaluating {config['name']}: {e}")
            continue
    
    return results


def create_comparison_plot(results, save_path="evaluation_samples/ensemble_comparison.png"):
    """
    Create a comparison plot showing performance across all ensemble configurations.
    
    Args:
        results: Dictionary of results from evaluate_all_ensembles
        save_path: Path to save the comparison plot
    """
    if not results:
        print("No results to plot!")
        return
    
    print(f"\n{'=' * 70}")
    print("Creating Comparison Plot...")
    print(f"{'=' * 70}")
    
    # Extract data
    config_names = list(results.keys())
    mean_errors = [results[name]['statistics']['mean'] for name in config_names]
    std_errors = [results[name]['statistics']['std'] for name in config_names]
    min_errors = [results[name]['statistics']['min'] for name in config_names]
    max_errors = [results[name]['statistics']['max'] for name in config_names]
    mse_losses = [results[name]['mse_loss'] for name in config_names]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Mean Euclidean Distance with error bars
    x_pos = np.arange(len(config_names))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'][:len(config_names)]
    
    bars1 = ax1.bar(x_pos, mean_errors, color=colors, alpha=0.7, edgecolor='black')
    ax1.errorbar(x_pos, mean_errors, yerr=std_errors, fmt='none', ecolor='black', 
                capsize=5, capthick=2)
    
    ax1.set_xlabel('Ensemble Configuration', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Mean Euclidean Distance (pixels)', fontsize=11, fontweight='bold')
    ax1.set_title('Localization Accuracy: Mean Error with Std Dev', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(config_names, rotation=15, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, mean_val) in enumerate(zip(bars1, mean_errors)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean_val:.2f}px',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 2: MSE Loss
    bars2 = ax2.bar(x_pos, mse_losses, color=colors, alpha=0.7, edgecolor='black')
    
    ax2.set_xlabel('Ensemble Configuration', fontsize=11, fontweight='bold')
    ax2.set_ylabel('MSE Loss', fontsize=11, fontweight='bold')
    ax2.set_title('Test Set MSE Loss', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(config_names, rotation=15, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, mse_val) in enumerate(zip(bars2, mse_losses)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{mse_val:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to: {save_path}")
    plt.close()
    
    # Print summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY TABLE")
    print(f"{'=' * 70}")
    print(f"{'Configuration':<25} {'MSE Loss':>12} {'Mean Error':>12} {'Std Dev':>10}")
    print("-" * 70)
    for name in config_names:
        mse = results[name]['mse_loss']
        mean_err = results[name]['statistics']['mean']
        std_err = results[name]['statistics']['std']
        print(f"{name:<25} {mse:>12.4f} {mean_err:>11.2f}px {std_err:>9.2f}px")
    print(f"{'=' * 70}")


def main():
    """Main function to run all ensemble evaluations."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate all ensemble configurations')
    parser.add_argument('-b', '--batch-size', type=int, default=32,
                       help='Batch size for evaluation (default: 32)')
    
    args = parser.parse_args()
    
    # Evaluate all ensembles
    results = evaluate_all_ensembles(batch_size=args.batch_size)
    
    # Create comparison plot
    if results:
        create_comparison_plot(results)
        
        print("\n" + "=" * 70)
        print(" " * 15 + "ALL EVALUATIONS COMPLETE!")
        print("=" * 70)
    else:
        print("\n❌ No ensemble configurations were successfully evaluated.")
        print("Please ensure all required model weights exist:")
        print("  - model_weights/snoutnet/best_snoutnet*.pth")
        print("  - model_weights/alexnet/best_alexnet*.pth")
        print("  - model_weights/vgg16/best_vgg16*.pth")


if __name__ == "__main__":
    main()

