#########################################################################################################
#
#   ELEC 475 - Lab 3: Semantic Segmentation
#   Fall 2025
#
#   evaluate.py - Evaluate FCN-ResNet50 on PASCAL VOC 2012 with mIoU metric
#

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models.segmentation as segmentation_models
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse
import os

from dataset import load_voc_dataset, decode_segmentation_mask, VOC_CLASSES
from metrics import compute_miou, compute_iou


def load_fcn_resnet50(pretrained=True, device='cpu'):
    """
    Load pretrained FCN-ResNet50 model.
    
    Args:
        pretrained (bool): Whether to load pretrained weights
        device (str): Device to load model on
        
    Returns:
        nn.Module: FCN-ResNet50 model
    """
    print("Loading FCN-ResNet50 model...")
    
    # Load pretrained model
    model = segmentation_models.fcn_resnet50(pretrained=pretrained)
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded with {num_params:,} parameters")
    
    return model


def evaluate_model(model, dataset, device='cpu', batch_size=4, num_samples=None, 
                   visualize=False, save_dir='evaluation_results'):
    """
    Evaluate FCN-ResNet50 model on PASCAL VOC dataset.
    
    Args:
        model (nn.Module): FCN-ResNet50 model
        dataset: PASCAL VOC segmentation dataset
        device (str): Device to run evaluation on
        batch_size (int): Batch size for evaluation
        num_samples (int): Number of samples to evaluate (None = all)
        visualize (bool): Whether to save visualizations
        save_dir (str): Directory to save visualizations
        
    Returns:
        dict: Evaluation results with mIoU and per-class statistics
    """
    print("\n" + "=" * 60)
    print("Starting Evaluation")
    print("=" * 60)
    
    # Get subset if requested
    if num_samples is not None:
        dataset = dataset.get_subset(num_samples)
        print(f"Evaluating on {len(dataset)} samples (subset)")
    else:
        print(f"Evaluating on {len(dataset)} samples (full dataset)")
    
    # Create data loader
    # Note: batch_size=1 is safer for variable-size images
    # FCN can handle different input sizes, but batching requires same size
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, 
                            num_workers=2, pin_memory=True)
    
    # Storage for per-image predictions and ground truth
    # Note: We can't concatenate because images have variable sizes
    all_predictions = []
    all_ground_truth = []
    
    # For visualization
    sample_images = []
    sample_preds = []
    sample_gts = []
    
    print("\nRunning inference...")
    model.eval()
    with torch.no_grad():
        for idx, (images, masks) in enumerate(tqdm(data_loader, desc="Evaluating")):
            # Move to device
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # FCN returns a dictionary with 'out' key containing logits [B, 21, H, W]
            logits = outputs['out']
            
            # Get predictions by taking argmax over classes
            preds = torch.argmax(logits, dim=1)  # [B, H, W]
            
            # Remove batch dimension and store (move to CPU)
            # Since batch_size=1, we can safely squeeze or index
            pred_single = preds[0].cpu()  # [H, W]
            mask_single = masks[0].cpu()  # [H, W]
            
            all_predictions.append(pred_single)
            all_ground_truth.append(mask_single)
            
            # Save samples for visualization (first 6)
            if visualize and len(sample_images) < 6:
                sample_images.append(images[0].cpu())
                sample_preds.append(pred_single)
                sample_gts.append(mask_single)
    
    # Note: We can't concatenate predictions because VOC images have variable sizes
    print(f"\nNumber of samples: {len(all_predictions)}")
    if len(all_predictions) > 0:
        print(f"Sample prediction shape: {all_predictions[0].shape}")
        print(f"Sample ground truth shape: {all_ground_truth[0].shape}")
    
    # Compute mIoU
    print("\n" + "-" * 60)
    print("Computing mIoU...")
    print("-" * 60)
    
    # Process each image individually and accumulate IoU
    # This handles variable image sizes correctly
    iou_accumulator = np.zeros(21)
    class_counts = np.zeros(21, dtype=int)
    
    for pred_mask, true_mask in zip(all_predictions, all_ground_truth):
        iou_per_class, valid_classes = compute_iou(
            pred_mask, true_mask, num_classes=21, ignore_index=255
        )
        for class_idx in range(21):
            if valid_classes[class_idx]:
                iou_accumulator[class_idx] += iou_per_class[class_idx]
                class_counts[class_idx] += 1
    
    # Compute mean IoU for each class
    iou_per_class_mean = np.zeros(21)
    for class_idx in range(21):
        if class_counts[class_idx] > 0:
            iou_per_class_mean[class_idx] = iou_accumulator[class_idx] / class_counts[class_idx]
        else:
            iou_per_class_mean[class_idx] = np.nan
    
    # Compute overall mIoU
    valid_class_ious = iou_per_class_mean[~np.isnan(iou_per_class_mean)]
    if len(valid_class_ious) > 0:
        miou = np.mean(valid_class_ious)
    else:
        miou = 0.0
    
    results = {
        'miou': miou,
        'iou_per_class': iou_per_class_mean,
        'class_counts': class_counts
    }
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nDataset: PASCAL VOC 2012")
    print(f"Number of samples: {len(dataset)}")
    print(f"Number of classes: 21")
    
    print(f"\n{'Class':<20} {'IoU':>10} {'Count':>10}")
    print("-" * 42)
    
    iou_per_class = results['iou_per_class']
    class_counts = results['class_counts']
    
    for i, class_name in enumerate(VOC_CLASSES):
        if class_counts[i] > 0:
            print(f"{class_name:<20} {iou_per_class[i]:>10.4f} {class_counts[i]:>10}")
        else:
            print(f"{class_name:<20} {'N/A':>10} {class_counts[i]:>10}")
    
    print("-" * 42)
    print(f"\n{'Overall mIoU:':<20} {results['miou']:>10.4f}")
    print("=" * 60)
    
    # Visualizations
    if visualize and len(sample_images) > 0:
        print(f"\nSaving visualizations to {save_dir}/")
        os.makedirs(save_dir, exist_ok=True)
        visualize_predictions(
            sample_images, sample_preds, sample_gts,
            save_path=os.path.join(save_dir, 'predictions.png')
        )
    
    return results


def visualize_predictions(images, predictions, ground_truths, save_path='predictions.png'):
    """
    Visualize segmentation predictions.
    
    Args:
        images (list): List of image tensors [3, H, W]
        predictions (list): List of prediction masks [H, W]
        ground_truths (list): List of ground truth masks [H, W]
        save_path (str): Path to save visualization
    """
    num_samples = len(images)
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Denormalize image
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = images[i] * std + mean
        image = torch.clamp(image, 0, 1)
        image_np = image.permute(1, 2, 0).numpy()
        
        # Convert masks to RGB
        pred_rgb = decode_segmentation_mask(predictions[i])
        gt_rgb = decode_segmentation_mask(ground_truths[i])
        
        # Plot
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(gt_rgb)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred_rgb)
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    plt.close()


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Evaluate FCN-ResNet50 on PASCAL VOC 2012 segmentation'
    )
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Number of samples to evaluate (default: None = full dataset)')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for evaluation (default: 4, but forced to 1 for variable sizes)')
    parser.add_argument('--visualize', action='store_true',
                       help='Save visualization of sample predictions')
    parser.add_argument('--save-dir', type=str, default='evaluation_results',
                       help='Directory to save results (default: evaluation_results)')
    parser.add_argument('--data-root', type=str, default='./data',
                       help='Root directory for dataset (default: ./data)')
    parser.add_argument('--image-set', type=str, default='val',
                       choices=['train', 'val', 'trainval', 'test'],
                       help='Dataset split to evaluate (test split lacks masks)')
    
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 60)
    print("FCN-ResNet50 Evaluation Configuration")
    print("=" * 60)
    print(f"Dataset: PASCAL VOC 2012")
    print(f"Data root: {args.data_root}")
    print(f"Image set: {args.image_set}")
    print(f"Num samples: {args.num_samples if args.num_samples else 'All'}")
    print(f"Batch size: {args.batch_size} (forced to 1 for variable-size images)")
    print(f"Visualize: {args.visualize}")
    print(f"Save directory: {args.save_dir}")
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Device: Apple Metal (MPS)")
    else:
        device = torch.device('cpu')
        print(f"Device: CPU")
    
    # Load dataset
    print("\n" + "-" * 60)
    print("Loading Dataset")
    print("-" * 60)
    if args.image_set == 'test':
        raise ValueError(
            "The PASCAL VOC 2012 test split does not include ground-truth segmentation masks.\n"
            "Choose from --image-set train, val, or trainval."
        )
    
    dataset = load_voc_dataset(root=args.data_root, image_set=args.image_set)
    
    # Load model
    print("\n" + "-" * 60)
    print("Loading Model")
    print("-" * 60)
    model = load_fcn_resnet50(pretrained=True, device=device)
    
    # Evaluate
    results = evaluate_model(
        model=model,
        dataset=dataset,
        device=device,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        visualize=args.visualize,
        save_dir=args.save_dir
    )
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

