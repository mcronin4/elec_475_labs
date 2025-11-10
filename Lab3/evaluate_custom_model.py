#########################################################################################################
#
#   ELEC 475 - Lab 3: Semantic Segmentation
#   Fall 2025
#
#   evaluate_custom_model.py - Evaluate trained compact segmentation model
#

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

from model import CompactSegmentationModel
from dataset import VOCSegmentationDataset, VOC_CLASSES, VOC_COLORMAP
from augmentation import get_validation_augmentation
from metrics import compute_iou
from utils import load_checkpoint, visualize_segmentation


def evaluate_model(model, dataloader, device, visualize_samples=True, output_dir='evaluation_results'):
    """
    Evaluate model on validation set.
    
    Args:
        model: Trained segmentation model
        dataloader: Validation dataloader
        device: Device to evaluate on
        visualize_samples (bool): Whether to save sample visualizations
        output_dir (str): Directory to save results
        
    Returns:
        dict: Evaluation results
    """
    model.eval()
    
    # Storage for mIoU computation
    iou_accumulator = np.zeros(21)
    class_counts = np.zeros(21, dtype=int)
    
    # Storage for visualizations
    sample_images = []
    sample_masks = []
    sample_preds = []
    
    print("\nRunning evaluation...")
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(dataloader, desc="Evaluating")):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Get predictions
            preds = torch.argmax(outputs, dim=1)
            
            # Compute IoU for each image in batch
            for pred_mask, true_mask in zip(preds, masks):
                iou_per_class, valid_classes = compute_iou(
                    pred_mask, true_mask, num_classes=21, ignore_index=255
                )
                
                # Accumulate
                for class_idx in range(21):
                    if valid_classes[class_idx]:
                        iou_accumulator[class_idx] += iou_per_class[class_idx]
                        class_counts[class_idx] += 1
            
            # Save samples for visualization
            if visualize_samples and len(sample_images) < 8:
                num_to_save = min(8 - len(sample_images), images.size(0))
                sample_images.extend(images[:num_to_save].cpu())
                sample_masks.extend(masks[:num_to_save].cpu())
                sample_preds.extend(preds[:num_to_save].cpu())
    
    # Compute mean IoU for each class
    iou_per_class_mean = np.zeros(21)
    for class_idx in range(21):
        if class_counts[class_idx] > 0:
            iou_per_class_mean[class_idx] = iou_accumulator[class_idx] / class_counts[class_idx]
        else:
            iou_per_class_mean[class_idx] = np.nan
    
    # Compute overall mIoU
    valid_ious = iou_per_class_mean[~np.isnan(iou_per_class_mean)]
    miou = np.mean(valid_ious) if len(valid_ious) > 0 else 0.0
    
    results = {
        'miou': miou,
        'iou_per_class': iou_per_class_mean,
        'class_counts': class_counts
    }
    
    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"\nDataset: PASCAL VOC 2012 Validation")
    print(f"Number of samples: {len(dataloader.dataset)}")
    print(f"Number of classes: 21")
    
    print(f"\n{'Class':<20} {'IoU':>10} {'Count':>10}")
    print("-" * 42)
    
    for i, class_name in enumerate(VOC_CLASSES):
        if class_counts[i] > 0:
            print(f"{class_name:<20} {iou_per_class_mean[i]:>10.4f} {class_counts[i]:>10}")
        else:
            print(f"{class_name:<20} {'N/A':>10} {class_counts[i]:>10}")
    
    print("-" * 42)
    print(f"\n{'Overall mIoU:':<20} {miou:>10.4f}")
    print("=" * 80)
    
    # Save visualizations
    if visualize_samples and len(sample_images) > 0:
        print(f"\nSaving visualizations to {output_dir}/...")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Stack samples
        sample_images = torch.stack(sample_images)
        sample_masks = torch.stack(sample_masks)
        sample_preds = torch.stack(sample_preds)
        
        visualize_segmentation(
            sample_images,
            sample_masks,
            sample_preds,
            VOC_CLASSES,
            VOC_COLORMAP,
            save_path=output_path / 'segmentation_samples.png',
            num_samples=8
        )
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate compact segmentation model')
    
    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate (should match training)')
    
    # Dataset
    parser.add_argument('--data-root', type=str, default='./data',
                       help='Root directory for dataset')
    parser.add_argument('--image-set', type=str, default='val',
                       choices=['train', 'val', 'trainval'],
                       help='Dataset split to evaluate')
    parser.add_argument('--input-size', type=int, default=320,
                       help='Input image size')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Output
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Save sample visualizations')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Evaluating Compact Segmentation Model")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: PASCAL VOC 2012 {args.image_set}")
    print(f"Input size: {args.input_size}")
    print(f"Batch size: {args.batch_size}")
    
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
    print("\n" + "-" * 80)
    print("Loading Dataset")
    print("-" * 80)
    
    val_transform = get_validation_augmentation(input_size=(args.input_size, args.input_size))
    
    val_dataset = VOCSegmentationDataset(
        root=args.data_root,
        image_set=args.image_set,
        transform=val_transform,
        use_albumentations=True
    )
    
    print(f"Validation samples: {len(val_dataset)}")
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    print("\n" + "-" * 80)
    print("Loading Model")
    print("-" * 80)
    
    model = CompactSegmentationModel(
        num_classes=21,
        pretrained=False,  # We're loading trained weights
        dropout=args.dropout
    )
    
    # Load checkpoint
    checkpoint = load_checkpoint(args.checkpoint, model)
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
    if 'val_miou' in checkpoint:
        print(f"Checkpoint validation mIoU: {checkpoint['val_miou']:.4f}")
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Evaluate
    results = evaluate_model(
        model,
        val_loader,
        device,
        visualize_samples=args.visualize,
        output_dir=args.output_dir
    )
    
    # Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / 'evaluation_results.txt'
    with open(results_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("EVALUATION RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"\nCheckpoint: {args.checkpoint}\n")
        f.write(f"Dataset: PASCAL VOC 2012 {args.image_set}\n")
        f.write(f"Number of samples: {len(val_dataset)}\n")
        f.write(f"Model parameters: {total_params:,}\n")
        f.write(f"\n{'Class':<20} {'IoU':>10} {'Count':>10}\n")
        f.write("-" * 42 + "\n")
        for i, class_name in enumerate(VOC_CLASSES):
            if results['class_counts'][i] > 0:
                f.write(f"{class_name:<20} {results['iou_per_class'][i]:>10.4f} {results['class_counts'][i]:>10}\n")
        f.write("-" * 42 + "\n")
        f.write(f"\n{'Overall mIoU:':<20} {results['miou']:>10.4f}\n")
        f.write("=" * 80 + "\n")
    
    print(f"\nResults saved to: {results_file}")
    
    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()




