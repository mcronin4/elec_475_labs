#########################################################################################################
#
#   ELEC 475 - Lab 3: Semantic Segmentation
#   Fall 2025
#
#   train.py - Training script for compact segmentation model
#

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import time

from model import CompactSegmentationModel
from dataset import VOCSegmentationDataset
from augmentation import get_training_augmentation, get_validation_augmentation
from metrics import compute_iou
from utils import (
    compute_class_weights, save_checkpoint, load_checkpoint,
    AverageMeter, EarlyStopping, plot_training_history,
    visualize_segmentation
)
from dataset import VOC_CLASSES, VOC_COLORMAP


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None, epoch=0):
    """
    Train for one epoch.
    
    Args:
        model: Segmentation model
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        scaler: GradScaler for mixed precision (optional)
        epoch: Current epoch number
        
    Returns:
        float: Average training loss
    """
    model.train()
    
    loss_meter = AverageMeter('Loss')
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} [Train]')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
        
        # Update metrics
        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
    
    return loss_meter.avg


def validate_epoch(model, dataloader, criterion, device, epoch=0):
    """
    Validate for one epoch.
    
    Args:
        model: Segmentation model
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number
        
    Returns:
        tuple: (avg_loss, miou, iou_per_class, class_counts)
    """
    model.eval()
    
    loss_meter = AverageMeter('Val Loss')
    
    # Storage for mIoU computation
    iou_accumulator = torch.zeros(21).to(device)
    class_counts = torch.zeros(21, dtype=torch.long).to(device)
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} [Val]')
    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Update loss
            batch_size = images.size(0)
            loss_meter.update(loss.item(), batch_size)
            
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
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
    
    # Compute mean IoU for each class
    iou_per_class_mean = torch.zeros(21).to(device)
    for class_idx in range(21):
        if class_counts[class_idx] > 0:
            iou_per_class_mean[class_idx] = iou_accumulator[class_idx] / class_counts[class_idx]
    
    # Compute overall mIoU
    valid_ious = iou_per_class_mean[class_counts > 0]
    miou = valid_ious.mean().item() if len(valid_ious) > 0 else 0.0
    
    return loss_meter.avg, miou, iou_per_class_mean.cpu().numpy(), class_counts.cpu().numpy()


def train(args):
    """Main training function."""
    print("=" * 80)
    print("Training Compact Segmentation Model")
    print("=" * 80)
    
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
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    # Load datasets
    print("\n" + "-" * 80)
    print("Loading Datasets")
    print("-" * 80)
    
    train_transform = get_training_augmentation(input_size=(args.input_size, args.input_size))
    val_transform = get_validation_augmentation(input_size=(args.input_size, args.input_size))
    
    train_dataset = VOCSegmentationDataset(
        root=args.data_root,
        image_set='train',
        transform=train_transform,
        use_albumentations=True
    )
    
    val_dataset = VOCSegmentationDataset(
        root=args.data_root,
        image_set='val',
        transform=val_transform,
        use_albumentations=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    print("\n" + "-" * 80)
    print("Creating Model")
    print("-" * 80)
    
    model = CompactSegmentationModel(
        num_classes=21,
        pretrained=args.pretrained,
        dropout=args.dropout
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Compute class weights
    print("\n" + "-" * 80)
    print("Computing Class Weights")
    print("-" * 80)
    
    if args.use_class_weights:
        class_weights = compute_class_weights(train_dataset, num_classes=21, ignore_index=255)
        class_weights = class_weights.to(device)
        print(f"Using class weights (range: {class_weights.min():.3f} - {class_weights.max():.3f})")
    else:
        class_weights = None
        print("Not using class weights")
    
    # Create loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create learning rate scheduler
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.num_epochs,
            eta_min=args.learning_rate * 0.01
        )
    elif args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
    else:
        scheduler = None
    
    # Create gradient scaler for mixed precision
    if args.mixed_precision and device.type == 'cuda':
        try:
            # Try new API first (PyTorch 2.0+)
            from torch.amp import GradScaler
            scaler = GradScaler('cuda')
        except (ImportError, TypeError):
            # Fall back to old API
            from torch.cuda.amp import GradScaler
            scaler = GradScaler()
    else:
        scaler = None
        
    if scaler:
        print(f"Using mixed precision training (FP16)")
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=args.patience,
        min_delta=0.001,
        mode='max'
    ) if args.early_stopping else None
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_miou = 0.0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_miou': []
    }
    
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_miou = checkpoint.get('best_miou', 0.0)
        history = checkpoint.get('history', history)
        print(f"Resuming from epoch {start_epoch}, best mIoU: {best_miou:.4f}")
    
    # Training loop
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)
    
    for epoch in range(start_epoch, args.num_epochs):
        epoch_start_time = time.time()
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler, epoch=epoch
        )
        history['train_loss'].append(train_loss)
        
        # Validate
        if (epoch + 1) % args.val_every == 0:
            val_loss, val_miou, iou_per_class, class_counts = validate_epoch(
                model, val_loader, criterion, device, epoch=epoch
            )
            history['val_loss'].append(val_loss)
            history['val_miou'].append(val_miou)
            
            # Print per-class results
            print(f"\n{'Class':<20} {'IoU':>10} {'Count':>10}")
            print("-" * 42)
            for i, class_name in enumerate(VOC_CLASSES):
                if class_counts[i] > 0:
                    print(f"{class_name:<20} {iou_per_class[i]:>10.4f} {class_counts[i]:>10}")
            print("-" * 42)
            print(f"{'Overall mIoU':<20} {val_miou:>10.4f}")
            
            # Update learning rate scheduler
            if scheduler is not None:
                if args.scheduler == 'plateau':
                    scheduler.step(val_miou)
                else:
                    scheduler.step()
            
            # Save checkpoint
            is_best = val_miou > best_miou
            if is_best:
                best_miou = val_miou
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_miou': val_miou,
                'best_miou': best_miou,
                'history': history,
                'args': vars(args)
            }
            
            # Save latest checkpoint (overwrites previous)
            latest_path = checkpoint_dir / 'checkpoint_latest.pth'
            save_checkpoint(checkpoint, latest_path, is_best=is_best)
            
            # Early stopping check
            if early_stopping is not None:
                if early_stopping(val_miou):
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    break
        
        epoch_time = time.time() - epoch_start_time
        print(f"\nEpoch {epoch+1}/{args.num_epochs} - Time: {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f}")
        if (epoch + 1) % args.val_every == 0:
            print(f"Val Loss: {val_loss:.4f}, Val mIoU: {val_miou:.4f}")
        print("-" * 80)
    
    # Final checkpoint is already saved as checkpoint_latest.pth
    print("\nTraining checkpoints saved.")
    
    # Plot training history
    print("\nGenerating training history plot...")
    plot_path = output_dir / 'training_history.png'
    plot_training_history(history, save_path=plot_path, val_every=args.val_every)
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Best validation mIoU: {best_miou:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"  - Latest checkpoint: {checkpoint_dir / 'checkpoint_latest.pth'}")
    print(f"  - Best checkpoint: {checkpoint_dir / 'best_checkpoint.pth'}")


def main():
    parser = argparse.ArgumentParser(description='Train compact segmentation model')
    
    # Dataset
    parser.add_argument('--data-root', type=str, default='./data',
                       help='Root directory for dataset')
    parser.add_argument('--input-size', type=int, default=512,
                       help='Input image size (default: 512)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Model
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained MobileNetV3 backbone')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate before classifier')
    
    # Training
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=150,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--use-class-weights', action='store_true', default=True,
                       help='Use class weights for loss')
    parser.add_argument('--mixed-precision', action='store_true', default=True,
                       help='Use mixed precision training (FP16)')
    
    # Scheduler
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'plateau', 'none'],
                       help='Learning rate scheduler')
    
    # Validation and checkpointing
    parser.add_argument('--val-every', type=int, default=2,
                       help='Validate every N epochs')
    parser.add_argument('--early-stopping', action='store_true', default=True,
                       help='Use early stopping')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    
    # Resume and output
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--output-dir', type=str, default='./training_output',
                       help='Output directory for checkpoints and logs')
    
    args = parser.parse_args()
    
    # Print configuration
    print("\n" + "=" * 80)
    print("Training Configuration")
    print("=" * 80)
    for arg, value in sorted(vars(args).items()):
        print(f"{arg:.<30} {value}")
    print("=" * 80)
    
    train(args)


if __name__ == '__main__':
    main()

