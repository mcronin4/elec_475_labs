import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import argparse

from model import SnoutNet, init_weights
from dataset import PetNoseDataset, get_transforms


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: SnoutNet model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (cuda/cpu)
        
    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    running_loss = 0.0
    num_batches = 0
    
    for images, coordinates in tqdm(train_loader, desc="Training", leave=False):
        # Move data to device
        images = images.to(device)
        coordinates = coordinates.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Compute loss
        loss = criterion(outputs, coordinates)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Track loss
        running_loss += loss.item()
        num_batches += 1
    
    # Return average loss
    avg_loss = running_loss / num_batches
    return avg_loss


def validate(model, test_loader, criterion, device):
    """
    Validate the model on test set.
    
    Args:
        model: SnoutNet model
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to evaluate on (cuda/cpu)
        
    Returns:
        float: Average validation loss
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, coordinates in tqdm(test_loader, desc="Validation", leave=False):
            # Move data to device
            images = images.to(device)
            coordinates = coordinates.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = criterion(outputs, coordinates)
            
            # Track loss
            running_loss += loss.item()
            num_batches += 1
    
    # Return average loss
    avg_loss = running_loss / num_batches
    return avg_loss


def plot_losses(train_losses, test_losses, save_path="loss_plots/training_loss.png"):
    """
    Plot training and validation losses.
    
    Args:
        train_losses: List of training losses per epoch
        test_losses: List of validation losses per epoch
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, test_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('SnoutNet Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nLoss plot saved to: {save_path}")
    plt.close()


def train_snoutnet(num_epochs=50, batch_size=32, learning_rate=0.001, 
                   color_aug=False, blur_aug=False,
                   save_model_path="model_weights/snoutnet/snoutnet.pth", 
                   best_model_path="model_weights/snoutnet/best_snoutnet.pth", 
                   save_plot_path="loss_plots/training_loss.png"):
    """
    Main training function for SnoutNet.
    
    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        color_aug: Whether to use color augmentation (ColorJitter)
        blur_aug: Whether to use blur augmentation (GaussianBlur)
        save_model_path: Path to save trained model
        save_plot_path: Path to save loss plot
    """
    print("=" * 60)
    print("SnoutNet Training Script")
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
    
    # Dataset paths
    images_dir = "oxford-iiit-pet-noses/images-original/images"
    train_labels = "oxford-iiit-pet-noses/train_noses.txt"
    test_labels = "oxford-iiit-pet-noses/test_noses.txt"
    
    print("\n" + "-" * 60)
    print("Loading Datasets...")
    print("-" * 60)
    
    # Get transforms
    train_transform = get_transforms(resize_size=227, color_aug=color_aug, blur_aug=blur_aug)
    test_transform = get_transforms(resize_size=227, color_aug=False, blur_aug=False)  # no augmentation for test set
    
    # Print augmentation info
    aug_types = []
    if color_aug:
        aug_types.append("ColorJitter")
    if blur_aug:
        aug_types.append("GaussianBlur")
    
    if aug_types:
        print(f"Using data augmentation: {', '.join(aug_types)}")
    else:
        print("Using basic transforms (no augmentation)")
    
    # Create datasets
    train_dataset = PetNoseDataset(images_dir, train_labels, transform=train_transform)
    test_dataset = PetNoseDataset(images_dir, test_labels, transform=test_transform)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    print("\n" + "-" * 60)
    print("Initializing Model...")
    print("-" * 60)
    
    # Create model
    model = SnoutNet()
    model.apply(init_weights)
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Loss function (MSE for regression)
    criterion = nn.MSELoss()
    print(f"Loss function: MSE Loss (Mean Squared Error)")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print(f"Optimizer: Adam (lr={learning_rate})")
    
    print("\n" + "-" * 60)
    print(f"Training for {num_epochs} epochs...")
    print("-" * 60)
    
    # Training history
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    best_epoch = 0
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        
        # Train for one epoch
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        test_loss = validate(model, test_loader, criterion, device)
        
        # Store losses
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        # Print statistics
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Test Loss:  {test_loss:.4f}")
        
        # Track best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch
            # Save best model
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ New best model saved! (Test Loss: {best_test_loss:.4f})")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best test loss: {best_test_loss:.4f} at epoch {best_epoch}")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Final test loss: {test_losses[-1]:.4f}")
    
    # Save final model
    torch.save(model.state_dict(), save_model_path)
    print(f"\nFinal model saved to: {save_model_path}")
    print(f"Best model saved to: best_{save_model_path}")
    
    # Plot losses
    plot_losses(train_losses, test_losses, save_path=save_plot_path)
    
    return model, train_losses, test_losses


def main():
    """Main function to handle command line arguments and start training."""
    parser = argparse.ArgumentParser(description='Train SnoutNet model for pet nose localization')
    parser.add_argument('--color-aug', action='store_true',
                       help='Enable color augmentation (ColorJitter: brightness, contrast, saturation, hue)')
    parser.add_argument('--blur-aug', action='store_true',
                       help='Enable blur augmentation (GaussianBlur)')
    parser.add_argument('-e', '--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('-b', '--batch-size', type=int, default=32,
                       help='Batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    
    args = parser.parse_args()
    
    # Set filenames based on augmentation flags
    suffix = ""
    if args.color_aug and args.blur_aug:
        suffix = "_aug_both"
    elif args.color_aug:
        suffix = "_aug_color"
    elif args.blur_aug:
        suffix = "_aug_blur"
    
    model_path = f"model_weights/snoutnet/snoutnet{suffix}.pth"
    best_model_path = f"model_weights/snoutnet/best_snoutnet{suffix}.pth"
    plot_path = f"loss_plots/training_loss{suffix}.png"
    
    print("\n" + "=" * 60)
    print("Training Configuration:")
    print("=" * 60)
    aug_info = []
    if args.color_aug:
        aug_info.append("Color")
    if args.blur_aug:
        aug_info.append("Blur")
    print(f"Augmentation: {', '.join(aug_info) if aug_info else 'Disabled'}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Output files:")
    print(f"  - Final model: {model_path}")
    print(f"  - Best model: {best_model_path}")
    print(f"  - Loss plot: {plot_path}")
    
    # Train the model
    model, train_losses, test_losses = train_snoutnet(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        color_aug=args.color_aug,
        blur_aug=args.blur_aug,
        save_model_path=model_path,
        best_model_path=best_model_path,
        save_plot_path=plot_path
    )
    
    print("\n" + "=" * 60)
    print("Training script finished successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

