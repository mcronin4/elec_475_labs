#########################################################################################################
#
#   ELEC 475 - Lab 3: Semantic Segmentation
#   Fall 2025
#
#   utils.py - Utility functions for training and evaluation
#

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path


def compute_class_weights(dataset, num_classes=21, ignore_index=255):
    """
    Compute class weights based on inverse frequency.
    
    Args:
        dataset: PyTorch dataset with masks
        num_classes (int): Number of classes
        ignore_index (int): Index to ignore
        
    Returns:
        torch.Tensor: Class weights for CrossEntropyLoss
    """
    print("Computing class weights from training set...")
    
    class_counts = np.zeros(num_classes, dtype=np.int64)
    
    # Count pixels per class
    for idx in tqdm(range(len(dataset)), desc="Scanning dataset"):
        _, mask = dataset[idx]
        
        # Convert to numpy if tensor
        if torch.is_tensor(mask):
            mask = mask.numpy()
        
        # Count each class (excluding ignore_index)
        for class_idx in range(num_classes):
            class_counts[class_idx] += np.sum(mask == class_idx)
    
    # Compute inverse frequency weights
    # Add small epsilon to avoid division by zero
    total_pixels = class_counts.sum()
    class_weights = total_pixels / (num_classes * (class_counts + 1e-6))
    
    # Normalize so mean weight is 1.0
    class_weights = class_weights / class_weights.mean()
    
    # Cap maximum weight to avoid extreme values
    class_weights = np.clip(class_weights, 0.1, 10.0)
    
    # Print statistics
    print("\nClass weight statistics:")
    print(f"  Min weight: {class_weights.min():.3f}")
    print(f"  Max weight: {class_weights.max():.3f}")
    print(f"  Mean weight: {class_weights.mean():.3f}")
    
    return torch.from_numpy(class_weights).float()


def save_checkpoint(state, filepath, is_best=False):
    """
    Save training checkpoint.
    
    Args:
        state (dict): Checkpoint state
        filepath (str): Path to save checkpoint
        is_best (bool): Whether this is the best model so far
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(state, filepath)
    
    if is_best:
        best_path = filepath.parent / "best_checkpoint.pth"
        torch.save(state, best_path)
        print(f"✓ Saved best checkpoint: {best_path}")


def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    """
    Load training checkpoint.
    
    Args:
        filepath (str): Path to checkpoint
        model (nn.Module): Model to load weights into
        optimizer (Optimizer): Optimizer to load state (optional)
        scheduler: Learning rate scheduler to load state (optional)
        
    Returns:
        dict: Checkpoint state
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name=''):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving."""
    
    def __init__(self, patience=10, min_delta=0.0, mode='max'):
        """
        Args:
            patience (int): How many epochs to wait after last improvement
            min_delta (float): Minimum change to qualify as improvement
            mode (str): 'max' or 'min' - whether higher or lower is better
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        # Check if score improved
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


def plot_training_history(history, save_path='training_history.png', val_every=1):
    """
    Plot training history.
    
    Args:
        history (dict): Training history with keys: 'train_loss', 'val_loss', 'val_miou'
        save_path (str): Path to save plot
        val_every (int): Validation frequency (epochs between validations)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Epoch indices for training (every epoch)
    train_epochs = list(range(1, len(history['train_loss']) + 1))
    
    # Epoch indices for validation (every val_every epochs)
    val_epochs = list(range(val_every, val_every * len(history.get('val_loss', [])) + 1, val_every))
    
    # Plot loss
    axes[0].plot(train_epochs, history['train_loss'], label='Train Loss', marker='o', markersize=2)
    if 'val_loss' in history and len(history['val_loss']) > 0:
        axes[0].plot(val_epochs, history['val_loss'], label='Val Loss', marker='s', markersize=3)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot mIoU
    if 'val_miou' in history and len(history['val_miou']) > 0:
        axes[1].plot(val_epochs, history['val_miou'], label='Val mIoU', color='green', marker='s', markersize=3)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('mIoU')
        axes[1].set_title('Validation mIoU')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved training history plot: {save_path}")
    plt.close()


def denormalize_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize image from ImageNet normalization.
    
    Args:
        image (torch.Tensor): Normalized image [C, H, W]
        mean (list): Mean used for normalization
        std (list): Std used for normalization
        
    Returns:
        torch.Tensor: Denormalized image [C, H, W] in range [0, 1]
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    
    image = image * std + mean
    image = torch.clamp(image, 0, 1)
    
    return image


def visualize_segmentation(images, masks, predictions, class_names, 
                          colormap, save_path='segmentation_samples.png', 
                          num_samples=4):
    """
    Visualize segmentation results.
    
    Args:
        images (torch.Tensor): Images [B, 3, H, W]
        masks (torch.Tensor): Ground truth masks [B, H, W]
        predictions (torch.Tensor): Predicted masks [B, H, W]
        class_names (list): List of class names
        colormap (list): List of RGB colors for each class
        save_path (str): Path to save visualization
        num_samples (int): Number of samples to visualize
    """
    from dataset import decode_segmentation_mask
    
    num_samples = min(num_samples, images.shape[0])
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Denormalize image
        image = denormalize_image(images[i].cpu())
        image_np = image.permute(1, 2, 0).numpy()
        
        # Decode masks
        mask_rgb = decode_segmentation_mask(masks[i].cpu(), colormap)
        pred_rgb = decode_segmentation_mask(predictions[i].cpu(), colormap)
        
        # Plot
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask_rgb)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred_rgb)
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved segmentation visualization: {save_path}")
    plt.close()


if __name__ == "__main__":
    """Test utility functions."""
    print("=" * 80)
    print("Testing Utility Functions")
    print("=" * 80)
    
    # Test AverageMeter
    print("\nTesting AverageMeter:")
    meter = AverageMeter('Loss')
    for i in range(10):
        meter.update(np.random.random())
    print(f"  {meter}")
    
    # Test EarlyStopping
    print("\nTesting EarlyStopping:")
    early_stop = EarlyStopping(patience=3, mode='max')
    scores = [0.5, 0.6, 0.65, 0.64, 0.63, 0.62]
    for i, score in enumerate(scores):
        should_stop = early_stop(score)
        print(f"  Epoch {i+1}: score={score:.2f}, counter={early_stop.counter}, stop={should_stop}")
    
    print("\n" + "=" * 80)
    print("Utility functions test complete!")
    print("=" * 80)

