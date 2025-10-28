#########################################################################################################
#
#   ELEC 475 - Lab 3: Semantic Segmentation
#   Fall 2025
#
#   metrics.py - Mean Intersection over Union (mIoU) metric for semantic segmentation
#

import torch
import numpy as np


def compute_iou(pred_mask, true_mask, num_classes=21, ignore_index=255):
    """
    Compute Intersection over Union (IoU) for a single image.
    
    IoU for class c = Intersection(c) / Union(c)
                    = TP / (TP + FP + FN)
    
    Args:
        pred_mask (torch.Tensor): [H, W] predicted segmentation mask
        true_mask (torch.Tensor): [H, W] ground truth segmentation mask
        num_classes (int): Number of classes (21 for PASCAL VOC)
        ignore_index (int): Index to ignore (255 for PASCAL VOC boundaries)
        
    Returns:
        tuple: (iou_per_class, valid_classes)
            - iou_per_class: [num_classes] IoU for each class (nan if class not present)
            - valid_classes: [num_classes] boolean mask of classes present in GT
    """
    # Ensure inputs are on CPU and are numpy arrays
    if torch.is_tensor(pred_mask):
        pred_mask = pred_mask.cpu().numpy()
    if torch.is_tensor(true_mask):
        true_mask = true_mask.cpu().numpy()
    
    # Create mask to ignore boundary pixels (255)
    valid_mask = (true_mask != ignore_index)
    
    # Apply ignore mask
    pred_valid = pred_mask[valid_mask]
    true_valid = true_mask[valid_mask]
    
    # Initialize IoU array
    iou_per_class = np.full(num_classes, np.nan)
    valid_classes = np.zeros(num_classes, dtype=bool)
    
    # Compute IoU for each class
    for class_idx in range(num_classes):
        # Check if class is present in ground truth
        gt_mask = (true_valid == class_idx)
        pred_mask_c = (pred_valid == class_idx)
        
        if not gt_mask.any():
            # Class not present in ground truth
            continue
        
        valid_classes[class_idx] = True
        
        # Compute intersection and union
        intersection = np.logical_and(gt_mask, pred_mask_c).sum()
        union = np.logical_or(gt_mask, pred_mask_c).sum()
        
        # Compute IoU (handle division by zero)
        if union > 0:
            iou_per_class[class_idx] = intersection / union
        else:
            iou_per_class[class_idx] = 0.0
    
    return iou_per_class, valid_classes


def compute_miou(pred_masks, true_masks, num_classes=21, ignore_index=255, 
                 return_per_class=False):
    """
    Compute mean Intersection over Union (mIoU) across multiple images.
    
    The mIoU is computed by:
    1. Computing IoU for each class in each image
    2. Averaging IoU for each class across all images where it appears
    3. Computing mean across all classes
    
    Args:
        pred_masks (torch.Tensor): [N, H, W] predicted segmentation masks
        true_masks (torch.Tensor): [N, H, W] ground truth segmentation masks
        num_classes (int): Number of classes (21 for PASCAL VOC)
        ignore_index (int): Index to ignore (255 for PASCAL VOC boundaries)
        return_per_class (bool): Whether to return per-class IoU statistics
        
    Returns:
        dict: Dictionary containing:
            - 'miou': Mean IoU across all classes
            - 'iou_per_class': [num_classes] mean IoU for each class (if return_per_class=True)
            - 'class_counts': [num_classes] number of images each class appears in
    """
    num_images = pred_masks.shape[0]
    
    # Storage for per-class IoU across all images
    iou_accumulator = np.zeros(num_classes)
    class_counts = np.zeros(num_classes, dtype=int)
    
    # Process each image
    for i in range(num_images):
        pred_mask = pred_masks[i]
        true_mask = true_masks[i]
        
        # Compute IoU for this image
        iou_per_class, valid_classes = compute_iou(
            pred_mask, true_mask, num_classes, ignore_index
        )
        
        # Accumulate IoU for classes present in this image
        for class_idx in range(num_classes):
            if valid_classes[class_idx]:
                iou_accumulator[class_idx] += iou_per_class[class_idx]
                class_counts[class_idx] += 1
    
    # Compute mean IoU for each class (across images where class appears)
    iou_per_class_mean = np.zeros(num_classes)
    for class_idx in range(num_classes):
        if class_counts[class_idx] > 0:
            iou_per_class_mean[class_idx] = iou_accumulator[class_idx] / class_counts[class_idx]
        else:
            iou_per_class_mean[class_idx] = np.nan
    
    # Compute overall mIoU (mean across classes that appear at least once)
    valid_class_ious = iou_per_class_mean[~np.isnan(iou_per_class_mean)]
    if len(valid_class_ious) > 0:
        miou = np.mean(valid_class_ious)
    else:
        miou = 0.0
    
    results = {
        'miou': miou,
        'class_counts': class_counts
    }
    
    if return_per_class:
        results['iou_per_class'] = iou_per_class_mean
    
    return results


def compute_confusion_matrix(pred_masks, true_masks, num_classes=21, ignore_index=255):
    """
    Compute confusion matrix for segmentation predictions.
    
    Args:
        pred_masks (torch.Tensor): [N, H, W] predicted segmentation masks
        true_masks (torch.Tensor): [N, H, W] ground truth segmentation masks
        num_classes (int): Number of classes
        ignore_index (int): Index to ignore
        
    Returns:
        np.ndarray: [num_classes, num_classes] confusion matrix
    """
    # Ensure numpy arrays
    if torch.is_tensor(pred_masks):
        pred_masks = pred_masks.cpu().numpy()
    if torch.is_tensor(true_masks):
        true_masks = true_masks.cpu().numpy()
    
    # Flatten arrays
    pred_flat = pred_masks.flatten()
    true_flat = true_masks.flatten()
    
    # Create valid mask (ignore boundary pixels)
    valid_mask = (true_flat != ignore_index)
    pred_valid = pred_flat[valid_mask]
    true_valid = true_flat[valid_mask]
    
    # Compute confusion matrix
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_class in range(num_classes):
        for pred_class in range(num_classes):
            confusion[true_class, pred_class] = np.sum(
                (true_valid == true_class) & (pred_valid == pred_class)
            )
    
    return confusion


def iou_from_confusion_matrix(confusion_matrix):
    """
    Compute IoU per class from confusion matrix.
    
    Args:
        confusion_matrix (np.ndarray): [num_classes, num_classes] confusion matrix
        
    Returns:
        np.ndarray: [num_classes] IoU for each class
    """
    num_classes = confusion_matrix.shape[0]
    iou_per_class = np.zeros(num_classes)
    
    for class_idx in range(num_classes):
        # True positives: diagonal element
        tp = confusion_matrix[class_idx, class_idx]
        
        # False positives: sum of column excluding diagonal
        fp = np.sum(confusion_matrix[:, class_idx]) - tp
        
        # False negatives: sum of row excluding diagonal
        fn = np.sum(confusion_matrix[class_idx, :]) - tp
        
        # Compute IoU
        union = tp + fp + fn
        if union > 0:
            iou_per_class[class_idx] = tp / union
        else:
            iou_per_class[class_idx] = np.nan
    
    return iou_per_class


if __name__ == "__main__":
    """Test mIoU computation with synthetic data."""
    print("Testing mIoU Metric Implementation...")
    print("=" * 60)
    
    num_classes = 21
    H, W = 100, 100
    
    # Test 1: Perfect predictions (IoU = 1.0)
    print("\nTest 1: Perfect predictions")
    true_mask = torch.randint(0, num_classes, (H, W))
    pred_mask = true_mask.clone()
    iou, valid = compute_iou(pred_mask, true_mask, num_classes)
    valid_ious = iou[~np.isnan(iou)]
    print(f"  Mean IoU: {np.mean(valid_ious):.4f} (expected: 1.0000)")
    print(f"  All IoUs = 1.0: {np.allclose(valid_ious, 1.0)}")
    
    # Test 2: No overlap (IoU = 0.0)
    print("\nTest 2: No overlap predictions")
    true_mask = torch.zeros((H, W), dtype=torch.long)  # All class 0
    pred_mask = torch.ones((H, W), dtype=torch.long)   # All class 1
    iou, valid = compute_iou(pred_mask, true_mask, num_classes)
    print(f"  IoU for class 0: {iou[0]:.4f} (expected: 0.0000)")
    print(f"  Class 0 IoU = 0.0: {np.isclose(iou[0], 0.0)}")
    
    # Test 3: Partial overlap (IoU = 0.5)
    print("\nTest 3: Partial overlap (50%)")
    true_mask = torch.zeros((H, W), dtype=torch.long)
    true_mask[:, W//2:] = 1
    pred_mask = torch.zeros((H, W), dtype=torch.long)
    pred_mask[H//2:, :] = 1
    iou, valid = compute_iou(pred_mask, true_mask, num_classes)
    print(f"  IoU for class 0: {iou[0]:.4f}")
    print(f"  IoU for class 1: {iou[1]:.4f}")
    
    # Test 4: Ignore index handling
    print("\nTest 4: Ignore index (255) handling")
    true_mask = torch.zeros((H, W), dtype=torch.long)
    true_mask[:, :W//2] = 255  # Mark half as ignore
    pred_mask = torch.zeros((H, W), dtype=torch.long)
    iou, valid = compute_iou(pred_mask, true_mask, num_classes)
    print(f"  IoU for class 0: {iou[0]:.4f} (should ignore pixels marked 255)")
    print(f"  Class 0 is valid: {valid[0]}")
    
    # Test 5: Multiple images mIoU
    print("\nTest 5: Multiple images mIoU")
    N = 5
    pred_masks = torch.randint(0, num_classes, (N, H, W))
    true_masks = pred_masks.clone()
    results = compute_miou(pred_masks, true_masks, num_classes, return_per_class=True)
    print(f"  Mean IoU: {results['miou']:.4f} (expected: 1.0000)")
    print(f"  mIoU = 1.0: {np.isclose(results['miou'], 1.0)}")
    
    print("\n" + "=" * 60)
    print("Metric tests complete!")

