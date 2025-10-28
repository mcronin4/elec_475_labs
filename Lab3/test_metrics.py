#########################################################################################################
#
#   ELEC 475 - Lab 3: Semantic Segmentation
#   Fall 2025
#
#   test_metrics.py - Verify mIoU metric calculation
#

import torch
import numpy as np
from metrics import compute_iou, compute_miou, compute_confusion_matrix, iou_from_confusion_matrix
import sys


def test_perfect_predictions():
    """Test IoU with perfect predictions (should be 1.0)."""
    print("\n" + "=" * 60)
    print("TEST 1: Perfect Predictions (IoU = 1.0)")
    print("=" * 60)
    
    try:
        num_classes = 21
        H, W = 100, 100
        
        # Create a random mask and use it for both prediction and ground truth
        true_mask = torch.randint(0, num_classes, (H, W))
        pred_mask = true_mask.clone()
        
        # Compute IoU
        iou, valid = compute_iou(pred_mask, true_mask, num_classes)
        
        # All valid classes should have IoU = 1.0
        valid_ious = iou[~np.isnan(iou)]
        
        print(f"Number of classes present: {len(valid_ious)}")
        print(f"IoU values: min={valid_ious.min():.4f}, max={valid_ious.max():.4f}, mean={valid_ious.mean():.4f}")
        
        assert np.allclose(valid_ious, 1.0), \
            f"Expected all IoUs to be 1.0, got min={valid_ious.min()}, max={valid_ious.max()}"
        print("✓ All classes have IoU = 1.0")
        
        print("\n✓ TEST 1 PASSED: Perfect predictions work correctly")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 1 FAILED: {str(e)}")
        return False


def test_no_overlap():
    """Test IoU with no overlap (should be 0.0)."""
    print("\n" + "=" * 60)
    print("TEST 2: No Overlap (IoU = 0.0)")
    print("=" * 60)
    
    try:
        num_classes = 21
        H, W = 100, 100
        
        # Create masks with no overlap: GT is all class 0, prediction is all class 1
        true_mask = torch.zeros((H, W), dtype=torch.long)
        pred_mask = torch.ones((H, W), dtype=torch.long)
        
        # Compute IoU
        iou, valid = compute_iou(pred_mask, true_mask, num_classes)
        
        # Class 0 should have IoU = 0.0 (no correct predictions)
        # Class 1 should not be in valid classes (not in GT)
        print(f"IoU for class 0: {iou[0]:.4f}")
        print(f"Class 0 is valid: {valid[0]}")
        print(f"Class 1 is valid: {valid[1]}")
        
        assert valid[0], "Class 0 should be valid (present in GT)"
        assert not valid[1], "Class 1 should not be valid (not in GT)"
        assert np.isclose(iou[0], 0.0), f"Expected IoU=0.0 for class 0, got {iou[0]}"
        print("✓ Class 0 has IoU = 0.0 (no overlap)")
        
        print("\n✓ TEST 2 PASSED: No overlap case works correctly")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 2 FAILED: {str(e)}")
        return False


def test_partial_overlap():
    """Test IoU with partial overlap."""
    print("\n" + "=" * 60)
    print("TEST 3: Partial Overlap")
    print("=" * 60)
    
    try:
        num_classes = 21
        H, W = 100, 100
        
        # Create masks with 25% overlap
        # GT: left half is class 0, right half is class 1
        true_mask = torch.zeros((H, W), dtype=torch.long)
        true_mask[:, W//2:] = 1
        
        # Pred: top half is class 0, bottom half is class 1
        pred_mask = torch.zeros((H, W), dtype=torch.long)
        pred_mask[H//2:, :] = 1
        
        # Compute IoU
        iou, valid = compute_iou(pred_mask, true_mask, num_classes)
        
        print(f"IoU for class 0: {iou[0]:.4f}")
        print(f"IoU for class 1: {iou[1]:.4f}")
        
        # Both classes should be valid
        assert valid[0] and valid[1], "Both classes should be valid"
        
        # IoU should be 1/3 for both classes (overlap is 1/4 of each class, union is 3/4)
        # Intersection: H/2 * W/2 = HW/4
        # Union: H*W/2 + H*W/2 - HW/4 = 3HW/4
        # IoU = (HW/4) / (3HW/4) = 1/3
        expected_iou = 1.0 / 3.0
        assert np.isclose(iou[0], expected_iou, atol=0.01), \
            f"Expected IoU≈{expected_iou:.4f} for class 0, got {iou[0]:.4f}"
        assert np.isclose(iou[1], expected_iou, atol=0.01), \
            f"Expected IoU≈{expected_iou:.4f} for class 1, got {iou[1]:.4f}"
        print(f"✓ Both classes have IoU ≈ {expected_iou:.4f} (partial overlap)")
        
        print("\n✓ TEST 3 PASSED: Partial overlap case works correctly")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 3 FAILED: {str(e)}")
        return False


def test_ignore_index():
    """Test that ignore index (255) is properly handled."""
    print("\n" + "=" * 60)
    print("TEST 4: Ignore Index Handling")
    print("=" * 60)
    
    try:
        num_classes = 21
        H, W = 100, 100
        
        # Create mask where left half is class 0, right half is ignore (255)
        true_mask = torch.zeros((H, W), dtype=torch.long)
        true_mask[:, W//2:] = 255
        
        # Prediction is all class 0
        pred_mask = torch.zeros((H, W), dtype=torch.long)
        
        # Compute IoU
        iou, valid = compute_iou(pred_mask, true_mask, num_classes, ignore_index=255)
        
        print(f"IoU for class 0: {iou[0]:.4f}")
        print(f"Class 0 is valid: {valid[0]}")
        
        # Only class 0 should be valid (ignore pixels should be excluded)
        assert valid[0], "Class 0 should be valid"
        
        # IoU should be 1.0 because we ignore the right half
        assert np.isclose(iou[0], 1.0), \
            f"Expected IoU=1.0 for class 0 (ignore pixels excluded), got {iou[0]:.4f}"
        print("✓ Ignore pixels (255) are properly excluded from IoU calculation")
        
        # Test with wrong predictions in ignore region (should still be ignored)
        pred_mask_wrong = torch.zeros((H, W), dtype=torch.long)
        pred_mask_wrong[:, W//2:] = 5  # Wrong predictions in ignore region
        
        iou2, valid2 = compute_iou(pred_mask_wrong, true_mask, num_classes, ignore_index=255)
        print(f"IoU with wrong predictions in ignore region: {iou2[0]:.4f}")
        
        assert np.isclose(iou2[0], 1.0), \
            f"Wrong predictions in ignore region should not affect IoU"
        print("✓ Wrong predictions in ignore region do not affect IoU")
        
        print("\n✓ TEST 4 PASSED: Ignore index is handled correctly")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 4 FAILED: {str(e)}")
        return False


def test_edge_cases():
    """Test edge cases: empty predictions, single class, etc."""
    print("\n" + "=" * 60)
    print("TEST 5: Edge Cases")
    print("=" * 60)
    
    try:
        num_classes = 21
        H, W = 100, 100
        
        # Test 1: Single class in GT and prediction (all correct)
        print("\nTest 5.1: Single class (all correct)")
        true_mask = torch.zeros((H, W), dtype=torch.long)
        pred_mask = torch.zeros((H, W), dtype=torch.long)
        iou, valid = compute_iou(pred_mask, true_mask, num_classes)
        
        assert valid[0], "Class 0 should be valid"
        assert np.isclose(iou[0], 1.0), f"Expected IoU=1.0, got {iou[0]:.4f}"
        print(f"  ✓ Single class with perfect prediction: IoU={iou[0]:.4f}")
        
        # Test 2: Class not in GT but in prediction (should not be in valid)
        print("\nTest 5.2: Class in prediction but not in GT")
        true_mask = torch.zeros((H, W), dtype=torch.long)
        pred_mask = torch.ones((H, W), dtype=torch.long)
        iou, valid = compute_iou(pred_mask, true_mask, num_classes)
        
        assert valid[0], "Class 0 should be valid (in GT)"
        assert not valid[1], "Class 1 should not be valid (not in GT)"
        print(f"  ✓ Class in prediction but not GT is correctly marked as invalid")
        
        # Test 3: Very small mask
        print("\nTest 5.3: Small mask (10x10)")
        true_mask = torch.randint(0, 5, (10, 10))
        pred_mask = true_mask.clone()
        iou, valid = compute_iou(pred_mask, true_mask, num_classes)
        valid_ious = iou[~np.isnan(iou)]
        
        assert np.allclose(valid_ious, 1.0), "Small mask with perfect predictions should have IoU=1.0"
        print(f"  ✓ Small mask works correctly: mean IoU={valid_ious.mean():.4f}")
        
        print("\n✓ TEST 5 PASSED: Edge cases handled correctly")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 5 FAILED: {str(e)}")
        return False


def test_miou_multiple_images():
    """Test mIoU computation across multiple images."""
    print("\n" + "=" * 60)
    print("TEST 6: mIoU Across Multiple Images")
    print("=" * 60)
    
    try:
        num_classes = 21
        N = 5
        H, W = 100, 100
        
        # Test 1: Perfect predictions on all images
        print("\nTest 6.1: Perfect predictions on all images")
        true_masks = torch.randint(0, num_classes, (N, H, W))
        pred_masks = true_masks.clone()
        
        results = compute_miou(pred_masks, true_masks, num_classes, return_per_class=True)
        
        print(f"  Overall mIoU: {results['miou']:.4f}")
        print(f"  Number of classes found: {np.sum(results['class_counts'] > 0)}")
        
        assert np.isclose(results['miou'], 1.0), \
            f"Expected mIoU=1.0 for perfect predictions, got {results['miou']:.4f}"
        print(f"  ✓ Perfect predictions: mIoU = {results['miou']:.4f}")
        
        # Test 2: No overlap on all images
        print("\nTest 6.2: No overlap on all images")
        true_masks = torch.zeros((N, H, W), dtype=torch.long)
        pred_masks = torch.ones((N, H, W), dtype=torch.long)
        
        results = compute_miou(pred_masks, true_masks, num_classes, return_per_class=True)
        
        print(f"  Overall mIoU: {results['miou']:.4f}")
        print(f"  IoU for class 0: {results['iou_per_class'][0]:.4f}")
        
        assert np.isclose(results['miou'], 0.0), \
            f"Expected mIoU=0.0 for no overlap, got {results['miou']:.4f}"
        print(f"  ✓ No overlap: mIoU = {results['miou']:.4f}")
        
        # Test 3: Mixed performance across images
        print("\nTest 6.3: Mixed performance")
        true_masks = torch.zeros((N, H, W), dtype=torch.long)
        pred_masks = torch.zeros((N, H, W), dtype=torch.long)
        
        # Make some images perfect, some with no overlap
        pred_masks[0] = 1  # No overlap
        pred_masks[1] = 0  # Perfect
        pred_masks[2] = 1  # No overlap
        pred_masks[3] = 0  # Perfect
        pred_masks[4] = 0  # Perfect
        
        results = compute_miou(pred_masks, true_masks, num_classes, return_per_class=True)
        
        print(f"  Overall mIoU: {results['miou']:.4f}")
        print(f"  Expected: 0.6 (3/5 perfect, 2/5 no overlap)")
        
        # Average IoU for class 0 should be 0.6 (3 perfect, 2 zero)
        expected_miou = 0.6
        assert np.isclose(results['miou'], expected_miou, atol=0.01), \
            f"Expected mIoU≈{expected_miou:.4f}, got {results['miou']:.4f}"
        print(f"  ✓ Mixed performance: mIoU = {results['miou']:.4f}")
        
        print("\n✓ TEST 6 PASSED: mIoU across multiple images works correctly")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 6 FAILED: {str(e)}")
        return False


def test_confusion_matrix():
    """Test confusion matrix computation."""
    print("\n" + "=" * 60)
    print("TEST 7: Confusion Matrix")
    print("=" * 60)
    
    try:
        num_classes = 3  # Simplified for easier verification
        H, W = 10, 10
        
        # Create simple masks
        # GT: top half class 0, bottom half class 1
        true_mask = torch.zeros((H, W), dtype=torch.long)
        true_mask[H//2:, :] = 1
        
        # Pred: left half class 0, right half class 1
        pred_mask = torch.zeros((H, W), dtype=torch.long)
        pred_mask[:, W//2:] = 1
        
        # Compute confusion matrix
        confusion = compute_confusion_matrix(
            pred_mask.unsqueeze(0), 
            true_mask.unsqueeze(0), 
            num_classes
        )
        
        print("Confusion matrix:")
        print(confusion)
        
        # Verify shape
        assert confusion.shape == (num_classes, num_classes), \
            f"Expected shape ({num_classes}, {num_classes}), got {confusion.shape}"
        print(f"✓ Confusion matrix has correct shape: {confusion.shape}")
        
        # Verify sum equals total pixels (minus ignored)
        total_pixels = H * W
        assert confusion.sum() == total_pixels, \
            f"Expected sum={total_pixels}, got {confusion.sum()}"
        print(f"✓ Confusion matrix sum equals total pixels: {confusion.sum()}")
        
        # Test IoU from confusion matrix
        iou_per_class = iou_from_confusion_matrix(confusion)
        print(f"\nIoU from confusion matrix: {iou_per_class[:num_classes]}")
        
        # Should match direct IoU computation
        iou_direct, _ = compute_iou(pred_mask, true_mask, num_classes)
        print(f"IoU from direct computation: {iou_direct[:num_classes]}")
        
        for i in range(2):  # Check first 2 classes
            if not np.isnan(iou_direct[i]) and not np.isnan(iou_per_class[i]):
                assert np.isclose(iou_direct[i], iou_per_class[i], atol=0.001), \
                    f"IoU mismatch for class {i}: {iou_direct[i]} vs {iou_per_class[i]}"
        print("✓ IoU from confusion matrix matches direct computation")
        
        print("\n✓ TEST 7 PASSED: Confusion matrix computation works correctly")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 7 FAILED: {str(e)}")
        return False


def main():
    """Run all metric tests."""
    print("\n" + "=" * 70)
    print(" " * 15 + "METRIC VERIFICATION TESTS")
    print("=" * 70)
    
    tests = [
        ("Perfect Predictions", test_perfect_predictions),
        ("No Overlap", test_no_overlap),
        ("Partial Overlap", test_partial_overlap),
        ("Ignore Index", test_ignore_index),
        ("Edge Cases", test_edge_cases),
        ("mIoU Multiple Images", test_miou_multiple_images),
        ("Confusion Matrix", test_confusion_matrix),
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    # Print summary
    print("\n" + "=" * 70)
    print(" " * 25 + "TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:<40} {status}")
    
    print("-" * 70)
    print(f"Total: {passed}/{total} tests passed")
    print("=" * 70)
    
    if passed == total:
        print("\n✓ ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n✗ {total - passed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())

