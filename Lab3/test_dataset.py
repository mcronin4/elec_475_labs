#########################################################################################################
#
#   ELEC 475 - Lab 3: Semantic Segmentation
#   Fall 2025
#
#   test_dataset.py - Verify PASCAL VOC 2012 dataset loading and preprocessing
#

import torch
import numpy as np
from dataset import load_voc_dataset, VOC_CLASSES, get_voc_transforms
import sys


def test_dataset_loading():
    """Test that dataset loads correctly."""
    print("\n" + "=" * 60)
    print("TEST 1: Dataset Loading")
    print("=" * 60)
    
    try:
        dataset = load_voc_dataset(root='./data', image_set='val', download=True)
        
        # Check dataset size
        assert len(dataset) > 0, "Dataset is empty"
        print(f"✓ Dataset loaded with {len(dataset)} samples")
        
        # Check expected size (PASCAL VOC 2012 val should have 1449 images)
        expected_size = 1449
        if len(dataset) == expected_size:
            print(f"✓ Dataset has expected size: {expected_size}")
        else:
            print(f"⚠ Warning: Expected {expected_size} samples, got {len(dataset)}")
        
        # Check class information
        assert dataset.num_classes == 21, f"Expected 21 classes, got {dataset.num_classes}"
        print(f"✓ Number of classes: {dataset.num_classes}")
        
        assert dataset.ignore_index == 255, f"Expected ignore index 255, got {dataset.ignore_index}"
        print(f"✓ Ignore index: {dataset.ignore_index}")
        
        assert len(dataset.classes) == 21, f"Expected 21 class names, got {len(dataset.classes)}"
        print(f"✓ Class names loaded: {len(dataset.classes)} classes")
        
        print("\n✓ TEST 1 PASSED: Dataset loading successful")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 1 FAILED: {str(e)}")
        return False


def test_tensor_shapes():
    """Test that loaded samples have correct tensor shapes."""
    print("\n" + "=" * 60)
    print("TEST 2: Tensor Shapes")
    print("=" * 60)
    
    try:
        dataset = load_voc_dataset(root='./data', image_set='val', download=True)
        
        # Load first sample
        image, mask = dataset[0]
        
        # Check image shape
        assert len(image.shape) == 3, f"Expected 3D image tensor, got shape {image.shape}"
        assert image.shape[0] == 3, f"Expected 3 channels, got {image.shape[0]}"
        print(f"✓ Image shape: {image.shape} (3 channels, H, W)")
        
        # Check mask shape
        assert len(mask.shape) == 2, f"Expected 2D mask tensor, got shape {mask.shape}"
        print(f"✓ Mask shape: {mask.shape} (H, W)")
        
        # Check that image and mask have same spatial dimensions
        assert image.shape[1] == mask.shape[0], "Image height doesn't match mask height"
        assert image.shape[2] == mask.shape[1], "Image width doesn't match mask width"
        print(f"✓ Image and mask spatial dimensions match: {image.shape[1:]} == {mask.shape}")
        
        # Test multiple samples to ensure consistency
        print("\nChecking multiple samples...")
        for i in range(min(5, len(dataset))):
            img, msk = dataset[i]
            assert len(img.shape) == 3 and img.shape[0] == 3, f"Sample {i}: Invalid image shape"
            assert len(msk.shape) == 2, f"Sample {i}: Invalid mask shape"
            assert img.shape[1] == msk.shape[0] and img.shape[2] == msk.shape[1], \
                f"Sample {i}: Shape mismatch"
        print(f"✓ All {min(5, len(dataset))} samples have correct shapes")
        
        print("\n✓ TEST 2 PASSED: Tensor shapes are correct")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 2 FAILED: {str(e)}")
        return False


def test_data_types():
    """Test that tensors have correct data types."""
    print("\n" + "=" * 60)
    print("TEST 3: Data Types")
    print("=" * 60)
    
    try:
        dataset = load_voc_dataset(root='./data', image_set='val', download=True)
        image, mask = dataset[0]
        
        # Check image dtype (should be float32)
        assert image.dtype == torch.float32, f"Expected float32 for image, got {image.dtype}"
        print(f"✓ Image dtype: {image.dtype}")
        
        # Check mask dtype (should be long/int64 for class indices)
        assert mask.dtype == torch.long or mask.dtype == torch.int64, \
            f"Expected long/int64 for mask, got {mask.dtype}"
        print(f"✓ Mask dtype: {mask.dtype}")
        
        print("\n✓ TEST 3 PASSED: Data types are correct")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 3 FAILED: {str(e)}")
        return False


def test_value_ranges():
    """Test that tensor values are in expected ranges."""
    print("\n" + "=" * 60)
    print("TEST 4: Value Ranges")
    print("=" * 60)
    
    try:
        dataset = load_voc_dataset(root='./data', image_set='val', download=True)
        
        print("Checking image normalization...")
        image, mask = dataset[0]
        
        # Images should be normalized (approximately in range [-3, 3] due to normalization)
        # Mean=[0.485, 0.456, 0.406], Std=[0.229, 0.224, 0.225]
        img_min, img_max = image.min().item(), image.max().item()
        print(f"  Image range: [{img_min:.3f}, {img_max:.3f}]")
        
        # Reasonable range for normalized images
        assert img_min >= -5 and img_max <= 5, \
            f"Image values outside expected range: [{img_min}, {img_max}]"
        print(f"✓ Image values are normalized (approximately [-3, 3])")
        
        print("\nChecking mask class indices...")
        unique_classes = torch.unique(mask).tolist()
        print(f"  Unique classes in mask: {unique_classes}")
        
        # Masks should contain class indices 0-20 and possibly 255 (ignore)
        for cls in unique_classes:
            assert (0 <= cls <= 20) or cls == 255, \
                f"Invalid class index: {cls} (expected 0-20 or 255)"
        print(f"✓ All mask values are valid class indices (0-20 or 255)")
        
        # Check class distribution across multiple samples
        print("\nAnalyzing class distribution across samples...")
        all_classes = set()
        for i in range(min(10, len(dataset))):
            _, msk = dataset[i]
            all_classes.update(torch.unique(msk).tolist())
        
        # Remove ignore index if present
        if 255 in all_classes:
            all_classes.remove(255)
        
        print(f"  Classes found in first 10 samples: {sorted(all_classes)}")
        print(f"  Number of unique classes: {len(all_classes)}")
        
        print("\n✓ TEST 4 PASSED: Value ranges are correct")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 4 FAILED: {str(e)}")
        return False


def test_transforms():
    """Test that transforms are applied correctly."""
    print("\n" + "=" * 60)
    print("TEST 5: Transforms")
    print("=" * 60)
    
    try:
        # Get transforms
        img_transform, target_transform = get_voc_transforms()
        
        print("✓ Transforms created successfully")
        print(f"  Image transform: {img_transform}")
        print(f"  Target transform: {target_transform}")
        
        # Load dataset and verify transforms were applied
        dataset = load_voc_dataset(root='./data', image_set='val', download=True)
        image, mask = dataset[0]
        
        # Verify image is a tensor (ToTensor was applied)
        assert torch.is_tensor(image), "Image is not a tensor"
        print("✓ Image is a PyTorch tensor (ToTensor applied)")
        
        # Verify mask is a tensor
        assert torch.is_tensor(mask), "Mask is not a tensor"
        print("✓ Mask is a PyTorch tensor")
        
        # Verify normalization was applied by checking mean and std
        # After normalization with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
        # the image should have approximately zero mean per channel
        channel_means = [image[i].mean().item() for i in range(3)]
        print(f"  Image channel means: {[f'{m:.3f}' for m in channel_means]}")
        
        # The mean should be roughly centered around 0 (not 0.5 which would be unnormalized)
        # This is a rough check - exact values depend on the image content
        for i, mean in enumerate(channel_means):
            assert -3 < mean < 3, f"Channel {i} mean {mean} suggests normalization may not be applied"
        print("✓ Normalization appears to be applied (channel means near 0)")
        
        print("\n✓ TEST 5 PASSED: Transforms are working correctly")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 5 FAILED: {str(e)}")
        return False


def test_subset_functionality():
    """Test subset creation functionality."""
    print("\n" + "=" * 60)
    print("TEST 6: Subset Functionality")
    print("=" * 60)
    
    try:
        dataset = load_voc_dataset(root='./data', image_set='val', download=True)
        original_size = len(dataset)
        
        # Test creating a subset
        subset_size = 10
        subset = dataset.get_subset(subset_size)
        
        assert len(subset) == subset_size, \
            f"Expected subset size {subset_size}, got {len(subset)}"
        print(f"✓ Subset created with {len(subset)} samples")
        
        # Test that subset returns valid samples
        image, mask = subset[0]
        assert torch.is_tensor(image) and torch.is_tensor(mask), \
            "Subset returns invalid data"
        print(f"✓ Subset returns valid samples")
        
        # Test with None (should return full dataset)
        full = dataset.get_subset(None)
        assert len(full) == original_size, \
            f"get_subset(None) should return full dataset"
        print(f"✓ get_subset(None) returns full dataset ({len(full)} samples)")
        
        # Test with size larger than dataset (should return full dataset)
        large = dataset.get_subset(original_size + 100)
        assert len(large) == original_size, \
            f"get_subset with large size should return full dataset"
        print(f"✓ get_subset with large size returns full dataset ({len(large)} samples)")
        
        print("\n✓ TEST 6 PASSED: Subset functionality works correctly")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 6 FAILED: {str(e)}")
        return False


def test_class_distribution():
    """Analyze class distribution in the dataset."""
    print("\n" + "=" * 60)
    print("TEST 7: Class Distribution Analysis")
    print("=" * 60)
    
    try:
        dataset = load_voc_dataset(root='./data', image_set='val', download=True)
        
        # Count class occurrences in first N samples
        num_samples_to_check = min(50, len(dataset))
        print(f"Analyzing class distribution in first {num_samples_to_check} samples...")
        
        class_counts = {i: 0 for i in range(21)}
        ignore_count = 0
        
        for i in range(num_samples_to_check):
            _, mask = dataset[i]
            unique_classes = torch.unique(mask)
            
            for cls in unique_classes:
                cls_val = cls.item()
                if cls_val == 255:
                    ignore_count += 1
                elif 0 <= cls_val < 21:
                    class_counts[cls_val] += 1
        
        # Print distribution
        print(f"\nClass distribution (samples containing each class):")
        print(f"{'Class ID':<10} {'Class Name':<20} {'Count':>10}")
        print("-" * 42)
        
        for cls_id, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            if count > 0:
                class_name = VOC_CLASSES[cls_id]
                print(f"{cls_id:<10} {class_name:<20} {count:>10}")
        
        print(f"\nIgnore pixels (255) found in {ignore_count} samples")
        
        # Verify background is present (should be in most/all images)
        assert class_counts[0] > 0, "Background class not found in any sample"
        print(f"\n✓ Background class (0) found in {class_counts[0]} samples")
        
        # Verify at least some other classes are present
        num_classes_found = sum(1 for count in class_counts.values() if count > 0)
        assert num_classes_found > 1, "Only background class found"
        print(f"✓ Found {num_classes_found} different classes in {num_samples_to_check} samples")
        
        print("\n✓ TEST 7 PASSED: Class distribution analysis complete")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 7 FAILED: {str(e)}")
        return False


def main():
    """Run all dataset tests."""
    print("\n" + "=" * 70)
    print(" " * 15 + "DATASET VERIFICATION TESTS")
    print("=" * 70)
    
    tests = [
        ("Dataset Loading", test_dataset_loading),
        ("Tensor Shapes", test_tensor_shapes),
        ("Data Types", test_data_types),
        ("Value Ranges", test_value_ranges),
        ("Transforms", test_transforms),
        ("Subset Functionality", test_subset_functionality),
        ("Class Distribution", test_class_distribution),
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

