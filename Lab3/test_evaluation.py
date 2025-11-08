#########################################################################################################
#
#   ELEC 475 - Lab 3: Semantic Segmentation
#   Fall 2025
#
#   test_evaluation.py - Verify full evaluation pipeline
#

import torch
import sys

from dataset import load_voc_dataset
from evaluate import load_fcn_resnet50, evaluate_model
from metrics import compute_miou


def test_model_loading():
    """Test that FCN-ResNet50 model loads correctly."""
    print("\n" + "=" * 60)
    print("TEST 1: Model Loading")
    print("=" * 60)
    
    try:
        # Determine device
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        
        print(f"Testing on device: {device}")
        
        # Load model
        model = load_fcn_resnet50(pretrained=True, device=device)
        
        # Check model is in eval mode
        assert not model.training, "Model should be in eval mode"
        print("✓ Model is in evaluation mode")
        
        # Check model parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model has {num_params:,} parameters")
        
        # Verify model is on correct device
        first_param_device = next(model.parameters()).device
        assert str(first_param_device).startswith(device.split(':')[0]), \
            f"Model should be on {device}, but is on {first_param_device}"
        print(f"✓ Model is on device: {first_param_device}")
        
        print("\n✓ TEST 1 PASSED: Model loads correctly")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 1 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_model_inference():
    """Test that model inference produces correct output shapes."""
    print("\n" + "=" * 60)
    print("TEST 2: Model Inference")
    print("=" * 60)
    
    try:
        # Determine device
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        
        print(f"Testing on device: {device}")
        
        # Load model
        model = load_fcn_resnet50(pretrained=True, device=device)
        
        # Load a sample from dataset
        print("Loading sample from dataset...")
        dataset = load_voc_dataset(root='./data', image_set='val')
        image, mask = dataset[0]
        
        print(f"Sample image shape: {image.shape}")
        print(f"Sample mask shape: {mask.shape}")
        
        # Add batch dimension
        image_batch = image.unsqueeze(0).to(device)
        
        # Run inference
        print("Running inference...")
        with torch.no_grad():
            output = model(image_batch)
        
        # Check output is a dictionary
        assert isinstance(output, dict), f"Expected dict output, got {type(output)}"
        print("✓ Model returns dictionary")
        
        # Check 'out' key exists
        assert 'out' in output, "Output should have 'out' key"
        print("✓ Output has 'out' key")
        
        # Check output shape
        logits = output['out']
        expected_shape = (1, 21, image.shape[1], image.shape[2])
        print(f"Logits shape: {logits.shape}")
        print(f"Expected shape: {expected_shape}")
        
        assert logits.shape[0] == 1, f"Expected batch size 1, got {logits.shape[0]}"
        assert logits.shape[1] == 21, f"Expected 21 classes, got {logits.shape[1]}"
        assert logits.shape[2] == image.shape[1], f"Height mismatch: {logits.shape[2]} vs {image.shape[1]}"
        assert logits.shape[3] == image.shape[2], f"Width mismatch: {logits.shape[3]} vs {image.shape[2]}"
        print("✓ Output shape is correct: [B, 21, H, W]")
        
        # Test argmax to get predictions
        preds = torch.argmax(logits, dim=1)
        print(f"Predictions shape after argmax: {preds.shape}")
        
        # Remove batch dimension for comparison with single mask
        preds = preds.squeeze(0)  # [H, W]
        print(f"Predictions shape after squeeze: {preds.shape}")
        
        assert preds.shape == mask.shape, \
            f"Prediction shape {preds.shape} doesn't match mask shape {mask.shape}"
        print("✓ Prediction shape matches mask shape")
        
        # Check prediction values are in valid range
        unique_preds = torch.unique(preds)
        print(f"Unique predicted classes: {unique_preds.tolist()}")
        
        for pred_class in unique_preds:
            assert 0 <= pred_class < 21, f"Invalid predicted class: {pred_class}"
        print("✓ All predicted classes are in valid range [0, 20]")
        
        print("\n✓ TEST 2 PASSED: Model inference works correctly")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 2 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation_on_subset():
    """Test full evaluation pipeline on a small subset."""
    print("\n" + "=" * 60)
    print("TEST 3: Evaluation on Small Subset")
    print("=" * 60)
    
    try:
        # Determine device
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        
        print(f"Testing on device: {device}")
        
        # Load model
        print("Loading model...")
        model = load_fcn_resnet50(pretrained=True, device=device)
        
        # Load dataset
        print("Loading dataset...")
        dataset = load_voc_dataset(root='./data', image_set='val')
        
        # Evaluate on small subset
        num_samples = 5
        print(f"\nEvaluating on {num_samples} samples...")
        
        results = evaluate_model(
            model=model,
            dataset=dataset,
            device=device,
            batch_size=1,
            num_samples=num_samples,
            visualize=False
        )
        
        # Check results structure
        assert 'miou' in results, "Results should contain 'miou'"
        assert 'iou_per_class' in results, "Results should contain 'iou_per_class'"
        assert 'class_counts' in results, "Results should contain 'class_counts'"
        print("✓ Results have correct structure")
        
        # Check mIoU is a valid number
        miou = results['miou']
        assert isinstance(miou, (float, np.float32, np.float64)), \
            f"mIoU should be a float, got {type(miou)}"
        assert 0.0 <= miou <= 1.0, f"mIoU should be in [0, 1], got {miou}"
        print(f"✓ mIoU is valid: {miou:.4f}")
        
        # Check per-class IoU
        iou_per_class = results['iou_per_class']
        assert len(iou_per_class) == 21, \
            f"Expected 21 per-class IoU values, got {len(iou_per_class)}"
        print("✓ Per-class IoU has 21 values")
        
        # Check class counts
        class_counts = results['class_counts']
        assert len(class_counts) == 21, \
            f"Expected 21 class counts, got {len(class_counts)}"
        print("✓ Class counts has 21 values")
        
        # Verify at least some classes were found
        num_classes_found = sum(1 for count in class_counts if count > 0)
        assert num_classes_found > 0, "No classes found in samples"
        print(f"✓ Found {num_classes_found} classes in {num_samples} samples")
        
        # Print some statistics
        print(f"\nEvaluation statistics:")
        print(f"  Overall mIoU: {miou:.4f}")
        print(f"  Classes found: {num_classes_found}/21")
        
        valid_ious = [iou for iou in iou_per_class if not np.isnan(iou)]
        if len(valid_ious) > 0:
            print(f"  Mean class IoU: {np.mean(valid_ious):.4f}")
            print(f"  Min class IoU: {np.min(valid_ious):.4f}")
            print(f"  Max class IoU: {np.max(valid_ious):.4f}")
        
        print("\n✓ TEST 3 PASSED: Evaluation pipeline works correctly")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 3 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_processing():
    """Test that batch processing works correctly."""
    print("\n" + "=" * 60)
    print("TEST 4: Batch Processing")
    print("=" * 60)
    
    try:
        # Determine device
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        
        print(f"Testing on device: {device}")
        
        # Load model
        model = load_fcn_resnet50(pretrained=True, device=device)
        
        # Create dummy batch (same size images)
        B, C, H, W = 2, 3, 256, 256
        dummy_batch = torch.randn(B, C, H, W).to(device)
        
        # Normalize dummy batch (simulate ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        dummy_batch = (dummy_batch - mean) / std
        
        print(f"Dummy batch shape: {dummy_batch.shape}")
        
        # Run inference
        with torch.no_grad():
            output = model(dummy_batch)
        
        logits = output['out']
        print(f"Output shape: {logits.shape}")
        
        # Check batch dimension
        assert logits.shape[0] == B, f"Expected batch size {B}, got {logits.shape[0]}"
        print(f"✓ Batch processing works: input batch={B}, output batch={logits.shape[0]}")
        
        # Get predictions
        preds = torch.argmax(logits, dim=1)
        print(f"Predictions shape: {preds.shape}")
        assert preds.shape == (B, H, W), f"Expected shape ({B}, {H}, {W}), got {preds.shape}"
        print("✓ Predictions have correct shape")
        
        print("\n✓ TEST 4 PASSED: Batch processing works correctly")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 4 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_device_handling():
    """Test device handling (CPU/CUDA/MPS)."""
    print("\n" + "=" * 60)
    print("TEST 5: Device Handling")
    print("=" * 60)
    
    try:
        # Test CPU
        print("Testing CPU...")
        model_cpu = load_fcn_resnet50(pretrained=True, device='cpu')
        first_param = next(model_cpu.parameters())
        assert first_param.device.type == 'cpu', "Model should be on CPU"
        print("✓ CPU device works")
        
        # Test CUDA if available
        if torch.cuda.is_available():
            print("\nTesting CUDA...")
            model_cuda = load_fcn_resnet50(pretrained=True, device='cuda')
            first_param = next(model_cuda.parameters())
            assert first_param.device.type == 'cuda', "Model should be on CUDA"
            print("✓ CUDA device works")
        else:
            print("\n⚠ CUDA not available, skipping CUDA test")
        
        # Test MPS if available
        if torch.backends.mps.is_available():
            print("\nTesting MPS...")
            model_mps = load_fcn_resnet50(pretrained=True, device='mps')
            first_param = next(model_mps.parameters())
            assert first_param.device.type == 'mps', "Model should be on MPS"
            print("✓ MPS device works")
        else:
            print("\n⚠ MPS not available, skipping MPS test")
        
        print("\n✓ TEST 5 PASSED: Device handling works correctly")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 5 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all evaluation pipeline tests."""
    print("\n" + "=" * 70)
    print(" " * 12 + "EVALUATION PIPELINE VERIFICATION TESTS")
    print("=" * 70)
    
    # Import numpy here (used in some tests)
    global np
    import numpy as np
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Model Inference", test_model_inference),
        ("Evaluation on Subset", test_evaluation_on_subset),
        ("Batch Processing", test_batch_processing),
        ("Device Handling", test_device_handling),
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
        print("\nYou can now run the full evaluation with:")
        print("  python evaluate.py")
        print("  python evaluate.py --num-samples 100")
        print("  python evaluate.py --num-samples 50 --visualize")
        return 0
    else:
        print(f"\n✗ {total - passed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())

