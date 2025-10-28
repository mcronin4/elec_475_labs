# Lab 3: Semantic Segmentation with FCN-ResNet50

This lab implements semantic segmentation evaluation using PyTorch's pretrained FCN-ResNet50 model on the PASCAL VOC 2012 dataset, with mean Intersection over Union (mIoU) as the evaluation metric.

## Overview

The project consists of:
- **Dataset loading**: PASCAL VOC 2012 segmentation dataset (automatically downloaded)
- **Model**: Pretrained FCN-ResNet50 from torchvision
- **Metric**: Mean Intersection over Union (mIoU)
- **Verification**: Comprehensive test scripts for all components

## Files

### Core Implementation

- `dataset.py` - PASCAL VOC 2012 dataset loading and preprocessing
- `metrics.py` - mIoU metric calculation
- `evaluate.py` - Main evaluation script

### Test/Verification Scripts

- `test_dataset.py` - Verify dataset loading and preprocessing
- `test_metrics.py` - Verify mIoU calculation with synthetic data
- `test_evaluation.py` - Verify full evaluation pipeline

## Setup

### Prerequisites

Make sure you have the virtual environment activated:

```bash
source ../venv/bin/activate  # On macOS/Linux
```

### Required Packages

All required packages should already be installed in the venv:
- PyTorch
- torchvision
- numpy
- matplotlib
- tqdm

## Usage

### Step 1: Run Test Scripts

Run the test scripts to verify all components work correctly:

```bash
# Test dataset loading
python test_dataset.py

# Test mIoU metric
python test_metrics.py

# Test evaluation pipeline
python test_evaluation.py
```

All tests should pass before running the main evaluation.

### Step 2: Run Evaluation

Run the main evaluation script:

```bash
# Full dataset evaluation (1449 validation images)
python evaluate.py

# Evaluate on a subset (faster for testing)
python evaluate.py --num-samples 100

# Evaluate with visualizations
python evaluate.py --num-samples 50 --visualize

# Custom save directory
python evaluate.py --num-samples 50 --visualize --save-dir my_results
```

### Command Line Arguments

`evaluate.py` supports the following arguments:

- `--num-samples N` - Number of samples to evaluate (default: None = full dataset)
- `--batch-size N` - Batch size for evaluation (default: 4, forced to 1 for variable-size images)
- `--visualize` - Save visualization of sample predictions
- `--save-dir PATH` - Directory to save results (default: `evaluation_results/`)
- `--data-root PATH` - Root directory for dataset (default: `./data`)

## Dataset

### PASCAL VOC 2012

- **Size**: 1464 training images, 1449 validation images
- **Classes**: 21 (background + 20 object categories)
- **Segmentation masks**: 
  - Values 0-20: class indices
  - Value 255: ignore/boundary pixels (excluded from evaluation)

### Class Names

0. background, 1. aeroplane, 2. bicycle, 3. bird, 4. boat, 5. bottle,
6. bus, 7. car, 8. cat, 9. chair, 10. cow, 11. diningtable, 12. dog,
13. horse, 14. motorbike, 15. person, 16. pottedplant, 17. sheep,
18. sofa, 19. train, 20. tvmonitor

## Model

### FCN-ResNet50

- **Architecture**: Fully Convolutional Network with ResNet-50 backbone
- **Input**: RGB images (variable size)
- **Output**: Segmentation masks with 21 classes
- **Pretrained**: ImageNet backbone + PASCAL VOC 2012 segmentation head

The model is loaded using:
```python
from torchvision.models.segmentation import fcn_resnet50
model = fcn_resnet50(pretrained=True)
```

## Metric

### Mean Intersection over Union (mIoU)

For each class:
```
IoU = Intersection / Union = TP / (TP + FP + FN)
```

Overall mIoU:
```
mIoU = mean(IoU_per_class)
```

- Computed per-class across all images
- Boundary pixels (class 255) are ignored
- Classes not present in an image are excluded from averaging

## Expected Results

On the full PASCAL VOC 2012 validation set, the pretrained FCN-ResNet50 should achieve:
- **mIoU**: ~60-65%

Individual class IoU values vary significantly:
- High IoU: person, car, bicycle, dog, cat (common, distinct objects)
- Lower IoU: bottle, chair, potted plant (smaller, more ambiguous objects)

## Output

### Console Output

The evaluation script prints:
- Dataset information
- Model information  
- Progress bar during inference
- Per-class IoU scores with class names
- Overall mIoU

Example:
```
Class                        IoU      Count
------------------------------------------
background                0.9234       1449
aeroplane                 0.7891        285
bicycle                   0.3456        337
...
------------------------------------------

Overall mIoU:             0.6234
```

### Visualization

If `--visualize` is used, the script saves a visualization showing:
- Input images
- Ground truth segmentation
- Predicted segmentation

Saved to: `evaluation_results/predictions.png`

## Test Scripts

### test_dataset.py

Verifies:
- Dataset loads correctly
- Tensor shapes are correct ([3, H, W] for images, [H, W] for masks)
- Data types are correct (float32 for images, long for masks)
- Value ranges are valid (normalized images, class indices 0-20 + 255)
- Transforms are applied correctly
- Subset functionality works
- Class distribution is reasonable

### test_metrics.py

Verifies mIoU calculation with synthetic data:
- Perfect predictions (IoU = 1.0)
- No overlap (IoU = 0.0)
- Partial overlap scenarios
- Ignore index (255) handling
- Edge cases (empty predictions, single class)
- mIoU across multiple images
- Confusion matrix computation

### test_evaluation.py

Verifies full pipeline:
- Model loading
- Model inference with correct output shapes
- Evaluation on small subset
- Batch processing
- Device handling (CPU/CUDA/MPS)

## Device Support

The code automatically detects and uses the best available device:
1. CUDA (NVIDIA GPU)
2. MPS (Apple Metal)
3. CPU (fallback)

## Troubleshooting

### Dataset Download Issues

If the dataset fails to download automatically:
1. Download manually from: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/
2. Extract to: `./data/VOCdevkit/VOC2012/`

### Memory Issues

If you run out of memory:
- Reduce `--num-samples` to evaluate on a subset
- Use `--batch-size 1` (already default)
- Disable visualization: don't use `--visualize`

### Slow Evaluation

Full dataset evaluation takes time:
- Use `--num-samples` to evaluate a subset first
- Ensure you're using GPU (CUDA/MPS) if available
- Typical speed: ~1-2 seconds per image on GPU, ~5-10 seconds on CPU

## References

- PyTorch Semantic Segmentation: https://pytorch.org/vision/stable/models.html#semantic-segmentation
- PASCAL VOC Dataset: http://host.robots.ox.ac.uk/pascal/VOC/
- FCN Paper: Long et al., "Fully Convolutional Networks for Semantic Segmentation", CVPR 2015
- mIoU Metric: Standard metric for semantic segmentation evaluation

## Authors

ELEC 475 - Fall 2025

