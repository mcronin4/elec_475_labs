# Lab 4: Fine-tuning ResNet50 for CLIP

## Dataset Preprocessing

This directory contains the preprocessing pipeline for the COCO 2014 dataset used in CLIP fine-tuning.

### Files

- `dataset.py`: COCODataset class for loading images and captions
- `preprocess.py`: Script to encode captions using CLIP text encoder and cache embeddings
- `verify_dataset.py`: Utility to verify dataset integrity and visualize samples

### Dataset Structure

The dataset should be organized as follows:

```
Lab4/data/
  ├── annotations/
  │   ├── captions_train2014.json
  │   └── captions_val2014.json
  ├── train2014/
  │   └── [COCO_train2014_*.jpg files]
  └── val2014/
      └── [COCO_val2014_*.jpg files]
```

### Usage

#### 1. Preprocess and Cache Text Embeddings

Run the preprocessing script to encode all captions using the frozen CLIP text encoder:

```bash
python preprocess.py
```

This will:
- Load the CLIP text encoder from HuggingFace (`openai/clip-vit-base-patch32`)
- Encode all captions from both train and validation sets
- Save cached embeddings to `data/train_text_embeddings.pt` and `data/val_text_embeddings.pt`

To force regeneration of cache files:
```bash
python preprocess.py --force
```

**Note**: This process can take a while (especially for the training set with ~414K captions). The cached embeddings significantly speed up training.

#### 2. Verify Dataset

Verify dataset integrity and visualize sample image-caption pairs:

```bash
python verify_dataset.py --split train --num-samples 5
```

Options:
- `--split`: Choose 'train' or 'val' (default: 'train')
- `--num-samples`: Number of random samples to display (default: 5)
- `--no-cache`: Don't use cached embeddings (for testing)
- `--save`: Path to save visualization image (optional)

#### 3. Use Dataset in Training

```python
from dataset import COCODataset

# Load dataset with cached embeddings
train_dataset = COCODataset(
    root='./data',
    split='train',
    use_cached_embeddings=True
)

# Or load without cache (captions will be returned as strings)
train_dataset = COCODataset(
    root='./data',
    split='train',
    use_cached_embeddings=False
)
```

### Image Preprocessing

Images are preprocessed with:
- Resize to 224×224
- Normalize with CLIP statistics:
  - Mean: [0.48145466, 0.4578275, 0.40821073]
  - Std: [0.26862954, 0.26130258, 0.27577711]

### Text Embeddings

- Each image in COCO has multiple captions (~5 on average)
- The dataset creates one (image, caption) pair for each caption
- Cached embeddings are normalized CLIP text features (512-dimensional)
- Embeddings are stored as: `{image_id: [list of caption embeddings]}`

### Dependencies

- `torch`, `torchvision`
- `transformers` (for CLIP model)
- `PIL` (Pillow)
- `matplotlib` (for verification script)
- `tqdm` (for progress bars)

