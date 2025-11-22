#########################################################################################################
#
#   ELEC 475 - Lab 4: Fine-tuning ResNet50 for CLIP
#   Fall 2025
#
#   dataset.py - COCO 2014 dataset handling for CLIP fine-tuning
#

from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


# CLIP normalization statistics
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def get_clip_image_transform():
    """
    Get transform pipeline for images to match CLIP preprocessing.
    
    Returns:
        transforms.Compose: Transform pipeline that resizes to 224x224 and normalizes with CLIP stats
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
    ])


def load_coco_annotations(annotations_path: Path) -> Tuple[Dict, Dict]:
    """
    Load COCO caption annotations from JSON file.
    
    Args:
        annotations_path: Path to COCO captions JSON file
        
    Returns:
        tuple: (images_dict, captions_dict)
            - images_dict: {image_id: {'file_name': str, 'id': int, ...}}
            - captions_dict: {image_id: [list of caption strings]}
    """
    with open(annotations_path, 'r') as f:
        data = json.load(f)
    
    # Build images dictionary
    images_dict = {img['id']: img for img in data['images']}
    
    # Build captions dictionary - group captions by image_id
    captions_dict = {}
    for ann in data['annotations']:
        image_id = ann['image_id']
        caption = ann['caption']
        if image_id not in captions_dict:
            captions_dict[image_id] = []
        captions_dict[image_id].append(caption)
    
    return images_dict, captions_dict


def load_text_embeddings_cache(cache_path: Path) -> Dict[int, List[torch.Tensor]]:
    """
    Load cached text embeddings from .pt file.
    
    Args:
        cache_path: Path to .pt cache file
        
    Returns:
        dict: {image_id: [list of caption embeddings]}
    """
    if not cache_path.exists():
        return None
    return torch.load(cache_path, map_location='cpu')


def save_text_embeddings_cache(embeddings_dict: Dict[int, List[torch.Tensor]], cache_path: Path):
    """
    Save text embeddings to .pt cache file.
    
    Args:
        embeddings_dict: Dictionary mapping image_id to list of caption embeddings
        cache_path: Path to save cache file
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings_dict, cache_path)
    print(f"Saved text embeddings cache to {cache_path}")


class COCODataset(Dataset):
    """
    COCO 2014 dataset loader for CLIP fine-tuning.
    
    This dataset loads images and their corresponding captions from COCO 2014.
    Images are preprocessed with CLIP normalization, and captions can be either
    tokenized text or pre-encoded embeddings (if cache is available).
    """
    
    def __init__(
        self,
        root: str = './data',
        split: str = 'train',
        transform=None,
        use_cached_embeddings: bool = True,
        text_embeddings_cache: Optional[Dict[int, List[torch.Tensor]]] = None
    ):
        """
        Initialize COCO dataset.
        
        Args:
            root: Root directory containing data/ folder
            split: 'train' or 'val'
            transform: Image transform pipeline (defaults to CLIP transform if None)
            use_cached_embeddings: Whether to use cached text embeddings if available
            text_embeddings_cache: Pre-loaded text embeddings cache (if None, will try to load from file)
        """
        self.root = Path(root)
        self.split = split
        
        # Set up paths
        self.images_dir = self.root / f'{split}2014'
        self.annotations_path = self.root / 'annotations' / f'captions_{split}2014.json'
        
        # Validate paths
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.annotations_path.exists():
            raise FileNotFoundError(f"Annotations file not found: {self.annotations_path}")
        
        # Load annotations
        self.images_dict, self.captions_dict = load_coco_annotations(self.annotations_path)
        
        # Build list of (image_id, caption_index) pairs
        # This allows us to iterate through all image-caption pairs
        self.pairs = []
        for image_id, captions in self.captions_dict.items():
            for caption_idx in range(len(captions)):
                self.pairs.append((image_id, caption_idx))
        
        # Set up transforms
        if transform is None:
            transform = get_clip_image_transform()
        self.transform = transform
        
        # Handle text embeddings
        self.use_cached_embeddings = use_cached_embeddings
        if text_embeddings_cache is None and use_cached_embeddings:
            cache_path = self.root / f'{split}_text_embeddings.pt'
            self.text_embeddings_cache = load_text_embeddings_cache(cache_path)
            if self.text_embeddings_cache is None:
                print(f"Warning: Text embeddings cache not found at {cache_path}. "
                      "Run preprocess.py to generate cache.")
                self.use_cached_embeddings = False
        else:
            self.text_embeddings_cache = text_embeddings_cache
        
        print(f"Loaded COCO {split} dataset: {len(self.pairs)} image-caption pairs from {len(self.images_dict)} images")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Returns:
            tuple: (image_tensor, caption_embedding_or_text)
                - image_tensor: [3, 224, 224] float tensor, normalized with CLIP stats
                - caption_embedding_or_text: Either pre-encoded embedding tensor or caption string
        """
        image_id, caption_idx = self.pairs[idx]
        
        # Load image
        image_info = self.images_dict[image_id]
        image_path = self.images_dir / image_info['file_name']
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        
        # Get caption or embedding
        if self.use_cached_embeddings and self.text_embeddings_cache is not None:
            # Return pre-encoded embedding
            caption_embedding = self.text_embeddings_cache[image_id][caption_idx]
            return image, caption_embedding
        else:
            # Return raw caption text (will need to be encoded during training)
            caption = self.captions_dict[image_id][caption_idx]
            return image, caption
    
    def get_image_info(self, image_id: int) -> Dict:
        """Get image metadata for a given image_id."""
        return self.images_dict.get(image_id, None)
    
    def get_captions(self, image_id: int) -> List[str]:
        """Get all captions for a given image_id."""
        return self.captions_dict.get(image_id, [])


if __name__ == "__main__":
    """Test dataset loading."""
    print("Testing COCO Dataset Loading...")
    print("=" * 60)
    
    # Test without cached embeddings
    print("\n1. Testing dataset without cached embeddings:")
    dataset = COCODataset(root='./data', split='train', use_cached_embeddings=False)
    print(f"   Dataset length: {len(dataset)}")
    
    # Test loading a sample
    if len(dataset) > 0:
        image, caption = dataset[0]
        print(f"   Image shape: {image.shape}")
        print(f"   Image dtype: {image.dtype}")
        print(f"   Image range: [{image.min():.3f}, {image.max():.3f}]")
        print(f"   Caption type: {type(caption)}")
        print(f"   Caption (first 100 chars): {str(caption)[:100]}")
    
    # Test validation dataset
    print("\n2. Testing validation dataset:")
    val_dataset = COCODataset(root='./data', split='val', use_cached_embeddings=False)
    print(f"   Validation dataset length: {len(val_dataset)}")
    
    print("\n" + "=" * 60)
    print("Dataset loading test complete!")

