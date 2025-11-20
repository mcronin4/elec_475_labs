#########################################################################################################
#
#   ELEC 475 - Lab 4: Fine-tuning ResNet50 for CLIP
#   Fall 2025
#
#   verify_dataset.py - Verify dataset integrity and display sample image-caption pairs
#

import random
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image

from dataset import COCODataset, get_clip_image_transform


def denormalize_image(tensor, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]):
    """
    Denormalize a normalized image tensor for visualization.
    
    Args:
        tensor: Normalized image tensor [3, H, W]
        mean: Mean values used for normalization
        std: Std values used for normalization
        
    Returns:
        numpy array: Denormalized image [H, W, 3] in range [0, 1]
    """
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    tensor = torch.clamp(tensor, 0, 1)
    return tensor.permute(1, 2, 0).numpy()


def display_random_samples(dataset: COCODataset, num_samples: int = 5, save_path: Optional[Path] = None):
    """
    Display random image-caption pairs from the dataset.
    
    Args:
        dataset: COCODataset instance
        num_samples: Number of samples to display
        save_path: Optional path to save the visualization
    """
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(10, 3 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for idx, ax in zip(indices, axes):
        image, caption_data = dataset[idx]
        image_id, caption_idx = dataset.pairs[idx]
        
        # Denormalize image for display
        image_display = denormalize_image(image)
        
        # Get caption text
        if isinstance(caption_data, torch.Tensor):
            caption_text = f"[Pre-encoded embedding, shape: {caption_data.shape}]"
            caption_info = dataset.get_captions(image_id)[caption_idx] if caption_idx < len(dataset.get_captions(image_id)) else "N/A"
            caption_text = f"{caption_info}\n{caption_text}"
        else:
            caption_text = caption_data
        
        # Display image
        ax.imshow(image_display)
        ax.axis('off')
        ax.set_title(f"Image ID: {image_id}, Caption {caption_idx + 1}/{len(dataset.get_captions(image_id))}\n{caption_text}", 
                    fontsize=10, wrap=True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()


def verify_dataset_integrity(dataset: COCODataset):
    """
    Verify dataset integrity by checking for missing files and data consistency.
    
    Args:
        dataset: COCODataset instance
    """
    print("=" * 60)
    print("DATASET INTEGRITY CHECK")
    print("=" * 60)
    
    # Check dataset length
    print(f"\nDataset length: {len(dataset)} pairs")
    print(f"Number of unique images: {len(dataset.images_dict)}")
    
    # Check for missing images
    print("\nChecking for missing images...")
    missing_count = 0
    sample_missing = []
    
    for i in range(min(1000, len(dataset))):  # Sample check
        image_id, caption_idx = dataset.pairs[i]
        image_info = dataset.images_dict[image_id]
        image_path = dataset.images_dir / image_info['file_name']
        
        if not image_path.exists():
            missing_count += 1
            if len(sample_missing) < 5:
                sample_missing.append(str(image_path))
    
    if missing_count > 0:
        print(f"  WARNING: Found {missing_count} missing images in sample")
        if sample_missing:
            print(f"  Examples: {sample_missing[:3]}")
    else:
        print("  ✓ All sampled images found")
    
    # Check caption consistency
    print("\nChecking caption consistency...")
    total_captions = sum(len(captions) for captions in dataset.captions_dict.values())
    print(f"  Total captions: {total_captions}")
    print(f"  Captions per image (avg): {total_captions / len(dataset.images_dict):.2f}")
    
    # Check text embeddings cache
    if dataset.use_cached_embeddings and dataset.text_embeddings_cache is not None:
        print("\nText embeddings cache:")
        print(f"  ✓ Cache loaded: {len(dataset.text_embeddings_cache)} images")
        
        # Check embedding dimensions
        sample_emb = next(iter(dataset.text_embeddings_cache.values()))[0]
        print(f"  Embedding dimension: {sample_emb.shape}")
        print(f"  Embedding dtype: {sample_emb.dtype}")
    else:
        print("\nText embeddings cache:")
        print("  ⚠ No cache loaded (using raw captions)")
    
    # Test loading a few samples
    print("\nTesting sample loading...")
    try:
        for i in range(min(5, len(dataset))):
            image, caption = dataset[i]
            assert image.shape == (3, 224, 224), f"Unexpected image shape: {image.shape}"
            if isinstance(caption, torch.Tensor):
                assert len(caption.shape) == 1, f"Unexpected caption embedding shape: {caption.shape}"
        print("  ✓ Sample loading successful")
    except Exception as e:
        print(f"  ✗ Error loading samples: {e}")
    
    print("\n" + "=" * 60)


def print_dataset_statistics(dataset: COCODataset):
    """
    Print detailed dataset statistics.
    
    Args:
        dataset: COCODataset instance
    """
    print("=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    
    print(f"\nSplit: {dataset.split}")
    print(f"Root directory: {dataset.root}")
    print(f"Images directory: {dataset.images_dir}")
    
    print(f"\nData counts:")
    print(f"  Total image-caption pairs: {len(dataset)}")
    print(f"  Unique images: {len(dataset.images_dict)}")
    
    # Caption statistics
    caption_counts = [len(captions) for captions in dataset.captions_dict.values()]
    print(f"\nCaption statistics:")
    print(f"  Total captions: {sum(caption_counts)}")
    print(f"  Min captions per image: {min(caption_counts)}")
    print(f"  Max captions per image: {max(caption_counts)}")
    print(f"  Average captions per image: {sum(caption_counts) / len(caption_counts):.2f}")
    
    # Image preprocessing info
    print(f"\nImage preprocessing:")
    print(f"  Size: 224×224")
    print(f"  Normalization: CLIP stats")
    print(f"    Mean: [0.48145466, 0.4578275, 0.40821073]")
    print(f"    Std: [0.26862954, 0.26130258, 0.27577711]")
    
    print("\n" + "=" * 60)


def main():
    """Main verification function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify COCO dataset and display samples')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'],
                        help='Dataset split to verify')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of random samples to display')
    parser.add_argument('--no-cache', action='store_true',
                        help='Do not use cached embeddings')
    parser.add_argument('--save', type=str, default=None,
                        help='Path to save visualization (optional)')
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading {args.split} dataset...")
    dataset = COCODataset(
        root='./data',
        split=args.split,
        use_cached_embeddings=not args.no_cache
    )
    
    # Print statistics
    print_dataset_statistics(dataset)
    
    # Verify integrity
    verify_dataset_integrity(dataset)
    
    # Display random samples
    print("\nDisplaying random samples...")
    save_path = Path(args.save) if args.save else None
    display_random_samples(dataset, num_samples=args.num_samples, save_path=save_path)


if __name__ == "__main__":
    main()

