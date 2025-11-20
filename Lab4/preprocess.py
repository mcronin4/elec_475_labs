#########################################################################################################
#
#   ELEC 475 - Lab 4: Fine-tuning ResNet50 for CLIP
#   Fall 2025
#
#   preprocess.py - Preprocess COCO dataset and cache text embeddings
#

import json
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import CLIPProcessor, CLIPModel

from dataset import load_coco_annotations, save_text_embeddings_cache


def encode_captions(
    annotations_path: Path,
    model_name: str = 'openai/clip-vit-base-patch32',
    batch_size: int = 32,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> dict:
    """
    Encode all captions in COCO annotations using CLIP text encoder.
    
    Args:
        annotations_path: Path to COCO captions JSON file
        model_name: HuggingFace model name for CLIP
        batch_size: Batch size for encoding
        device: Device to run encoding on
        
    Returns:
        dict: {image_id: [list of caption embeddings]}
    """
    print(f"Loading CLIP model: {model_name}")
    print(f"Using device: {device}")
    
    # Load CLIP model and processor
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    # Move model to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    # Freeze all parameters (as per lab requirements)
    for param in model.parameters():
        param.requires_grad = False
    
    print("CLIP model loaded and frozen.")
    
    # Load annotations
    print(f"Loading annotations from {annotations_path}")
    with open(annotations_path, 'r') as f:
        data = json.load(f)
    
    # Group captions by image_id
    captions_by_image = {}
    for ann in data['annotations']:
        image_id = ann['image_id']
        caption = ann['caption']
        if image_id not in captions_by_image:
            captions_by_image[image_id] = []
        captions_by_image[image_id].append(caption)
    
    print(f"Found {len(captions_by_image)} images with {len(data['annotations'])} total captions")
    
    # Encode all captions
    print("Encoding captions...")
    embeddings_dict = {}
    
    # Process in batches for efficiency
    all_image_ids = list(captions_by_image.keys())
    all_captions = []
    caption_to_image_id = []
    
    for image_id in all_image_ids:
        captions = captions_by_image[image_id]
        for caption in captions:
            all_captions.append(caption)
            caption_to_image_id.append(image_id)
    
    # Encode in batches
    with torch.no_grad():
        for i in tqdm(range(0, len(all_captions), batch_size), desc="Encoding batches"):
            batch_captions = all_captions[i:i+batch_size]
            batch_image_ids = caption_to_image_id[i:i+batch_size]
            
            # Tokenize and encode
            inputs = processor(text=batch_captions, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            text_outputs = model.get_text_features(**inputs)
            # Normalize embeddings (CLIP uses normalized embeddings)
            text_embeddings = text_outputs / text_outputs.norm(dim=-1, keepdim=True)
            
            # Store embeddings
            for j, image_id in enumerate(batch_image_ids):
                if image_id not in embeddings_dict:
                    embeddings_dict[image_id] = []
                # Move to CPU to save memory
                embeddings_dict[image_id].append(text_embeddings[j].cpu())
    
    print(f"Encoded {len(all_captions)} captions for {len(embeddings_dict)} images")
    return embeddings_dict


def main(force_regenerate: bool = False):
    """
    Main preprocessing function.
    
    Args:
        force_regenerate: If True, regenerate cache even if it exists
    """
    data_dir = Path('./data')
    annotations_dir = data_dir / 'annotations'
    
    # Process training set
    print("=" * 60)
    print("PROCESSING TRAINING SET")
    print("=" * 60)
    train_annotations = annotations_dir / 'captions_train2014.json'
    train_cache_path = data_dir / 'train_text_embeddings.pt'
    
    if train_cache_path.exists() and not force_regenerate:
        print(f"Training cache already exists at {train_cache_path}")
        print("Skipping training set preprocessing. Use --force to regenerate.")
    else:
        if train_cache_path.exists():
            print(f"Regenerating training cache...")
        train_embeddings = encode_captions(train_annotations)
        save_text_embeddings_cache(train_embeddings, train_cache_path)
    
    # Process validation set
    print("\n" + "=" * 60)
    print("PROCESSING VALIDATION SET")
    print("=" * 60)
    val_annotations = annotations_dir / 'captions_val2014.json'
    val_cache_path = data_dir / 'val_text_embeddings.pt'
    
    if val_cache_path.exists() and not force_regenerate:
        print(f"Validation cache already exists at {val_cache_path}")
        print("Skipping validation set preprocessing. Use --force to regenerate.")
    else:
        if val_cache_path.exists():
            print(f"Regenerating validation cache...")
        val_embeddings = encode_captions(val_annotations)
        save_text_embeddings_cache(val_embeddings, val_cache_path)
    
    # Print statistics
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    
    if train_cache_path.exists():
        train_embeddings = torch.load(train_cache_path, map_location='cpu')
        total_train_captions = sum(len(embs) for embs in train_embeddings.values())
        print(f"Training set: {len(train_embeddings)} images, {total_train_captions} captions")
    
    if val_cache_path.exists():
        val_embeddings = torch.load(val_cache_path, map_location='cpu')
        total_val_captions = sum(len(embs) for embs in val_embeddings.values())
        print(f"Validation set: {len(val_embeddings)} images, {total_val_captions} captions")
    
    print("\nCache files saved:")
    print(f"  - {train_cache_path}")
    print(f"  - {val_cache_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess COCO dataset and cache text embeddings')
    parser.add_argument('--force', action='store_true', help='Force regeneration of cache files')
    args = parser.parse_args()
    main(force_regenerate=args.force)

