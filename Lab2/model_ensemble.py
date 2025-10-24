import torch
import torch.nn as nn
import os

from model import SnoutNet
from model_alexnet import AlexNetNose
from model_vgg16 import VGG16Nose


class EnsembleModel(nn.Module):
    """
    Ensemble model that combines predictions from SnoutNet, AlexNet, and VGG16.
    Uses simple averaging of predictions from all three models.
    
    Input: (batch_size, 3, 227, 227)
    Output: (batch_size, 2) - (u, v) coordinates of nose location
    """
    def __init__(self, snoutnet_path, alexnet_path, vgg16_path, device='cpu'):
        """
        Initialize ensemble model by loading three pre-trained models.
        
        Args:
            snoutnet_path: Path to trained SnoutNet weights
            alexnet_path: Path to trained AlexNet weights
            vgg16_path: Path to trained VGG16 weights
            device: Device to load models on ('cuda', 'mps', or 'cpu')
        """
        super(EnsembleModel, self).__init__()
        
        self.device = device
        
        # Initialize models
        print("Loading ensemble models...")
        
        # SnoutNet
        if not os.path.exists(snoutnet_path):
            raise FileNotFoundError(f"SnoutNet model not found: {snoutnet_path}")
        self.snoutnet = SnoutNet()
        self.snoutnet.load_state_dict(torch.load(snoutnet_path, map_location=device))
        self.snoutnet.eval()
        print(f"  ✓ Loaded SnoutNet from {snoutnet_path}")
        
        # AlexNet
        if not os.path.exists(alexnet_path):
            raise FileNotFoundError(f"AlexNet model not found: {alexnet_path}")
        self.alexnet = AlexNetNose(pretrained=False)
        self.alexnet.load_state_dict(torch.load(alexnet_path, map_location=device))
        self.alexnet.eval()
        print(f"  ✓ Loaded AlexNet from {alexnet_path}")
        
        # VGG16
        if not os.path.exists(vgg16_path):
            raise FileNotFoundError(f"VGG16 model not found: {vgg16_path}")
        self.vgg16 = VGG16Nose(pretrained=False)
        self.vgg16.load_state_dict(torch.load(vgg16_path, map_location=device))
        self.vgg16.eval()
        print(f"  ✓ Loaded VGG16 from {vgg16_path}")
        
        # Move all models to device
        self.snoutnet = self.snoutnet.to(device)
        self.alexnet = self.alexnet.to(device)
        self.vgg16 = self.vgg16.to(device)
        
        # Store model paths for reference
        self.model_paths = {
            'snoutnet': snoutnet_path,
            'alexnet': alexnet_path,
            'vgg16': vgg16_path
        }
        
        print("Ensemble model ready!")
    
    def forward(self, x):
        """
        Forward pass through all three models and average predictions.
        
        Args:
            x: Input tensor (batch_size, 3, 227, 227)
            
        Returns:
            Averaged predictions (batch_size, 2)
        """
        with torch.no_grad():
            # Get predictions from all three models
            pred_snoutnet = self.snoutnet(x)
            pred_alexnet = self.alexnet(x)
            pred_vgg16 = self.vgg16(x)
            
            # Average the predictions
            ensemble_pred = (pred_snoutnet + pred_alexnet + pred_vgg16) / 3.0
            
        return ensemble_pred
    
    def get_individual_predictions(self, x):
        """
        Get predictions from each individual model (useful for debugging/analysis).
        
        Args:
            x: Input tensor (batch_size, 3, 227, 227)
            
        Returns:
            dict: Dictionary with predictions from each model
        """
        with torch.no_grad():
            predictions = {
                'snoutnet': self.snoutnet(x),
                'alexnet': self.alexnet(x),
                'vgg16': self.vgg16(x)
            }
            predictions['ensemble'] = (predictions['snoutnet'] + 
                                      predictions['alexnet'] + 
                                      predictions['vgg16']) / 3.0
        
        return predictions


def create_ensemble(suffix="", device='cpu'):
    """
    Helper function to create an ensemble model based on augmentation suffix.
    
    Args:
        suffix: Augmentation suffix ("", "_aug_color", or "_aug_both")
        device: Device to load models on
        
    Returns:
        EnsembleModel instance
    """
    # Construct paths based on suffix
    snoutnet_path = f"model_weights/snoutnet/best_snoutnet{suffix}.pth"
    alexnet_path = f"model_weights/alexnet/best_alexnet{suffix}.pth"
    vgg16_path = f"model_weights/vgg16/best_vgg16{suffix}.pth"
    
    # Create and return ensemble
    return EnsembleModel(snoutnet_path, alexnet_path, vgg16_path, device=device)


if __name__ == "__main__":
    # Test the ensemble model
    print("Testing EnsembleModel...")
    
    # Test with non-augmented models
    try:
        model = create_ensemble(suffix="", device='cpu')
        
        # Test forward pass
        dummy_input = torch.randn(4, 3, 227, 227)
        output = model(dummy_input)
        
        print(f"\nInput shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Expected output shape: (4, 2)")
        
        assert output.shape == (4, 2), f"Output shape mismatch: {output.shape}"
        print("\n✓ Ensemble model test passed!")
        
        # Test individual predictions
        individual_preds = model.get_individual_predictions(dummy_input)
        print("\nIndividual model predictions (first sample):")
        for model_name, preds in individual_preds.items():
            print(f"  {model_name}: {preds[0].tolist()}")
            
    except FileNotFoundError as e:
        print(f"\nWarning: Could not test ensemble - {e}")
        print("Make sure to train all three models first!")

