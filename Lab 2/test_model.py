import torch
from model import SnoutNet, init_weights
from torchsummary import summary

def test_snoutnet():
    """
    Test SnoutNet model with dummy input to verify architecture.
    Expected: Input (1, 3, 227, 227) -> Output (1, 2)
    """
    print("Testing SnoutNet architecture...")
    
    # Create model
    model = SnoutNet()
    
    # Initialize weights
    model.apply(init_weights)
    
    # Create dummy input: batch_size=1, channels=3, width=227, height=227
    dummy_input = torch.randn(1, 3, 227, 227)
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output}")
    
    # Verify shapes
    expected_input_shape = (1, 3, 227, 227)
    expected_output_shape = (1, 2)
    
    assert dummy_input.shape == expected_input_shape, f"Expected input shape {expected_input_shape}, got {dummy_input.shape}"
    assert output.shape == expected_output_shape, f"Expected output shape {expected_output_shape}, got {output.shape}"
    
    print("✓ Model architecture test passed!")
    
    # Print model summary
    print("\nModel Summary:")
    try:
        summary(model, model.input_shape)
    except Exception as e:
        print(f"Could not generate summary: {e}")
        print("Model architecture:")
        print(model)
    
    return model

if __name__ == "__main__":
    model = test_snoutnet()
