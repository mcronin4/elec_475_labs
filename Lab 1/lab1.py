import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from model import autoencoderMLP4Layer

def load_autoencoder_model(weights_path='MLP.8.pth', bottleneck_size=8):
    # Create model instance
    model = autoencoderMLP4Layer(N_input=784, N_bottleneck=bottleneck_size, N_output=784)
    
    # Load the trained weights
    try:
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        print(f"Successfully loaded weights from {weights_path}")
    except FileNotFoundError:
        print(f"Warning: Could not find weights file {weights_path}")
        print("Using randomly initialized model")
    
    # Set model to evaluation mode and disable gradients
    model.eval()
    
    return model

def preprocess_image(image_tensor):
    # Convert to float32 and normalize to [0, 1]
    image = image_tensor.float() / 255.0
    
    # Flatten to 1x784
    image = image.view(1, -1)
    
    return image

def add_noise(image_tensor, noise_factor=255):

    # Add uniform noise
    noise = torch.rand(image_tensor.size()) * noise_factor
    noisy_image = image_tensor + noise
    
    return noisy_image

def encode_image(model, image_tensor):
    # Preprocess the image
    processed_image = preprocess_image(image_tensor)
    
    # Disable gradient calculations for inference
    with torch.no_grad():
        # Use the existing encode method
        bottleneck = model.encode(processed_image)
    
    return bottleneck

def decode_bottleneck(model, bottleneck_tensor):
    # Disable gradient calculations for inference
    with torch.no_grad():
        # Use the existing decode method
        reconstructed = model.decode(bottleneck_tensor)
    
    # Reshape back to 28x28 for display
    reconstructed = reconstructed.view(28, 28)
    
    return reconstructed

def reconstruct_image(model, image_tensor):
    # Preprocess the image
    processed_image = preprocess_image(image_tensor)
    
    # Disable gradient calculations for inference
    with torch.no_grad():
        # Forward pass through the model
        reconstructed = model(processed_image)
    
    # Reshape back to 28x28 for display
    reconstructed = reconstructed.view(28, 28)
    
    return reconstructed

def interpolate_between_images(model, image1, image2, n_steps=10):
    # Encode both images to bottleneck space
    bottleneck1 = encode_image(model, image1)
    bottleneck2 = encode_image(model, image2)
    
    # Create interpolation steps
    interpolated_images = []
    
    for i in range(n_steps + 1):
        # Linear interpolation between bottleneck representations
        alpha = i / n_steps
        interpolated_bottleneck = (1 - alpha) * bottleneck1 + alpha * bottleneck2
        
        # Decode the interpolated bottleneck
        interpolated_image = decode_bottleneck(model, interpolated_bottleneck)
        interpolated_images.append(interpolated_image)
    
    return interpolated_images


def visualize():
    # Load the trained autoencoder model
    model = load_autoencoder_model()
    
    # Define the transform to convert images to tensors
    train_transform = transforms.Compose([transforms.ToTensor()])
    
    # Load the MNIST training dataset
    train_set = MNIST('./data/mnist', train=True, download=True,
                              transform=train_transform)
    
    while True:
        # Step 1: Ask for first number and run visualize + visualize_noise
        idx1 = int(input("Enter first image index (or -1 to quit): "))
        if idx1 == -1:
            break
            
        # Get the first image and label
        original_image1 = train_set.data[idx1]
        label1 = train_set.targets[idx1]
        
        # Step 1a: Visualize original and reconstructed
        reconstructed_image1 = reconstruct_image(model, original_image1)
        
        f = plt.figure()
        f.add_subplot(1, 2, 1)
        plt.imshow(original_image1, cmap='gray')
        plt.title('Original')
        f.add_subplot(1, 2, 2)
        plt.imshow(reconstructed_image1, cmap='gray')
        plt.title('Reconstructed')
        plt.show()
        
        print(f"Step 4: Displayed original and reconstructed images at index {idx1} with label: {label1.item()}")
        
        # Step 1b: Visualize noise denoising
        noisy_image1 = add_noise(original_image1)
        denoised_image1 = reconstruct_image(model, noisy_image1)
        
        f = plt.figure()
        f.add_subplot(1, 3, 1)
        plt.imshow(original_image1, cmap='gray')
        plt.title('Original')
        f.add_subplot(1, 3, 2)
        plt.imshow(noisy_image1, cmap='gray')
        plt.title('Noisy')
        f.add_subplot(1, 3, 3)
        plt.imshow(denoised_image1, cmap='gray')
        plt.title('Denoised')
        plt.show()
        
        print(f"Step 5: Displayed original, noisy, and denoised images at index {idx1} with label: {label1.item()}")
        
        # Step 2: Ask for second number and run interpolation
        idx2 = int(input("Enter second image index for interpolation: "))
        n_steps = int(input("Enter number of interpolation steps (default 10): ") or "10")
        
        # Get the second image and label
        original_image2 = train_set.data[idx2]
        label2 = train_set.targets[idx2]
        
        # Step 2: Perform interpolation
        interpolated_images = interpolate_between_images(model, original_image1, original_image2, n_steps)
        
        # Display the interpolation sequence
        fig, axes = plt.subplots(1, len(interpolated_images), figsize=(len(interpolated_images) * 2, 2))
        
        for i, img in enumerate(interpolated_images):
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'Step {i}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Step 6: Interpolated between image {idx1} (label: {label1.item()}) and image {idx2} (label: {label2.item()})")
        print("=" * 60)

if __name__ == "__main__":
    visualize()