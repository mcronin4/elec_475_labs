#########################################################################################################
#
#   ELEC 475 - Lab 3: Semantic Segmentation
#   Fall 2025
#
#   model.py - Compact semantic segmentation model with MobileNetV3-Small + ASPP
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small


class CompactSegmentationModel(nn.Module):
    """
    Compact semantic segmentation model with MobileNetV3-Small backbone.
    
    Architecture:
        - Encoder: MobileNetV3-Small (pretrained) with feature taps at stride 4, 8, 16, 32
        - Context: Lightweight ASPP with depthwise separable convolutions
        - Decoder: Multi-level skip connections (4 levels) + progressive upsampling
        - Output: 21 classes (PASCAL VOC)
        
    Total parameters: ~1.1M
    """
    
    def __init__(self, num_classes=21, pretrained=True, dropout=0.5):
        super(CompactSegmentationModel, self).__init__()
        
        # ========== ENCODER: MobileNetV3-Small ==========
        mobilenet = mobilenet_v3_small(pretrained=pretrained)
        self.features = mobilenet.features
        
        # Feature extraction points (verified from MobileNetV3-Small structure):
        # stride=4:  24 channels  (after features[2])
        # stride=8:  40 channels  (after features[4])
        # stride=16: 48 channels  (after features[8])
        # stride=32: 576 channels (after features[12])
        
        # ========== LIGHTWEIGHT ASPP MODULE ==========
        # Uses depthwise separable convolutions for efficiency
        # Input: 576 channels (from features[12] at stride 32)
        # Output: 128 channels
        self.aspp = LightweightASPP(
            in_channels=576,
            out_channels=128,
            dilation_rates=[6, 12, 18]
        )
        
        # ========== SKIP CONNECTION PROJECTIONS ==========
        # Project skip features to consistent channel counts
        self.skip_stride16 = self._make_skip_proj(48, 32)
        self.skip_stride8 = self._make_skip_proj(40, 32)
        self.skip_stride4 = self._make_skip_proj(24, 24)
        
        # ========== DECODER ==========
        # Stage 1: stride 32 → 16, concat with stride16 skip (128 + 32 = 160 input)
        self.decoder1 = self._make_decoder_block(128 + 32, 64)
        
        # Stage 2: stride 16 → 8, concat with stride8 skip (64 + 32 = 96 input)
        self.decoder2 = self._make_decoder_block(64 + 32, 48)
        
        # Stage 3: stride 8 → 4, concat with stride4 skip (48 + 24 = 72 input)
        self.decoder3 = self._make_decoder_block(48 + 24, 32)
        
        # ========== CLASSIFIER HEAD ==========
        self.dropout = nn.Dropout2d(p=dropout)
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)
        
        # Initialize decoder weights (encoder already pretrained)
        self._init_weights()
    
    def _make_skip_proj(self, in_ch, out_ch):
        """1x1 projection for skip connections."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def _make_decoder_block(self, in_ch, out_ch):
        """Decoder block with 2 convolutions."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def _init_weights(self):
        """Initialize decoder weights only (encoder already pretrained)."""
        # CRITICAL: Only initialize decoder components, not the pretrained encoder
        modules_to_init = [
            self.aspp, 
            self.skip_stride16, self.skip_stride8, self.skip_stride4,
            self.decoder1, self.decoder2, self.decoder3,
            self.classifier
        ]
        
        for module in modules_to_init:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: [B, 3, H, W] input images
            
        Returns:
            [B, 21, H, W] class logits (same spatial size as input)
        """
        input_size = x.shape[-2:]
        
        # ========== ENCODER ==========
        # Extract multi-level features from MobileNetV3-Small
        features = {}
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == 2:
                features['stride4'] = x    # [B, 24, H/4, W/4]
            elif i == 4:
                features['stride8'] = x    # [B, 40, H/8, W/8]
            elif i == 8:
                features['stride16'] = x   # [B, 48, H/16, W/16]
            elif i == 12:
                features['stride32'] = x   # [B, 576, H/32, W/32]
                break  # Stop here
        
        # ========== ASPP ==========
        x = self.aspp(features['stride32'])  # [B, 128, H/32, W/32]
        
        # ========== DECODER ==========
        # Process skip connections
        skip16 = self.skip_stride16(features['stride16'])  # [B, 32, H/16, W/16]
        skip8 = self.skip_stride8(features['stride8'])     # [B, 32, H/8, W/8]
        skip4 = self.skip_stride4(features['stride4'])     # [B, 24, H/4, W/4]
        
        # Upsample: stride 32 → 16
        x = F.interpolate(x, size=skip16.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip16], dim=1)  # [B, 160, H/16, W/16]
        x = self.decoder1(x)                # [B, 64, H/16, W/16]
        
        # Upsample: stride 16 → 8
        x = F.interpolate(x, size=skip8.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip8], dim=1)   # [B, 96, H/8, W/8]
        x = self.decoder2(x)                # [B, 48, H/8, W/8]
        
        # Upsample: stride 8 → 4
        x = F.interpolate(x, size=skip4.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip4], dim=1)   # [B, 72, H/4, W/4]
        x = self.decoder3(x)                # [B, 32, H/4, W/4]
        
        # ========== CLASSIFIER ==========
        x = self.dropout(x)
        x = self.classifier(x)              # [B, 21, H/4, W/4]
        
        # Upsample to input size
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        
        return x  # [B, 21, H, W]


class LightweightASPP(nn.Module):
    """
    Lightweight ASPP using depthwise separable convolutions.
    Reduces parameters significantly compared to standard ASPP.
    """
    
    def __init__(self, in_channels, out_channels, dilation_rates=[6, 12, 18]):
        super(LightweightASPP, self).__init__()
        
        branch_channels = out_channels // 4
        
        # Branch 1: 1x1 conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branches 2-4: Depthwise separable dilated convolutions
        self.branch2 = self._make_dsep_branch(in_channels, branch_channels, dilation_rates[0])
        self.branch3 = self._make_dsep_branch(in_channels, branch_channels, dilation_rates[1])
        self.branch4 = self._make_dsep_branch(in_channels, branch_channels, dilation_rates[2])
        
        # Branch 5: Global pooling
        self.branch5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, branch_channels, 1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        # Fusion: 5 branches × branch_channels input
        concat_channels = 5 * branch_channels
        self.fusion = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _make_dsep_branch(self, in_ch, out_ch, dilation):
        """Depthwise separable dilated convolution branch."""
        return nn.Sequential(
            # Depthwise: each input channel convolved separately
            nn.Conv2d(in_ch, in_ch, 3, padding=dilation, dilation=dilation, 
                     groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            # Pointwise: 1x1 conv to change channels
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        
        b5 = self.branch5(x)
        b5 = F.interpolate(b5, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        # Concatenate all 5 branches
        out = torch.cat([b1, b2, b3, b4, b5], dim=1)
        out = self.fusion(out)
        return out


if __name__ == "__main__":
    """Test model architecture and count parameters."""
    print("=" * 80)
    print("Testing CompactSegmentationModel")
    print("=" * 80)
    
    # Create model
    model = CompactSegmentationModel(num_classes=21, pretrained=False)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Parameters:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Test forward pass with dummy input
    print(f"\nTesting forward pass...")
    batch_size = 2
    height, width = 320, 320
    dummy_input = torch.randn(batch_size, 3, height, width)
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    
    # Verify output shape
    expected_shape = (batch_size, 21, height, width)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    print(f"  ✓ Output shape is correct!")
    
    # Test with different input sizes
    print(f"\nTesting with variable input sizes...")
    for h, w in [(256, 256), (384, 384), (512, 512)]:
        dummy_input = torch.randn(1, 3, h, w)
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (1, 21, h, w), f"Failed for size {h}x{w}"
        print(f"  ✓ {h}x{w} -> {output.shape}")
    
    print("\n" + "=" * 80)
    print("Model test complete!")
    print("=" * 80)

