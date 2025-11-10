#########################################################################################################
#
#   ELEC 475 - Lab 3: Semantic Segmentation
#   Fall 2025
#
#   distillation_losses.py - Knowledge distillation losses for teacher-student training
#

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResponseBasedDistillationLoss(nn.Module):
    """
    Response-based knowledge distillation loss.
    
    Uses temperature-scaled softmax and KL divergence to transfer knowledge
    from teacher to student model.
    
    Loss = α * L_task + β * L_distillation
    
    where:
    - L_task: Standard cross-entropy loss (student vs ground truth)
    - L_distillation: KL divergence between student and teacher softmax outputs
    
    Args:
        temperature (float): Temperature for softmax scaling (default: 4.0)
        alpha (float): Weight for task loss (default: 1.0)
        beta (float): Weight for distillation loss (default: 0.5)
        ignore_index (int): Index to ignore in loss computation (default: 255)
        reduction (str): Reduction method for loss (default: 'mean')
    """
    
    def __init__(self, temperature=4.0, alpha=1.0, beta=0.5, ignore_index=255, 
                 reduction='mean', class_weights=None):
        super(ResponseBasedDistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.ignore_index = ignore_index
        self.reduction = reduction
        
        # Task loss (student vs ground truth)
        # Can be set externally if class weights are needed
        self.task_loss = nn.CrossEntropyLoss(
            weight=class_weights, 
            ignore_index=ignore_index, 
            reduction=reduction
        )
        
    def forward(self, student_logits, teacher_logits, targets):
        """
        Compute combined task and distillation loss.
        
        Args:
            student_logits: [B, C, H, W] Student model logits
            teacher_logits: [B, C, H, W] Teacher model logits (may have different spatial size)
            targets: [B, H, W] Ground truth segmentation masks
            
        Returns:
            torch.Tensor: Combined loss value
        """
        # Task loss: student vs ground truth
        loss_task = self.task_loss(student_logits, targets)
        
        # Handle size mismatch: resize teacher logits to match student logits if needed
        if teacher_logits.shape[2:] != student_logits.shape[2:]:
            teacher_logits = F.interpolate(
                teacher_logits, 
                size=student_logits.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # Distillation loss: student vs teacher (using temperature-scaled softmax)
        # Reshape to [B*H*W, C] for easier computation
        B, C, H, W = student_logits.shape
        student_logits_flat = student_logits.permute(0, 2, 3, 1).contiguous().view(-1, C)
        teacher_logits_flat = teacher_logits.permute(0, 2, 3, 1).contiguous().view(-1, C)
        targets_flat = targets.view(-1)
        
        # Create mask to ignore pixels with ignore_index
        valid_mask = (targets_flat != self.ignore_index)
        
        if valid_mask.sum() > 0:
            # Apply valid mask
            student_logits_valid = student_logits_flat[valid_mask]
            teacher_logits_valid = teacher_logits_flat[valid_mask]
            
            # Temperature-scaled softmax
            student_soft = F.log_softmax(student_logits_valid / self.temperature, dim=1)
            teacher_soft = F.softmax(teacher_logits_valid / self.temperature, dim=1)
            
            # KL divergence: KL(student || teacher)
            # KL(p||q) = sum(p * log(p/q)) = sum(p * log(p) - p * log(q))
            # Since student_soft is log_softmax, we have log(p)
            # We compute: sum(teacher_soft * (log(teacher_soft) - log(student_soft)))
            # But more efficiently: KL = sum(teacher_soft * (log(teacher_soft) - student_soft))
            # However, we need to be careful: student_soft is already log_softmax
            # So: KL = sum(teacher_soft * (-student_soft)) + constant
            # The constant is H(teacher) which doesn't affect gradients
            # So we use: KL ≈ -sum(teacher_soft * student_soft)
            
            # More standard approach: use F.kl_div which expects log probabilities for first arg
            loss_distillation = F.kl_div(
                student_soft, 
                teacher_soft, 
                reduction='batchmean'
            ) * (self.temperature ** 2)
            
            # Scale by number of valid pixels to maintain proper scaling
            # kl_div with 'batchmean' already divides by batch size, but we need to scale
            # by the ratio of valid pixels to total pixels for proper weighting
            # Actually, since we're already using 'batchmean', it should be fine
            # But we need to account for the fact that we're only using valid pixels
            # So we multiply by temperature^2 (standard in distillation literature)
        else:
            # No valid pixels, set distillation loss to 0
            loss_distillation = torch.tensor(0.0, device=student_logits.device, requires_grad=True)
        
        # Combined loss
        loss_total = self.alpha * loss_task + self.beta * loss_distillation
        
        return loss_total, loss_task, loss_distillation


def test_response_distillation_loss():
    """Test response-based distillation loss."""
    print("=" * 80)
    print("Testing ResponseBasedDistillationLoss")
    print("=" * 80)
    
    # Create loss function
    loss_fn = ResponseBasedDistillationLoss(
        temperature=4.0,
        alpha=1.0,
        beta=0.5,
        ignore_index=255
    )
    
    # Create dummy data
    batch_size = 2
    num_classes = 21
    height, width = 64, 64
    
    # Student and teacher logits
    # Student needs gradients, teacher doesn't
    student_logits = torch.randn(batch_size, num_classes, height, width, requires_grad=True)
    teacher_logits = torch.randn(batch_size, num_classes, height, width)
    
    # Ground truth (random classes, with some ignore_index)
    targets = torch.randint(0, num_classes, (batch_size, height, width))
    # Set some pixels to ignore_index
    targets[0, :10, :10] = 255
    
    print(f"Student logits shape: {student_logits.shape}")
    print(f"Teacher logits shape: {teacher_logits.shape}")
    print(f"Targets shape: {targets.shape}")
    
    # Compute loss
    loss_total, loss_task, loss_distillation = loss_fn(student_logits, teacher_logits, targets)
    
    print(f"\nLoss total: {loss_total.item():.4f}")
    print(f"Loss task: {loss_task.item():.4f}")
    print(f"Loss distillation: {loss_distillation.item():.4f}")
    
    # Test gradient flow
    loss_total.backward()
    print(f"\n✓ Gradient computation successful")
    print(f"✓ Student logits requires grad: {student_logits.requires_grad}")
    
    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)


class FeatureBasedDistillationLoss(nn.Module):
    """
    Feature-based knowledge distillation loss.
    
    Uses cosine similarity to align intermediate feature maps between
    teacher and student models.
    
    Loss = α * L_task + γ * L_feature
    
    where:
    - L_task: Standard cross-entropy loss (student vs ground truth)
    - L_feature: Cosine similarity loss between student and teacher features
    
    Args:
        alpha (float): Weight for task loss (default: 1.0)
        gamma (float): Weight for feature distillation loss (default: 0.1)
        ignore_index (int): Index to ignore in loss computation (default: 255)
        feature_levels (list): List of feature levels to use (default: ['stride4', 'stride8', 'stride16'])
    """
    
    def __init__(self, alpha=1.0, gamma=0.1, ignore_index=255, 
                 reduction='mean', class_weights=None,
                 feature_levels=['stride4', 'stride8', 'stride16']):
        super(FeatureBasedDistillationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.feature_levels = feature_levels
        
        # Task loss (student vs ground truth)
        self.task_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index,
            reduction=reduction
        )
    
    def cosine_similarity_loss(self, student_feat, teacher_feat):
        """
        Compute cosine similarity loss between student and teacher features.
        
        Args:
            student_feat: [B, C_s, H, W] Student features
            teacher_feat: [B, C_t, H, W] Teacher features
            
        Returns:
            torch.Tensor: Cosine similarity loss (1 - cosine similarity)
        """
        # Handle size mismatch: interpolate teacher to match student
        if teacher_feat.shape[2:] != student_feat.shape[2:]:
            teacher_feat = F.interpolate(
                teacher_feat,
                size=student_feat.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Normalize features for cosine similarity
        # Flatten spatial dimensions: [B, C, H, W] -> [B, C, H*W]
        student_flat = student_feat.view(student_feat.shape[0], student_feat.shape[1], -1)
        teacher_flat = teacher_feat.view(teacher_feat.shape[0], teacher_feat.shape[1], -1)
        
        # Normalize along channel dimension
        student_norm = F.normalize(student_flat, p=2, dim=1)  # [B, C_s, H*W]
        teacher_norm = F.normalize(teacher_flat, p=2, dim=1)  # [B, C_t, H*W]
        
        # Handle channel mismatch: if channels differ, we need to project
        # For simplicity, we'll use the smaller channel dimension or project
        if student_norm.shape[1] != teacher_norm.shape[1]:
            # Project to same dimension (use smaller)
            min_channels = min(student_norm.shape[1], teacher_norm.shape[1])
            student_norm = student_norm[:, :min_channels, :]
            teacher_norm = teacher_norm[:, :min_channels, :]
        
        # Compute cosine similarity: [B, C, H*W]
        cosine_sim = (student_norm * teacher_norm).sum(dim=1)  # [B, H*W]
        
        # Average over spatial dimensions and batch
        # Loss is 1 - cosine similarity (we want to maximize similarity, so minimize (1 - sim))
        loss = 1.0 - cosine_sim.mean()
        
        return loss
    
    def forward(self, student_logits, student_features, teacher_features, targets):
        """
        Compute combined task and feature distillation loss.
        
        Args:
            student_logits: [B, C, H, W] Student model logits
            student_features: dict with student features at different stride levels
            teacher_features: dict with teacher features at different stride levels
            targets: [B, H, W] Ground truth segmentation masks
            
        Returns:
            tuple: (total_loss, task_loss, feature_loss)
        """
        # Task loss: student vs ground truth
        loss_task = self.task_loss(student_logits, targets)
        
        # Feature distillation loss: cosine similarity between student and teacher features
        feature_losses = []
        for level in self.feature_levels:
            if level in student_features and level in teacher_features:
                student_feat = student_features[level]
                teacher_feat = teacher_features[level]
                
                # Compute cosine similarity loss for this level
                level_loss = self.cosine_similarity_loss(student_feat, teacher_feat)
                feature_losses.append(level_loss)
        
        # Average feature loss across all levels
        if len(feature_losses) > 0:
            loss_feature = torch.stack(feature_losses).mean()
        else:
            loss_feature = torch.tensor(0.0, device=student_logits.device, requires_grad=True)
        
        # Combined loss
        loss_total = self.alpha * loss_task + self.gamma * loss_feature
        
        return loss_total, loss_task, loss_feature


class CombinedDistillationLoss(nn.Module):
    """
    Combined response-based and feature-based distillation loss.
    
    Loss = α * L_task + β * L_response + γ * L_feature
    
    Args:
        temperature (float): Temperature for response distillation (default: 4.0)
        alpha (float): Weight for task loss (default: 1.0)
        beta (float): Weight for response distillation loss (default: 0.5)
        gamma (float): Weight for feature distillation loss (default: 0.1)
        ignore_index (int): Index to ignore in loss computation (default: 255)
        class_weights: Class weights for task loss
        feature_levels (list): List of feature levels to use
    """
    
    def __init__(self, temperature=4.0, alpha=1.0, beta=0.5, gamma=0.1,
                 ignore_index=255, reduction='mean', class_weights=None,
                 feature_levels=['stride4', 'stride8', 'stride16']):
        super(CombinedDistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        
        # Task loss
        self.task_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index,
            reduction=reduction
        )
        
        # Feature loss helper
        self.feature_levels = feature_levels
    
    def cosine_similarity_loss(self, student_feat, teacher_feat):
        """Compute cosine similarity loss between features."""
        if teacher_feat.shape[2:] != student_feat.shape[2:]:
            teacher_feat = F.interpolate(
                teacher_feat,
                size=student_feat.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        
        student_flat = student_feat.view(student_feat.shape[0], student_feat.shape[1], -1)
        teacher_flat = teacher_feat.view(teacher_feat.shape[0], teacher_feat.shape[1], -1)
        
        student_norm = F.normalize(student_flat, p=2, dim=1)
        teacher_norm = F.normalize(teacher_flat, p=2, dim=1)
        
        if student_norm.shape[1] != teacher_norm.shape[1]:
            min_channels = min(student_norm.shape[1], teacher_norm.shape[1])
            student_norm = student_norm[:, :min_channels, :]
            teacher_norm = teacher_norm[:, :min_channels, :]
        
        cosine_sim = (student_norm * teacher_norm).sum(dim=1)
        loss = 1.0 - cosine_sim.mean()
        return loss
    
    def forward(self, student_logits, teacher_logits, student_features, teacher_features, targets):
        """
        Compute combined task, response, and feature distillation loss.
        
        Args:
            student_logits: [B, C, H, W] Student logits
            teacher_logits: [B, C, H, W] Teacher logits
            student_features: dict with student features
            teacher_features: dict with teacher features
            targets: [B, H, W] Ground truth masks
            
        Returns:
            tuple: (total_loss, task_loss, response_loss, feature_loss)
        """
        # Task loss
        loss_task = self.task_loss(student_logits, targets)
        
        # Response distillation loss
        if teacher_logits.shape[2:] != student_logits.shape[2:]:
            teacher_logits = F.interpolate(
                teacher_logits,
                size=student_logits.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        
        B, C, H, W = student_logits.shape
        student_logits_flat = student_logits.permute(0, 2, 3, 1).contiguous().view(-1, C)
        teacher_logits_flat = teacher_logits.permute(0, 2, 3, 1).contiguous().view(-1, C)
        targets_flat = targets.view(-1)
        
        valid_mask = (targets_flat != self.ignore_index)
        
        if valid_mask.sum() > 0:
            student_logits_valid = student_logits_flat[valid_mask]
            teacher_logits_valid = teacher_logits_flat[valid_mask]
            
            student_soft = F.log_softmax(student_logits_valid / self.temperature, dim=1)
            teacher_soft = F.softmax(teacher_logits_valid / self.temperature, dim=1)
            
            loss_response = F.kl_div(
                student_soft,
                teacher_soft,
                reduction='batchmean'
            ) * (self.temperature ** 2)
        else:
            loss_response = torch.tensor(0.0, device=student_logits.device, requires_grad=True)
        
        # Feature distillation loss
        feature_losses = []
        for level in self.feature_levels:
            if level in student_features and level in teacher_features:
                level_loss = self.cosine_similarity_loss(
                    student_features[level],
                    teacher_features[level]
                )
                feature_losses.append(level_loss)
        
        if len(feature_losses) > 0:
            loss_feature = torch.stack(feature_losses).mean()
        else:
            loss_feature = torch.tensor(0.0, device=student_logits.device, requires_grad=True)
        
        # Combined loss
        loss_total = (self.alpha * loss_task + 
                     self.beta * loss_response + 
                     self.gamma * loss_feature)
        
        return loss_total, loss_task, loss_response, loss_feature


if __name__ == "__main__":
    test_response_distillation_loss()


