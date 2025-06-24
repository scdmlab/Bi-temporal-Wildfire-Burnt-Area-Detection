"""
Loss functions for wildfire burnt area detection

Author: Your Name
Email: your.email@example.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss for binary/multi-class segmentation"""
    
    def __init__(self, smooth=1.0, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions (B, C, H, W) or (B, H, W) for binary
            targets: Ground truth (B, H, W)
        """
        if inputs.dim() == 4 and inputs.size(1) > 1:
            # Multi-class case
            inputs = F.softmax(inputs, dim=1)
            dice_scores = []
            for c in range(inputs.size(1)):
                input_c = inputs[:, c, :, :]
                target_c = (targets == c).float()
                intersection = (input_c * target_c).sum(dim=(1, 2))
                union = input_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))
                dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
                dice_scores.append(dice)
            dice_scores = torch.stack(dice_scores, dim=1)
            dice_loss = 1 - dice_scores.mean(dim=1)
        else:
            # Binary case
            if inputs.dim() == 4:
                inputs = inputs.squeeze(1)
            inputs = torch.sigmoid(inputs)
            targets = targets.float()
            
            intersection = (inputs * targets).sum(dim=(1, 2))
            union = inputs.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_loss = 1 - dice
        
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss


class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance"""
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions (B, C, H, W) or (B, H, W) for binary
            targets: Ground truth (B, H, W)
        """
        if inputs.dim() == 4 and inputs.size(1) > 1:
            # Multi-class case
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        else:
            # Binary case
            if inputs.dim() == 4:
                inputs = inputs.squeeze(1)
            targets = targets.float()
            
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            pt = torch.exp(-bce_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """Combined loss function (CE/BCE + Dice + Focal)"""
    
    def __init__(self, ce_weight=1.0, dice_weight=1.0, focal_weight=0.0, 
                 num_classes=1, class_weights=None):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.num_classes = num_classes
        
        if num_classes == 1:
            self.ce_loss = nn.BCEWithLogitsLoss(weight=class_weights)
        else:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss() if focal_weight > 0 else None
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions (B, C, H, W) or (B, H, W) for binary
            targets: Ground truth (B, H, W)
        """
        total_loss = 0.0
        
        # Cross-entropy/BCE loss
        if self.ce_weight > 0:
            if self.num_classes == 1:
                if inputs.dim() == 4:
                    inputs_ce = inputs.squeeze(1)
                else:
                    inputs_ce = inputs
                targets_ce = targets.float()
            else:
                inputs_ce = inputs
                targets_ce = targets.long()
            
            ce_loss = self.ce_loss(inputs_ce, targets_ce)
            total_loss += self.ce_weight * ce_loss
        
        # Dice loss
        if self.dice_weight > 0:
            dice_loss = self.dice_loss(inputs, targets)
            total_loss += self.dice_weight * dice_loss
        
        # Focal loss
        if self.focal_weight > 0 and self.focal_loss is not None:
            focal_loss = self.focal_loss(inputs, targets)
            total_loss += self.focal_weight * focal_loss
        
        return total_loss


class IoULoss(nn.Module):
    """IoU (Jaccard) loss"""
    
    def __init__(self, smooth=1.0):
        super(IoULoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions (B, C, H, W) or (B, H, W) for binary
            targets: Ground truth (B, H, W)
        """
        if inputs.dim() == 4 and inputs.size(1) > 1:
            # Multi-class case
            inputs = F.softmax(inputs, dim=1)
            iou_scores = []
            for c in range(inputs.size(1)):
                input_c = inputs[:, c, :, :]
                target_c = (targets == c).float()
                intersection = (input_c * target_c).sum(dim=(1, 2))
                union = input_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2)) - intersection
                iou = (intersection + self.smooth) / (union + self.smooth)
                iou_scores.append(iou)
            iou_scores = torch.stack(iou_scores, dim=1)
            iou_loss = 1 - iou_scores.mean(dim=1)
        else:
            # Binary case
            if inputs.dim() == 4:
                inputs = inputs.squeeze(1)
            inputs = torch.sigmoid(inputs)
            targets = targets.float()
            
            intersection = (inputs * targets).sum(dim=(1, 2))
            union = inputs.sum(dim=(1, 2)) + targets.sum(dim=(1, 2)) - intersection
            iou = (intersection + self.smooth) / (union + self.smooth)
            iou_loss = 1 - iou
        
        return iou_loss.mean()


# Loss factory function
def get_loss_function(loss_type='combined', **kwargs):
    """
    Factory function to get loss function
    
    Args:
        loss_type (str): Type of loss ('bce', 'ce', 'dice', 'focal', 'iou', 'combined')
        **kwargs: Loss function parameters
    
    Returns:
        nn.Module: Loss function
    """
    losses = {
        'bce': lambda: nn.BCEWithLogitsLoss(**kwargs),
        'ce': lambda: nn.CrossEntropyLoss(**kwargs),
        'dice': lambda: DiceLoss(**kwargs),
        'focal': lambda: FocalLoss(**kwargs),
        'iou': lambda: IoULoss(**kwargs),
        'combined': lambda: CombinedLoss(**kwargs)
    }
    
    if loss_type not in losses:
        raise ValueError(f"Unknown loss type: {loss_type}. Available: {list(losses.keys())}")
    
    return losses[loss_type]()


if __name__ == "__main__":
    # Test loss functions
    batch_size, height, width = 2, 64, 64
    num_classes = 1
    
    # Create dummy data
    if num_classes == 1:
        inputs = torch.randn(batch_size, 1, height, width)
        targets = torch.randint(0, 2, (batch_size, height, width))
    else:
        inputs = torch.randn(batch_size, num_classes, height, width)
        targets = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Test different losses
    losses = ['bce', 'dice', 'focal', 'iou', 'combined']
    
    for loss_name in losses:
        try:
            if loss_name == 'bce' and num_classes > 1:
                continue  # Skip BCE for multi-class
            
            loss_fn = get_loss_function(loss_name, num_classes=num_classes)
            loss_value = loss_fn(inputs, targets)
            print(f"{loss_name.upper()} Loss: {loss_value.item():.4f}")
        except Exception as e:
            print(f"Error with {loss_name}: {e}")
