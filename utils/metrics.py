"""
Evaluation metrics for wildfire burnt area detection

Author: Tang Sui
Email: tsui5@wisc.edu
"""

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt


class SegmentationMetrics:
    """Metrics for wildfire burnt area detection tasks"""
    
    def __init__(self, num_classes=2, threshold=0.5):
        self.num_classes = num_classes
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.total_correct = 0
        self.total_pixels = 0
        self.class_correct = np.zeros(self.num_classes)
        self.class_total = np.zeros(self.num_classes)
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self.predictions = []
        self.targets = []
    
    def update(self, predictions, targets):
        """
        Update metrics with new predictions and targets
        
        Args:
            predictions: Model predictions (B, C, H, W) or (B, H, W)
            targets: Ground truth (B, H, W)
        """
        # Convert to numpy
        if torch.is_tensor(predictions):
            if predictions.dim() == 4:
                if predictions.size(1) == 1:
                    # Binary case with single channel
                    predictions = torch.sigmoid(predictions).squeeze(1)
                    predictions = (predictions > self.threshold).float()
                else:
                    # Multi-class case
                    predictions = torch.softmax(predictions, dim=1)
                    predictions = torch.argmax(predictions, dim=1)
            else:
                # Already squeezed
                if self.num_classes == 2:
                    predictions = torch.sigmoid(predictions)
                    predictions = (predictions > self.threshold).float()
            
            predictions = predictions.cpu().numpy()
        
        if torch.is_tensor(targets):
            targets = targets.cpu().numpy()
        
        # Flatten arrays
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        
        # Store for later use (e.g., ROC curve)
        self.predictions.extend(pred_flat.tolist())
        self.targets.extend(target_flat.tolist())
        
        # Overall accuracy
        correct = (pred_flat == target_flat).sum()
        self.total_correct += correct
        self.total_pixels += len(pred_flat)
        
        # Per-class accuracy
        for c in range(self.num_classes):
            class_mask = (target_flat == c)
            self.class_correct[c] += ((pred_flat == c) & class_mask).sum()
            self.class_total[c] += class_mask.sum()
        
        # Confusion matrix
        cm = confusion_matrix(target_flat, pred_flat, labels=list(range(self.num_classes)))
        self.confusion_matrix += cm
    
    def get_metrics(self):
        """Get all computed metrics"""
        metrics = {}
        
        # Overall accuracy
        metrics['accuracy'] = self.total_correct / max(self.total_pixels, 1)
        
        # Per-class accuracy
        class_acc = self.class_correct / np.maximum(self.class_total, 1)
        for i, acc in enumerate(class_acc):
            metrics[f'class_{i}_accuracy'] = acc
        
        # Mean accuracy
        valid_classes = self.class_total > 0
        metrics['mean_accuracy'] = class_acc[valid_classes].mean() if valid_classes.any() else 0
        
        # Precision, Recall, F1 from confusion matrix
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - tp
        fn = self.confusion_matrix.sum(axis=1) - tp
        
        precision = tp / np.maximum(tp + fp, 1)
        recall = tp / np.maximum(tp + fn, 1)
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1)
        
        for i in range(self.num_classes):
            metrics[f'class_{i}_precision'] = precision[i]
            metrics[f'class_{i}_recall'] = recall[i]
            metrics[f'class_{i}_f1'] = f1[i]
        
        # Mean metrics
        metrics['mean_precision'] = precision.mean()
        metrics['mean_recall'] = recall.mean()
        metrics['mean_f1'] = f1.mean()
        
        # IoU (Intersection over Union)
        intersection = tp
        union = tp + fp + fn
        iou = intersection / np.maximum(union, 1)
        
        for i in range(self.num_classes):
            metrics[f'class_{i}_iou'] = iou[i]
        
        metrics['mean_iou'] = iou.mean()
          # For binary classification, add burnt area detection specific metrics
        if self.num_classes == 2:
            # Assuming class 1 is the positive class (burnt area)
            metrics['burnt_area_precision'] = precision[1]
            metrics['burnt_area_recall'] = recall[1]
            metrics['burnt_area_f1'] = f1[1]
            metrics['burnt_area_iou'] = iou[1]
            
            # Specificity (True Negative Rate)
            tn = self.confusion_matrix[0, 0]
            fp_binary = self.confusion_matrix[0, 1]
            metrics['specificity'] = tn / max(tn + fp_binary, 1)
            
            # Balanced accuracy
            sensitivity = recall[1]  # Same as recall for positive class
            specificity = metrics['specificity']
            metrics['balanced_accuracy'] = (sensitivity + specificity) / 2
        
        return metrics
    
    def get_confusion_matrix(self):
        """Get normalized confusion matrix"""
        cm_normalized = self.confusion_matrix.astype('float') / (
            self.confusion_matrix.sum(axis=1)[:, np.newaxis] + 1e-8
        )
        return self.confusion_matrix, cm_normalized
    
    def get_roc_curve(self):
        """Get ROC curve data (for binary classification)"""
        if self.num_classes != 2:
            raise ValueError("ROC curve is only available for binary classification")
        
        fpr, tpr, thresholds = roc_curve(self.targets, self.predictions)
        roc_auc = auc(fpr, tpr)
        
        return {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': roc_auc
        }
    
    def get_pr_curve(self):
        """Get Precision-Recall curve data (for binary classification)"""
        if self.num_classes != 2:
            raise ValueError("PR curve is only available for binary classification")
        
        precision, recall, thresholds = precision_recall_curve(self.targets, self.predictions)
        pr_auc = auc(recall, precision)
        
        return {
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds,
            'auc': pr_auc
        }


def plot_confusion_matrix(cm, class_names=None, normalize=True, title='Confusion Matrix', 
                         save_path=None, show=True):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        normalize: Whether to normalize the matrix
        title: Plot title
        save_path: Path to save the plot
        show: Whether to show the plot
    """
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
        fmt = '.2f'
    else:
        fmt = 'd'
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(cm.shape[0])]
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_roc_curve(roc_data, title='ROC Curve', save_path=None, show=True):
    """Plot ROC curve"""
    plt.figure(figsize=(8, 6))
    plt.plot(roc_data['fpr'], roc_data['tpr'], 
             label=f'ROC Curve (AUC = {roc_data["auc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_pr_curve(pr_data, title='Precision-Recall Curve', save_path=None, show=True):
    """Plot Precision-Recall curve"""
    plt.figure(figsize=(8, 6))
    plt.plot(pr_data['recall'], pr_data['precision'], 
             label=f'PR Curve (AUC = {pr_data["auc"]:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    # Test metrics
    metrics = SegmentationMetrics(num_classes=2)
    
    # Create dummy data
    batch_size, height, width = 2, 32, 32
    predictions = torch.randn(batch_size, 1, height, width)
    targets = torch.randint(0, 2, (batch_size, height, width))
    
    # Update metrics
    metrics.update(predictions, targets)
    
    # Get results
    results = metrics.get_metrics()
    print("Metrics:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")
    
    # Test ROC curve
    try:
        roc_data = metrics.get_roc_curve()
        print(f"\nROC AUC: {roc_data['auc']:.4f}")
    except Exception as e:
        print(f"ROC curve error: {e}")
    
    # Test confusion matrix
    cm, cm_norm = metrics.get_confusion_matrix()
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\nNormalized Confusion Matrix:\n{cm_norm}")
