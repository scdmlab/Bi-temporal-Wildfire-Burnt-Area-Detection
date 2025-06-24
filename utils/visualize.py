"""
Visualization utilities for Bi-temporal Wildfire Burnt Area Detection

Author: Your Name
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2
import os


def save_prediction_overlay(pre_img, post_img, prediction, save_path, alpha=0.6):
    """
    Save overlay visualization of prediction on post image
    
    Args:
        pre_img (np.ndarray): Pre-event image [H, W, 3]
        post_img (np.ndarray): Post-event image [H, W, 3]
        prediction (np.ndarray): Binary prediction mask [H, W]
        save_path (str): Path to save the overlay image
        alpha (float): Transparency of overlay
    """
    # Create figure with subplots
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Pre-event image
    axes[0].imshow(pre_img)
    axes[0].set_title('Pre-event')
    axes[0].axis('off')
    
    # Post-event image
    axes[1].imshow(post_img)
    axes[1].set_title('Post-event')
    axes[1].axis('off')
    
    # Prediction mask
    axes[2].imshow(prediction, cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    # Overlay
    axes[3].imshow(post_img)
      # Create colored overlay for burnt areas
    overlay = np.zeros_like(post_img)
    overlay[prediction > 0.5] = [1, 0, 0]  # Red for burnt areas
    
    axes[3].imshow(overlay, alpha=alpha)
    axes[3].set_title('Burnt Area Overlay')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_comparison_grid(pre_imgs, post_imgs, predictions, gt_masks=None, save_path=None):
    """
    Create a grid comparison of multiple samples
    
    Args:
        pre_imgs (list): List of pre-event images
        post_imgs (list): List of post-event images
        predictions (list): List of prediction masks
        gt_masks (list, optional): List of ground truth masks
        save_path (str, optional): Path to save the grid
    """
    n_samples = len(pre_imgs)
    n_cols = 4 if gt_masks is None else 5
    
    fig, axes = plt.subplots(n_samples, n_cols, figsize=(n_cols * 3, n_samples * 3))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        # Pre-event
        axes[i, 0].imshow(pre_imgs[i])
        if i == 0:
            axes[i, 0].set_title('Pre-event')
        axes[i, 0].axis('off')
        
        # Post-event
        axes[i, 1].imshow(post_imgs[i])
        if i == 0:
            axes[i, 1].set_title('Post-event')
        axes[i, 1].axis('off')
        
        # Prediction
        axes[i, 2].imshow(predictions[i], cmap='gray', vmin=0, vmax=1)
        if i == 0:
            axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
        
        # Overlay
        axes[i, 3].imshow(post_imgs[i])
        overlay = np.zeros_like(post_imgs[i])
        overlay[predictions[i] > 0.5] = [1, 0, 0]
        axes[i, 3].imshow(overlay, alpha=0.6)
        if i == 0:
            axes[i, 3].set_title('Overlay')
        axes[i, 3].axis('off')
        
        # Ground truth (if available)
        if gt_masks is not None:
            axes[i, 4].imshow(gt_masks[i], cmap='gray', vmin=0, vmax=1)
            if i == 0:
                axes[i, 4].set_title('Ground Truth')
            axes[i, 4].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_curves(train_losses, val_losses, train_metrics=None, val_metrics=None, save_path=None):
    """
    Plot training and validation curves
    
    Args:
        train_losses (list): Training losses
        val_losses (list): Validation losses
        train_metrics (dict, optional): Training metrics (e.g., {'accuracy': [...], 'iou': [...]})
        val_metrics (dict, optional): Validation metrics
        save_path (str, optional): Path to save the plot
    """
    n_plots = 1
    if train_metrics:
        n_plots += len(train_metrics)
    
    fig, axes = plt.subplots(1, n_plots, figsize=(n_plots * 5, 4))
    
    if n_plots == 1:
        axes = [axes]
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    axes[0].plot(epochs, train_losses, 'b-', label='Training Loss')
    axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Metrics plots
    if train_metrics:
        for idx, (metric_name, train_values) in enumerate(train_metrics.items()):
            ax_idx = idx + 1
            axes[ax_idx].plot(epochs, train_values, 'b-', label=f'Training {metric_name.title()}')
            
            if val_metrics and metric_name in val_metrics:
                axes[ax_idx].plot(epochs, val_metrics[metric_name], 'r-', 
                                label=f'Validation {metric_name.title()}')
            
            axes[ax_idx].set_title(metric_name.title())
            axes[ax_idx].set_xlabel('Epoch')
            axes[ax_idx].set_ylabel(metric_name.title())
            axes[ax_idx].legend()
            axes[ax_idx].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_attention_visualization(attention_maps, input_img, save_path=None):
    """
    Visualize attention maps
    
    Args:
        attention_maps (list): List of attention maps from different layers
        input_img (np.ndarray): Input image
        save_path (str, optional): Path to save visualization
    """
    n_maps = len(attention_maps)
    fig, axes = plt.subplots(2, n_maps + 1, figsize=((n_maps + 1) * 3, 6))
    
    # Original image
    axes[0, 0].imshow(input_img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')
    
    # Attention maps
    for i, att_map in enumerate(attention_maps):
        # Resize attention map to match input size
        att_resized = cv2.resize(att_map, (input_img.shape[1], input_img.shape[0]))
        
        # Raw attention map
        axes[0, i + 1].imshow(att_resized, cmap='hot')
        axes[0, i + 1].set_title(f'Attention Map {i + 1}')
        axes[0, i + 1].axis('off')
        
        # Attention overlay
        axes[1, i + 1].imshow(input_img)
        axes[1, i + 1].imshow(att_resized, cmap='hot', alpha=0.6)
        axes[1, i + 1].set_title(f'Overlay {i + 1}')
        axes[1, i + 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def save_change_detection_gif(pre_img, post_img, prediction, save_path, duration=1000):
    """
    Create an animated GIF showing pre_fire/post_fire/burnt_area sequence
    
    Args:
        pre_img (np.ndarray): Pre-event image
        post_img (np.ndarray): Post-event image
        prediction (np.ndarray): Change prediction
        save_path (str): Path to save GIF
        duration (int): Duration between frames in milliseconds
    """
    # Convert to PIL Images
    pre_pil = Image.fromarray((pre_img * 255).astype(np.uint8))
    post_pil = Image.fromarray((post_img * 255).astype(np.uint8))
      # Create burnt area overlay
    overlay = post_img.copy()
    overlay[prediction > 0.5] = [1, 0, 0]  # Red for burnt areas
    overlay_pil = Image.fromarray((overlay * 255).astype(np.uint8))
    
    # Create GIF
    frames = [pre_pil, post_pil, overlay_pil]
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )


def create_error_analysis_plot(predictions, ground_truths, save_path=None):
    """
    Create error analysis visualization
    
    Args:
        predictions (np.ndarray): Model predictions
        ground_truths (np.ndarray): Ground truth masks
        save_path (str, optional): Path to save plot
    """
    # Calculate error types
    tp = ((predictions == 1) & (ground_truths == 1)).astype(int)  # True Positive
    fp = ((predictions == 1) & (ground_truths == 0)).astype(int)  # False Positive
    fn = ((predictions == 0) & (ground_truths == 1)).astype(int)  # False Negative
    tn = ((predictions == 0) & (ground_truths == 0)).astype(int)  # True Negative
    
    # Create color-coded error map
    error_map = np.zeros((*predictions.shape, 3))
    error_map[tp == 1] = [0, 1, 0]  # Green for TP
    error_map[fp == 1] = [1, 0, 0]  # Red for FP
    error_map[fn == 1] = [1, 1, 0]  # Yellow for FN
    error_map[tn == 1] = [0, 0, 0]  # Black for TN
    
    # Create legend patches
    legend_elements = [
        patches.Patch(color='green', label='True Positive'),
        patches.Patch(color='red', label='False Positive'),
        patches.Patch(color='yellow', label='False Negative'),
        patches.Patch(color='black', label='True Negative')
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Ground truth
    axes[0].imshow(ground_truths, cmap='gray')
    axes[0].set_title('Ground Truth')
    axes[0].axis('off')
    
    # Prediction
    axes[1].imshow(predictions, cmap='gray')
    axes[1].set_title('Prediction')
    axes[1].axis('off')
    
    # Error analysis
    axes[2].imshow(error_map)
    axes[2].set_title('Error Analysis')
    axes[2].axis('off')
    axes[2].legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_feature_maps(feature_maps, save_path=None, max_channels=16):
    """
    Visualize feature maps from CNN layers
    
    Args:
        feature_maps (torch.Tensor): Feature maps [C, H, W]
        save_path (str, optional): Path to save visualization
        max_channels (int): Maximum number of channels to display
    """
    n_channels = min(feature_maps.shape[0], max_channels)
    cols = 4
    rows = (n_channels + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(rows * cols):
        row, col = i // cols, i % cols
        
        if i < n_channels:
            axes[row, col].imshow(feature_maps[i], cmap='viridis')
            axes[row, col].set_title(f'Channel {i}')
        
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
