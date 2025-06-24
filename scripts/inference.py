#!/usr/bin/env python
"""
Inference script for Bi-temporal Wildfire Burnt Area Detection

Usage:
    python scripts/inference.py --config configs/config.yaml --model outputs/best_model.pth --pre-dir data/test/pre --post-dir data/test/post --output-dir results/

Author: Your Name
"""

import os
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

from models.attention_unet import get_model
from datasets.bitemporal_dataset import BiTemporalDataset
from utils.visualize import save_prediction_overlay


def parse_args():
    parser = argparse.ArgumentParser(description='Bi-temporal Wildfire Burnt Area Detection Inference')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--model', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--pre-dir', type=str, required=True, help='Pre-fire images directory')
    parser.add_argument('--post-dir', type=str, required=True, help='Post-fire images directory')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--save-prob', action='store_true', help='Save probability maps')
    parser.add_argument('--save-overlay', action='store_true', help='Save overlay visualizations')
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config, device):
    """Create and load model"""
    model_config = config['model']
    model = get_model(model_config['type'], **model_config['params'])
    model.to(device)
    return model


def load_checkpoint(model, checkpoint_path, device):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"✅ Model loaded from {checkpoint_path}")
    return model


def inference(model, dataloader, device, output_dir, save_prob=False, save_overlay=False):
    """Run inference on test dataset"""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    if save_prob:
        prob_dir = os.path.join(output_dir, 'probabilities')
        os.makedirs(prob_dir, exist_ok=True)
    
    if save_overlay:
        overlay_dir = os.path.join(output_dir, 'overlays')
        os.makedirs(overlay_dir, exist_ok=True)
    
    pred_dir = os.path.join(output_dir, 'predictions')
    os.makedirs(pred_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing")):
            if len(batch) == 3:  # with ground truth
                pre_img, post_img, gt_mask = batch
                pre_img = pre_img.to(device)
                post_img = post_img.to(device)
                gt_mask = gt_mask.to(device)
            else:  # without ground truth
                pre_img, post_img = batch
                pre_img = pre_img.to(device)
                post_img = post_img.to(device)
                gt_mask = None
            
            # Forward pass
            if hasattr(model, 'forward') and len(model.forward.__code__.co_varnames) > 3:
                # Bi-temporal model
                output = model(pre_img, post_img)
            else:
                # Single input model (concatenated)
                concat_input = torch.cat([pre_img, post_img], dim=1)
                output = model(concat_input)
            
            # Convert to probabilities
            if output.shape[1] == 1:  # Binary segmentation
                probs = torch.sigmoid(output)
                preds = (probs > 0.5).float()
            else:  # Multi-class
                probs = torch.softmax(output, dim=1)
                preds = torch.argmax(probs, dim=1, keepdim=True).float()
            
            # Save results for each image in batch
            for i in range(pre_img.shape[0]):
                img_idx = batch_idx * dataloader.batch_size + i
                
                # Save binary prediction
                pred_np = preds[i, 0].cpu().numpy()
                pred_img = Image.fromarray((pred_np * 255).astype(np.uint8))
                pred_img.save(os.path.join(pred_dir, f'pred_{img_idx:04d}.png'))
                
                # Save probability map
                if save_prob:
                    prob_np = probs[i, 0].cpu().numpy()
                    prob_img = Image.fromarray((prob_np * 255).astype(np.uint8))
                    prob_img.save(os.path.join(prob_dir, f'prob_{img_idx:04d}.png'))
                
                # Save overlay visualization
                if save_overlay:
                    pre_np = pre_img[i].permute(1, 2, 0).cpu().numpy()
                    post_np = post_img[i].permute(1, 2, 0).cpu().numpy()
                    
                    # Denormalize if needed
                    pre_np = (pre_np * 0.5 + 0.5).clip(0, 1)
                    post_np = (post_np * 0.5 + 0.5).clip(0, 1)
                    
                    save_prediction_overlay(
                        pre_np, post_np, pred_np, 
                        os.path.join(overlay_dir, f'overlay_{img_idx:04d}.png')
                    )
    
    print(f"✅ Inference completed. Results saved to {output_dir}")


def main():
    args = parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Create dataset
    dataset = BiTemporalDataset(
        pre_dir=args.pre_dir,
        post_dir=args.post_dir,
        mask_dir=None,  # No ground truth for inference
        image_size=config['data']['image_size'],
        normalize=config['data']['normalize']
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=2
    )
    
    print(f"📊 Found {len(dataset)} image pairs for inference")
    
    # Create and load model
    model = create_model(config, device)
    model = load_checkpoint(model, args.model, device)
    
    # Run inference
    inference(
        model, dataloader, device, args.output_dir,
        save_prob=args.save_prob,
        save_overlay=args.save_overlay
    )


if __name__ == "__main__":
    main()
