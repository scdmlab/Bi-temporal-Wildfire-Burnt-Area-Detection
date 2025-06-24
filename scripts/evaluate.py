#!/usr/bin/env python
"""
Evaluation script for Bi-temporal Wildfire Burnt Area Detection

Usage:
    python scripts/evaluate.py --config configs/config.yaml --model outputs/best_model.pth --data-dir data/test

Author: Tang Sui
Email: tsui5@wisc.edu
"""

import os
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from models.attention_unet import get_model
from datasets.bitemporal_dataset import BiTemporalDataset
from utils.metrics import calculate_metrics, plot_roc_curve, plot_pr_curve


def parse_args():
    parser = argparse.ArgumentParser(description='Bi-temporal Wildfire Burnt Area Detection Evaluation')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--model', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--data-dir', type=str, required=True, help='Test data directory')
    parser.add_argument('--output-dir', type=str, default='evaluation_results', help='Output directory')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for evaluation')
    
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
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"✅ Model loaded from {checkpoint_path}")
    return model


def evaluate_model(model, dataloader, device, output_dir):
    """Evaluate model performance"""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    all_preds = []
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pre_img, post_img, targets = batch
            pre_img = pre_img.to(device)
            post_img = post_img.to(device)
            targets = targets.to(device)
            
            # Forward pass
            if hasattr(model, 'forward') and len(model.forward.__code__.co_varnames) > 3:
                outputs = model(pre_img, post_img)
            else:
                concat_input = torch.cat([pre_img, post_img], dim=1)
                outputs = model(concat_input)
            
            # Convert to probabilities and predictions
            if outputs.shape[1] == 1:  # Binary segmentation
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
            else:  # Multi-class
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1, keepdim=True).float()
                probs = probs[:, 1:2]  # Take positive class probability
            
            # Flatten and collect
            preds_flat = preds.view(-1).cpu().numpy()
            probs_flat = probs.view(-1).cpu().numpy()
            targets_flat = targets.view(-1).cpu().numpy()
            
            all_preds.extend(preds_flat)
            all_probs.extend(probs_flat)
            all_targets.extend(targets_flat)
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    
    # Calculate metrics
    metrics = calculate_metrics(all_targets, all_preds, all_probs)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    print(f"IoU: {metrics['iou']:.4f}")
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"AUC-PR: {metrics['auc_pr']:.4f}")
    print("="*50)
    
    # Save metrics to file
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write("Evaluation Results\n")
        f.write("="*30 + "\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
    
    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Change', 'Change'],
                yticklabels=['No Change', 'Change'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ROC Curve
    plot_roc_curve(all_targets, all_probs, save_path=os.path.join(output_dir, 'roc_curve.png'))
    
    # Precision-Recall Curve
    plot_pr_curve(all_targets, all_probs, save_path=os.path.join(output_dir, 'pr_curve.png'))
    
    # Classification Report
    report = classification_report(all_targets, all_preds, 
                                 target_names=['No Change', 'Change'],
                                 output_dict=True)
    
    # Save classification report
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write("Classification Report\n")
        f.write("="*40 + "\n")
        f.write(classification_report(all_targets, all_preds, 
                                    target_names=['No Change', 'Change']))
    
    print(f"✅ Evaluation completed. Results saved to {output_dir}")
    return metrics


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
    
    # Create test dataset
    pre_dir = os.path.join(args.data_dir, 'pre')
    post_dir = os.path.join(args.data_dir, 'post')
    mask_dir = os.path.join(args.data_dir, 'masks')
    
    dataset = BiTemporalDataset(
        pre_dir=pre_dir,
        post_dir=post_dir,
        mask_dir=mask_dir,
        image_size=config['data']['image_size'],
        normalize=config['data']['normalize']
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=2
    )
    
    print(f"📊 Found {len(dataset)} test samples")
    
    # Create and load model
    model = create_model(config, device)
    model = load_checkpoint(model, args.model, device)
    
    # Run evaluation
    metrics = evaluate_model(model, dataloader, device, args.output_dir)


if __name__ == "__main__":
    main()
