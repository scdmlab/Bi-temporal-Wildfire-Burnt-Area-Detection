"""
Training script for Bi-temporal Wildfire Burnt Area Detection

Author: Tang Sui
Email: tsui5@wisc.edu
"""

import os
import sys
import yaml
import argparse
import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.attention_unet import get_model
from datasets.bitemporal_dataset import BiTemporalDataset, get_transforms
from utils.losses import get_loss_function
from utils.metrics import SegmentationMetrics, plot_confusion_matrix


def setup_logging(log_dir):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_datasets(config):
    """Create train, validation, and test datasets"""
    data_config = config['data']
    
    # Transforms
    train_transform = get_transforms('train', 
                                   image_size=tuple(data_config['image_size']),
                                   normalize=data_config.get('normalize', True))
    val_transform = get_transforms('val', 
                                  image_size=tuple(data_config['image_size']),
                                  normalize=data_config.get('normalize', True))
    
    # Datasets
    train_dataset = BiTemporalDataset(
        pre_dir=data_config['pre_dir'],
        post_dir=data_config['post_dir'],
        mask_dir=data_config['mask_dir'],
        transform=train_transform,
        image_size=tuple(data_config['image_size'])
    )
    
    val_dataset = BiTemporalDataset(
        pre_dir=data_config.get('val_pre_dir', data_config['pre_dir']),
        post_dir=data_config.get('val_post_dir', data_config['post_dir']),
        mask_dir=data_config.get('val_mask_dir', data_config['mask_dir']),
        transform=val_transform,
        image_size=tuple(data_config['image_size'])
    )
    
    # Split dataset if validation directories not provided
    if data_config.get('val_pre_dir') is None:
        total_size = len(train_dataset)
        val_size = int(data_config.get('val_split', 0.2) * total_size)
        train_size = total_size - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(config.get('seed', 42))
        )
    
    return train_dataset, val_dataset


def create_dataloaders(train_dataset, val_dataset, config):
    """Create data loaders"""
    train_config = config['training']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        num_workers=train_config.get('num_workers', 4),
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=train_config.get('num_workers', 4),
        pin_memory=True
    )
    
    return train_loader, val_loader


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, logger, writer=None):
    """Train for one epoch"""
    model.train()
    
    running_loss = 0.0
    metrics = SegmentationMetrics(num_classes=2 if model.num_classes == 1 else model.num_classes)
    
    for batch_idx, batch in enumerate(train_loader):
        pre_images = batch['pre_image'].to(device)
        post_images = batch['post_image'].to(device)
        masks = batch['mask'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        if hasattr(model, 'encode'):  # Bi-temporal model
            outputs = model(pre_images, post_images)
        else:  # Single image model (for comparison)
            outputs = model(torch.cat([pre_images, post_images], dim=1))
        
        # Compute loss
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        running_loss += loss.item()
        metrics.update(outputs, masks)
        
        # Log progress
        if batch_idx % 10 == 0:
            logger.info(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        # TensorBoard logging
        if writer and batch_idx % 100 == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
    
    # Compute epoch metrics
    epoch_loss = running_loss / len(train_loader)
    epoch_metrics = metrics.get_metrics()
    
    # Log epoch results
    logger.info(f'Train Epoch {epoch}: Loss={epoch_loss:.4f}, Acc={epoch_metrics["accuracy"]:.4f}')
    
    if writer:
        writer.add_scalar('Train/EpochLoss', epoch_loss, epoch)
        writer.add_scalar('Train/Accuracy', epoch_metrics['accuracy'], epoch)
        if 'mean_iou' in epoch_metrics:
            writer.add_scalar('Train/mIoU', epoch_metrics['mean_iou'], epoch)
    
    return epoch_loss, epoch_metrics


def validate_epoch(model, val_loader, criterion, device, epoch, logger, writer=None):
    """Validate for one epoch"""
    model.eval()
    
    running_loss = 0.0
    metrics = SegmentationMetrics(num_classes=2 if model.num_classes == 1 else model.num_classes)
    
    with torch.no_grad():
        for batch in val_loader:
            pre_images = batch['pre_image'].to(device)
            post_images = batch['post_image'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            if hasattr(model, 'encode'):  # Bi-temporal model
                outputs = model(pre_images, post_images)
            else:  # Single image model
                outputs = model(torch.cat([pre_images, post_images], dim=1))
            
            # Compute loss
            loss = criterion(outputs, masks)
            
            # Update metrics
            running_loss += loss.item()
            metrics.update(outputs, masks)
    
    # Compute epoch metrics
    epoch_loss = running_loss / len(val_loader)
    epoch_metrics = metrics.get_metrics()
    
    # Log epoch results
    logger.info(f'Val Epoch {epoch}: Loss={epoch_loss:.4f}, Acc={epoch_metrics["accuracy"]:.4f}')
    
    if writer:
        writer.add_scalar('Val/EpochLoss', epoch_loss, epoch)
        writer.add_scalar('Val/Accuracy', epoch_metrics['accuracy'], epoch)
        if 'mean_iou' in epoch_metrics:
            writer.add_scalar('Val/mIoU', epoch_metrics['mean_iou'], epoch)
    
    return epoch_loss, epoch_metrics


def save_checkpoint(model, optimizer, scheduler, epoch, loss, save_path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }
    torch.save(checkpoint, save_path)


def main():
    parser = argparse.ArgumentParser(description='Train Bi-temporal Wildfire Burnt Area Detection Model')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--output-dir', type=str, default='./outputs', help='Output directory')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir / 'logs')
    logger.info(f'Starting training with config: {args.config}')
    
    # Setup TensorBoard
    writer = SummaryWriter(output_dir / 'tensorboard')
    
    # Set random seed
    if 'seed' in config:
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Create datasets and dataloaders
    train_dataset, val_dataset = create_datasets(config)
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, config)
    
    logger.info(f'Train dataset size: {len(train_dataset)}')
    logger.info(f'Val dataset size: {len(val_dataset)}')
    
    # Create model
    model_config = config['model']
    model = get_model(model_config['type'], **model_config.get('params', {}))
    model.to(device)
    
    logger.info(f'Model: {model_config["type"]}')
    logger.info(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Create loss function
    loss_config = config['loss']
    criterion = get_loss_function(loss_config['type'], **loss_config.get('params', {}))
    
    # Create optimizer
    opt_config = config['optimizer']
    if opt_config['type'] == 'adam':
        optimizer = optim.Adam(model.parameters(), **opt_config.get('params', {}))
    elif opt_config['type'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), **opt_config.get('params', {}))
    else:
        raise ValueError(f"Unknown optimizer: {opt_config['type']}")
    
    # Create scheduler
    scheduler = None
    if 'scheduler' in config:
        sched_config = config['scheduler']
        if sched_config['type'] == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, **sched_config.get('params', {}))
        elif sched_config['type'] == 'multistep':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **sched_config.get('params', {}))
        elif sched_config['type'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **sched_config.get('params', {}))
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if args.resume:
        logger.info(f'Resuming from checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    # Training loop
    train_config = config['training']
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, train_config['epochs']):
        logger.info(f'Epoch {epoch}/{train_config["epochs"]-1}')
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, logger, writer
        )
        
        # Validate
        val_loss, val_metrics = validate_epoch(
            model, val_loader, criterion, device, epoch, logger, writer
        )
        
        # Update scheduler
        if scheduler:
            scheduler.step()
            writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Save checkpoint
        if epoch % train_config.get('save_freq', 10) == 0:
            save_path = output_dir / f'checkpoint_epoch_{epoch}.pth'
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, save_path)
            logger.info(f'Checkpoint saved: {save_path}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = output_dir / 'best_model.pth'
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, save_path)
            logger.info(f'Best model saved: {save_path}')
    
    # Save final model
    final_path = output_dir / 'final_model.pth'
    save_checkpoint(model, optimizer, scheduler, train_config['epochs']-1, val_loss, final_path)
    logger.info(f'Final model saved: {final_path}')
    
    writer.close()
    logger.info('Training completed!')


if __name__ == '__main__':
    main()
