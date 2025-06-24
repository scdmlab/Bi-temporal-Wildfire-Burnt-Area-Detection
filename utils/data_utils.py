"""
Data preprocessing utilities for Bi-temporal Wildfire Burnt Area Detection

Author: Tang Sui
Email: tsui5@wisc.edu
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
from tqdm import tqdm
import json


def analyze_dataset(pre_dir, post_dir, mask_dir=None):
    """
    Analyze dataset statistics
    
    Args:
        pre_dir (str): Directory containing pre-event images
        post_dir (str): Directory containing post-event images
        mask_dir (str, optional): Directory containing masks
    
    Returns:
        dict: Dataset statistics
    """
    stats = {
        'num_samples': 0,
        'image_sizes': [],
        'file_formats': set(),
        'has_masks': mask_dir is not None
    }
    
    # Get file lists
    pre_files = sorted([f for f in os.listdir(pre_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))])
    post_files = sorted([f for f in os.listdir(post_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))])
    
    if mask_dir:
        mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.npy'))])
        stats['num_samples'] = min(len(pre_files), len(post_files), len(mask_files))
    else:
        stats['num_samples'] = min(len(pre_files), len(post_files))
    
    print(f"📊 Dataset Analysis")
    print(f"Pre-event images: {len(pre_files)}")
    print(f"Post-event images: {len(post_files)}")
    if mask_dir:
        print(f"Mask files: {len(mask_files)}")
    print(f"Valid pairs: {stats['num_samples']}")
    
    # Analyze image properties
    for i in tqdm(range(min(10, stats['num_samples'])), desc="Analyzing samples"):
        # Pre image
        pre_path = os.path.join(pre_dir, pre_files[i])
        pre_img = Image.open(pre_path)
        stats['image_sizes'].append(pre_img.size)
        stats['file_formats'].add(pre_img.format)
        
        # Post image
        post_path = os.path.join(post_dir, post_files[i])
        post_img = Image.open(post_path)
        
        # Check if sizes match
        if pre_img.size != post_img.size:
            print(f"⚠️  Size mismatch in pair {i}: {pre_img.size} vs {post_img.size}")
    
    # Summary
    if stats['image_sizes']:
        unique_sizes = list(set(stats['image_sizes']))
        print(f"Image sizes found: {unique_sizes}")
        if len(unique_sizes) == 1:
            print(f"✅ All images have consistent size: {unique_sizes[0]}")
        else:
            print(f"⚠️  Multiple image sizes found")
    
    print(f"File formats: {list(stats['file_formats'])}")
    
    return stats


def check_file_naming(pre_dir, post_dir, mask_dir=None):
    """
    Check file naming consistency between directories
    
    Args:
        pre_dir (str): Directory containing pre-event images
        post_dir (str): Directory containing post-event images
        mask_dir (str, optional): Directory containing masks
    
    Returns:
        dict: Naming analysis results
    """
    print("🔍 Checking file naming consistency...")
    
    # Get base names (without extensions)
    pre_files = os.listdir(pre_dir)
    post_files = os.listdir(post_dir)
    
    pre_names = {os.path.splitext(f)[0] for f in pre_files if not f.startswith('.')}
    post_names = {os.path.splitext(f)[0] for f in post_files if not f.startswith('.')}
    
    common_names = pre_names & post_names
    pre_only = pre_names - post_names
    post_only = post_names - pre_names
    
    results = {
        'common_pairs': len(common_names),
        'pre_only': list(pre_only),
        'post_only': list(post_only),
        'valid_pairs': sorted(list(common_names))
    }
    
    print(f"Common pairs: {len(common_names)}")
    if pre_only:
        print(f"Pre-only files: {len(pre_only)}")
        if len(pre_only) <= 5:
            print(f"  {pre_only}")
    if post_only:
        print(f"Post-only files: {len(post_only)}")
        if len(post_only) <= 5:
            print(f"  {post_only}")
    
    # Check masks if provided
    if mask_dir:
        mask_files = os.listdir(mask_dir)
        mask_names = {os.path.splitext(f)[0] for f in mask_files if not f.startswith('.')}
        
        mask_common = common_names & mask_names
        missing_masks = common_names - mask_names
        extra_masks = mask_names - common_names
        
        results['mask_pairs'] = len(mask_common)
        results['missing_masks'] = list(missing_masks)
        results['extra_masks'] = list(extra_masks)
        
        print(f"Mask files matching pairs: {len(mask_common)}")
        if missing_masks:
            print(f"Missing masks: {len(missing_masks)}")
        if extra_masks:
            print(f"Extra masks: {len(extra_masks)}")
    
    return results


def create_train_val_split(data_dir, output_dir, val_ratio=0.2, seed=42):
    """
    Create train/validation split
    
    Args:
        data_dir (str): Directory containing pre_fire/post_fire/burnt_masks subdirectories
        output_dir (str): Output directory for split data
        val_ratio (float): Ratio for validation set
        seed (int): Random seed for reproducibility
    """
    np.random.seed(seed)
    
    pre_dir = os.path.join(data_dir, 'pre')
    post_dir = os.path.join(data_dir, 'post')
    mask_dir = os.path.join(data_dir, 'masks')
    
    # Get valid pairs
    naming_results = check_file_naming(pre_dir, post_dir, mask_dir if os.path.exists(mask_dir) else None)
    valid_names = naming_results['valid_pairs']
    
    # Random shuffle and split
    valid_names = np.array(valid_names)
    indices = np.random.permutation(len(valid_names))
    
    val_size = int(len(valid_names) * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    train_names = valid_names[train_indices]
    val_names = valid_names[val_indices]
    
    print(f"📊 Creating train/val split:")
    print(f"  Total samples: {len(valid_names)}")
    print(f"  Training: {len(train_names)}")
    print(f"  Validation: {len(val_names)}")
    
    # Create output directories
    for split in ['train', 'val']:
        for subdir in ['pre', 'post', 'masks']:
            os.makedirs(os.path.join(output_dir, split, subdir), exist_ok=True)
    
    # Copy files
    def copy_files(names, split):
        for name in tqdm(names, desc=f"Copying {split} files"):
            # Find files with this base name
            for subdir in ['pre', 'post', 'masks']:
                src_dir = os.path.join(data_dir, subdir)
                dst_dir = os.path.join(output_dir, split, subdir)
                
                if not os.path.exists(src_dir):
                    continue
                
                # Find matching file
                for ext in ['.png', '.jpg', '.jpeg', '.npy']:
                    src_file = os.path.join(src_dir, name + ext)
                    if os.path.exists(src_file):
                        dst_file = os.path.join(dst_dir, name + ext)
                        shutil.copy2(src_file, dst_file)
                        break
    
    copy_files(train_names, 'train')
    copy_files(val_names, 'val')
    
    # Save split information
    split_info = {
        'train_samples': train_names.tolist(),
        'val_samples': val_names.tolist(),
        'val_ratio': val_ratio,
        'seed': seed
    }
    
    with open(os.path.join(output_dir, 'split_info.json'), 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"✅ Split completed. Files saved to {output_dir}")


def resize_images(input_dir, output_dir, target_size=(224, 224), keep_aspect_ratio=False):
    """
    Resize all images in a directory
    
    Args:
        input_dir (str): Input directory
        output_dir (str): Output directory
        target_size (tuple): Target size (width, height)
        keep_aspect_ratio (bool): Whether to keep aspect ratio
    """
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
    
    for filename in tqdm(image_files, desc="Resizing images"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        with Image.open(input_path) as img:
            if keep_aspect_ratio:
                img.thumbnail(target_size, Image.Resampling.LANCZOS)
                # Create new image with target size and paste resized image
                new_img = Image.new('RGB', target_size, (0, 0, 0))
                offset = ((target_size[0] - img.width) // 2, 
                         (target_size[1] - img.height) // 2)
                new_img.paste(img, offset)
                new_img.save(output_path)
            else:
                resized = img.resize(target_size, Image.Resampling.LANCZOS)
                resized.save(output_path)
    
    print(f"✅ Resized {len(image_files)} images to {target_size}")


def convert_masks_to_binary(mask_dir, output_dir, change_class_id=1):
    """
    Convert multi-class masks to binary change/no-change masks
    
    Args:
        mask_dir (str): Directory containing mask files
        output_dir (str): Output directory for binary masks
        change_class_id (int): Class ID that represents change
    """
    os.makedirs(output_dir, exist_ok=True)
    
    mask_files = [f for f in os.listdir(mask_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.npy'))]
    
    for filename in tqdm(mask_files, desc="Converting masks"):
        input_path = os.path.join(mask_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        if filename.endswith('.npy'):
            # Handle numpy files
            mask = np.load(input_path)
            binary_mask = (mask == change_class_id).astype(np.uint8)
            np.save(output_path, binary_mask)
        else:
            # Handle image files
            mask = np.array(Image.open(input_path))
            binary_mask = (mask == change_class_id).astype(np.uint8) * 255
            Image.fromarray(binary_mask).save(output_path)
    
    print(f"✅ Converted {len(mask_files)} masks to binary")


def visualize_random_samples(pre_dir, post_dir, mask_dir=None, n_samples=4, save_path=None):
    """
    Visualize random samples from dataset
    
    Args:
        pre_dir (str): Directory containing pre-event images
        post_dir (str): Directory containing post-event images
        mask_dir (str, optional): Directory containing masks
        n_samples (int): Number of samples to visualize
        save_path (str, optional): Path to save visualization
    """
    # Get valid pairs
    naming_results = check_file_naming(pre_dir, post_dir, mask_dir)
    valid_names = naming_results['valid_pairs']
    
    if len(valid_names) < n_samples:
        n_samples = len(valid_names)
        print(f"⚠️  Only {n_samples} valid pairs available")
    
    # Select random samples
    selected_names = np.random.choice(valid_names, n_samples, replace=False)
    
    # Create visualization
    n_cols = 3 if mask_dir else 2
    fig, axes = plt.subplots(n_samples, n_cols, figsize=(n_cols * 4, n_samples * 4))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, name in enumerate(selected_names):
        # Load pre image
        for ext in ['.png', '.jpg', '.jpeg']:
            pre_path = os.path.join(pre_dir, name + ext)
            if os.path.exists(pre_path):
                pre_img = Image.open(pre_path)
                break
        
        # Load post image
        for ext in ['.png', '.jpg', '.jpeg']:
            post_path = os.path.join(post_dir, name + ext)
            if os.path.exists(post_path):
                post_img = Image.open(post_path)
                break
        
        # Display images
        axes[i, 0].imshow(pre_img)
        axes[i, 0].set_title(f'Pre-event: {name}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(post_img)
        axes[i, 1].set_title(f'Post-event: {name}')
        axes[i, 1].axis('off')
        
        # Load and display mask if available
        if mask_dir:
            mask_loaded = False
            for ext in ['.png', '.jpg', '.jpeg', '.npy']:
                mask_path = os.path.join(mask_dir, name + ext)
                if os.path.exists(mask_path):
                    if ext == '.npy':
                        mask = np.load(mask_path)
                    else:
                        mask = np.array(Image.open(mask_path))
                    
                    axes[i, 2].imshow(mask, cmap='gray')
                    axes[i, 2].set_title(f'Mask: {name}')
                    axes[i, 2].axis('off')
                    mask_loaded = True
                    break
            
            if not mask_loaded:
                axes[i, 2].text(0.5, 0.5, 'No mask found', 
                               ha='center', va='center', transform=axes[i, 2].transAxes)
                axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Visualization saved to {save_path}")
    else:
        plt.show()


def create_data_summary_report(data_dir, output_file):
    """
    Create a comprehensive data summary report
    
    Args:
        data_dir (str): Data directory containing pre_fire/post_fire/burnt_masks
        output_file (str): Output file for the report
    """
    report = []
    report.append("# Dataset Summary Report\n")
    
    # Basic statistics
    pre_dir = os.path.join(data_dir, 'pre')
    post_dir = os.path.join(data_dir, 'post')
    mask_dir = os.path.join(data_dir, 'masks')
    
    if os.path.exists(pre_dir) and os.path.exists(post_dir):
        stats = analyze_dataset(pre_dir, post_dir, mask_dir if os.path.exists(mask_dir) else None)
        naming = check_file_naming(pre_dir, post_dir, mask_dir if os.path.exists(mask_dir) else None)
        
        report.append(f"## Dataset Statistics\n")
        report.append(f"- Total valid pairs: {naming['common_pairs']}\n")
        report.append(f"- Pre-only files: {len(naming['pre_only'])}\n")
        report.append(f"- Post-only files: {len(naming['post_only'])}\n")
        
        if 'mask_pairs' in naming:
            report.append(f"- Mask pairs: {naming['mask_pairs']}\n")
            report.append(f"- Missing masks: {len(naming['missing_masks'])}\n")
        
        report.append(f"- File formats: {list(stats['file_formats'])}\n")
        
        if stats['image_sizes']:
            unique_sizes = list(set(stats['image_sizes']))
            report.append(f"- Image sizes: {unique_sizes}\n")
        
        report.append(f"\n")
    
    # Save report
    with open(output_file, 'w') as f:
        f.writelines(report)
    
    print(f"✅ Data summary report saved to {output_file}")


if __name__ == "__main__":
    # Example usage
    print("Data preprocessing utilities loaded")
    print("Available functions:")
    print("- analyze_dataset()")
    print("- check_file_naming()")
    print("- create_train_val_split()")
    print("- resize_images()")
    print("- convert_masks_to_binary()")
    print("- visualize_random_samples()")
    print("- create_data_summary_report()")
