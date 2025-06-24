# GitHub Project Structure Documentation

This is a streamlined bi-temporal attention U-Net wildfire burnt area detection project structure suitable for uploading to GitHub.

## 📁 Project Structure

```
bitemporal-attention-unet/
├── 📂 configs/                 # Configuration files
│   ├── config.yaml            # Main configuration file
│   └── config_example.yaml    # Configuration example file
├── 📂 datasets/               # Dataset processing
│   ├── __init__.py
│   └── bitemporal_dataset.py  # Bi-temporal dataset class
├── 📂 examples/               # Examples and tutorials
│   └── quick_start.ipynb      # Quick start Jupyter tutorial
├── 📂 models/                 # Model definitions
│   ├── __init__.py
│   └── attention_unet.py      # Attention U-Net models
├── 📂 scripts/                # Training and inference scripts
│   ├── train.py              # Training script
│   ├── inference.py          # Inference script
│   └── evaluate.py           # Evaluation script
├── 📂 utils/                  # Utility functions
│   ├── __init__.py
│   ├── data_utils.py         # Data processing utilities
│   ├── losses.py             # Loss functions
│   ├── metrics.py            # Evaluation metrics
│   └── visualize.py          # Visualization tools
├── 📄 .gitignore             # Git ignore file
├── 📄 LICENSE                # Open source license
├── 📄 README.md              # Project documentation
└── 📄 requirements.txt       # Python dependencies
```

## 🎯 Core Files Description

### 📋 Model Files (`models/`)
- **`attention_unet.py`**: Contains all model architectures
  - `AttentionUNet`: Single-temporal attention U-Net
  - `BiTemporalUNet`: Bi-temporal U-Net
  - `BiTemporalAttentionUNet`: Bi-temporal attention U-Net (recommended)

### 📊 Data Processing (`datasets/`)
- **`bitemporal_dataset.py`**: Bi-temporal dataset loader
  - Supports multiple image formats
  - Automatic data augmentation
  - Flexible file naming

### 🚀 Scripts (`scripts/`)
- **`train.py`**: Complete training pipeline
- **`inference.py`**: Batch inference script
- **`evaluate.py`**: Model evaluation script

### 🛠️ Utilities (`utils/`)
- **`losses.py`**: Multiple loss functions (BCE, Dice, Focal, Combined)
- **`metrics.py`**: Evaluation metrics (IoU, F1, AUC, etc.)
- **`visualize.py`**: Result visualization tools
- **`data_utils.py`**: Data preprocessing and analysis tools

### ⚙️ Configuration (`configs/`)
- **`config.yaml`**: Main configuration file
- **`config_example.yaml`**: Detailed configuration examples

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
Organize your data in the following structure:
```
data/
├── pre_fire/     # Pre-fire images
├── post_fire/    # Post-fire images
└── burnt_masks/  # Burnt area masks (optional)
```

### 3. Modify Configuration
Edit `configs/config.yaml` and update data paths:
```yaml
data:
  pre_dir: "/path/to/your/pre_fire/images"
  post_dir: "/path/to/your/post_fire/images"
  mask_dir: "/path/to/your/burnt_masks"
```

### 4. Train Model
```bash
python scripts/train.py --config configs/config.yaml --output-dir outputs/
```

### 5. Run Inference
```bash
python scripts/inference.py --config configs/config.yaml --model outputs/best_model.pth --pre-dir data/test/pre_fire --post-dir data/test/post_fire --output-dir results/
```

## 📚 Features

### ✨ Model Architecture
- 🎯 Bi-temporal architecture designed specifically for wildfire burnt area detection
- 🔍 Attention mechanism for enhanced feature selection
- 🏗️ Multi-scale feature fusion
- ⚡ Efficient memory usage

### 📊 Data Processing
- 🔄 Automatic data augmentation
- 📏 Flexible image size handling
- 🗂️ Multiple file format support
- ✅ Data integrity checking

### 🎨 Visualization
- 📈 Training curve plotting
- 🖼️ Prediction result overlays
- 📊 Confusion matrices and ROC curves
- 🔍 Attention map visualization

### 🎛️ Configuration Flexibility
- 📝 YAML configuration files
- 🔧 Multiple loss function combinations
- 📊 Rich evaluation metrics
- ⚙️ Adjustable training parameters

## 📦 Differences from Original Project

### 🗂️ Streamlined Content
- ❌ Removed dataset files
- ❌ Removed temporary files and cache
- ❌ Removed experimental result files
- ❌ Removed redundant code

### ✅ Core Features Retained
- ✅ Complete model implementation
- ✅ Training and inference pipelines
- ✅ Evaluation and visualization tools
- ✅ Detailed documentation and examples

### 🆕 New Features
- 📓 Jupyter tutorial examples
- 🔧 Unified configuration management
- 📊 Enhanced visualization capabilities
- 🛠️ Data preprocessing tools

## 📄 Estimated File Size

Estimated size of the entire GitHub project (excluding data):
- Core code: ~50KB
- Documentation and examples: ~100KB
- Configuration files: ~10KB
- **Total: <200KB**

Perfect for GitHub upload, fast cloning, easy sharing and collaboration.

## 🤝 Usage Recommendations

1. After **Fork or Clone**, first read `README.md`
2. Run `examples/quick_start.ipynb` to understand basic usage
3. Modify `configs/config.yaml` according to your data
4. Use `utils/data_utils.py` to analyze your dataset
5. Start training and experimenting!

This structure maintains the complete functionality of the original project while being suitable for open source sharing and academic exchange.
