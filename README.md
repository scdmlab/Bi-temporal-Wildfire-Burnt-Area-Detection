# Bi-temporal Attention U-Net for Wildfire Burnt Area Detection

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A PyTorch implementation of Bi-temporal Attention U-Net for wildfire burnt area detection in remote sensing images.

## 🔥 Features

- **Multiple Model Architectures**: U-Net, Attention U-Net, Bi-temporal U-Net, Bi-temporal Attention U-Net
- **Flexible Loss Functions**: BCE, Dice, Focal, IoU, and Combined losses
- **Comprehensive Metrics**: Accuracy, IoU, F1-score, Precision, Recall, ROC-AUC
- **Easy Configuration**: YAML-based configuration system
- **Visualization Tools**: Training curves, confusion matrices, ROC curves
- **Production Ready**: Modular design with proper logging and checkpointing

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/scdmlab/Bi-temporal-Wildfire-Burnt-Area-Detection.git
cd Bi-temporal-Wildfire-Burnt-Area-Detection

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

Organize your data in the following structure:
```
data/
├── pre_fire/     # Pre-fire images
├── post_fire/    # Post-fire images
└── burnt_masks/  # Burnt area masks (binary: 0=no burning, 1=burnt)
```

### Training

1. Update the configuration file `configs/config.yaml` with your data paths
2. Run training:

```bash
python scripts/train.py --config configs/config.yaml --output-dir outputs/
```

### Inference

```bash
python scripts/inference.py --model outputs/best_model.pth --pre-dir data/test/pre_fire --post-dir data/test/post_fire --output-dir results/
```

## 📊 Model Architectures

### 1. Bi-temporal U-Net
- Shared encoder for both temporal images
- Feature fusion at multiple scales
- Designed specifically for wildfire burnt area detection

### 2. Bi-temporal Attention U-Net
- Incorporates attention gates for better feature selection
- Attention-guided feature fusion
- Superior performance on complex wildfire detection tasks

## 🛠️ Configuration

The `configs/config.yaml` file allows you to configure:

- **Data paths and preprocessing**
- **Model architecture and parameters**
- **Loss function and weights**
- **Optimizer and learning rate scheduler**
- **Training hyperparameters**

Example configuration:
```yaml
model:
  type: "bitemporal_attention_unet"
  params:
    in_channels: 3
    num_classes: 1
    channels: [64, 128, 256, 512, 1024]

loss:
  type: "combined"
  params:
    ce_weight: 1.0
    dice_weight: 1.0

training:
  epochs: 100
  batch_size: 4
  lr: 0.001
```

## 📈 Results

The model achieves state-of-the-art performance on various wildfire detection benchmarks:

- **Overall Accuracy**: >95%
- **Burnt Area Detection F1**: >85%
- **Mean IoU**: >80%

## 🔧 Advanced Usage

### Custom Loss Functions

```python
from utils.losses import get_loss_function

# Create custom combined loss
loss_fn = get_loss_function(
    'combined', 
    ce_weight=1.0, 
    dice_weight=1.0, 
    focal_weight=0.5
)
```

### Custom Datasets

```python
from datasets.bitemporal_dataset import BiTemporalDataset

dataset = BiTemporalDataset(
    pre_dir='path/to/pre_fire',
    post_dir='path/to/post_fire',
    mask_dir='path/to/burnt_masks',
    image_size=(224, 224)
)
```

### Model Creation

```python
from models.attention_unet import get_model

# Create Bi-temporal Attention U-Net
model = get_model(
    'bitemporal_attention_unet',
    num_classes=1,
    channels=[64, 128, 256, 512, 1024]
)
```

## 📊 Visualization

The framework includes comprehensive visualization tools:

- Training/validation curves
- Confusion matrices
- ROC and PR curves
- Burnt area detection overlays

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you find this work useful, please consider citing:
- [BiAU-Net: Wildfire burnt area mapping using bi-temporal Sentinel-2 imagery and U-Net with attention mechanism](https://www.sciencedirect.com/science/article/pii/S1569843224003881)

```bibtex
@article{sui2024biau,
  title={BiAU-Net: Wildfire burnt area mapping using bi-temporal Sentinel-2 imagery and U-Net with attention mechanism},
  author={Sui, Tang and Huang, Qunying and Wu, Mingda and Wu, Meiliu and Zhang, Zhou},
  journal={International Journal of Applied Earth Observation and Geoinformation},
  volume={132},
  pages={104034},
  year={2024},
  publisher={Elsevier}
}
```

## 🔗 Related Work from Our Lab

### Publications:

- [A Remote Sensing Spectral Index Guided Bitemporal Residual Attention Network for Wildfire Burn Severity Mapping](https://ieeexplore.ieee.org/document/10680302)
- [Empowering Wildfire Damage Assessment with Bi-temporal Sentinel-2 Data and Deep Learning](https://ui.adsabs.harvard.edu/abs/2023AGUFMNH21A..03H/abstract)
- [Pixel-wise Wildfire Burn Severity Classification with Bi-temporal Sentinel-2 Data and Deep Learning](https://dl.acm.org/doi/abs/10.1145/3627377.3627433?)

### Citations:

```bibtex
@article{wu2024remote,
  title={A Remote Sensing Spectral Index Guided Bitemporal Residual Attention Network for Wildfire Burn Severity Mapping},
  author={Wu, Mingda and Huang, Qunying and Sui, Tang and Peng, Bo and Yu, Manzhu},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year={2024},
  publisher={IEEE}
}

@inproceedings{huang2023empowering,
  title={Empowering Wildfire Damage Assessment with Bi-temporal Sentinel-2 Data and Deep Learning},
  author={Huang, Qunying and Wu, Mingda and Sui, Tang},
  booktitle={AGU Fall Meeting Abstracts},
  volume={2023},
  pages={NH21A--03},
  year={2023}
}

@inproceedings{wu2023pixel,
  title={Pixel-wise Wildfire Burn Severity Classification with Bi-temporal Sentinel-2 Data and Deep Learning},
  author={Wu, Mingda and Huang, Qunying and Sui, Tang and Wu, Meiliu},
  booktitle={Proceedings of the 2023 6th International Conference on Big Data Technologies},
  pages={360--364},
  year={2023}
}
```

## 📞 Contact

- **Author**: Tang Sui
- **Email**: tsui5@wisc.edu
- **GitHub**: [@scdmlab](https://github.com/scdmlab)

---

⭐ **Star this repository if you found it helpful!**
