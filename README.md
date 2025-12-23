# DETR Object Detection on COCO Dataset

This project implements object detection using DETR (Detection Transformer) on a subset of the COCO dataset, focusing on 10 specific object classes. The project includes training, evaluation, data augmentation, and class imbalance analysis.

## Overview

The project fine-tunes a pre-trained DETR model (`facebook/detr-resnet-50`) for detecting 10 object classes:
- **Animals**: cat, dog, horse, cow, sheep
- **Vehicles**: bus, truck, train, motorcycle, airplane

## Project Structure

```
cv_hw2/
├── train.py                 # Training script for base model
├── train_augmented.py       # Training script with augmented data
├── trainer.py               # Custom training loop implementation
├── eval.py                  # Model evaluation and metrics
├── augment.py               # Data augmentation using Stable Diffusion
├── dataset.py               # COCO dataset loader
├── classes_imbalance.py     # Class imbalance analysis
├── check_annotations.py     # Annotation verification
├── ckpts/                   # Model checkpoints (base training)
├── ckpts_augmented/         # Model checkpoints (augmented training)
├── runs/                    # TensorBoard logs (base training)
├── runs_augmented/          # TensorBoard logs (augmented training)
└── data/                    # COCO dataset directory
    └── coco/
        ├── train2017/       # Training images
        ├── val2017/         # Validation images
        ├── train2017_synthetic/  # Generated synthetic images
        └── annotations/     # COCO annotation files
```

## Features

### 1. Training
- **Base Training** (`train.py`): Trains DETR on original COCO data
- **Augmented Training** (`train_augmented.py`): Trains on data augmented with synthetic images
- Custom training loop with:
  - Mixed precision training (FP16)
  - Gradient clipping
  - Separate learning rates for backbone and detection head
  - TensorBoard logging
  - Automatic checkpoint saving

### 2. Data Augmentation
- Uses Stable Diffusion with ControlNet (Canny edge detection) to generate synthetic images
- Maintains original bounding box annotations
- Configurable augmentation factor and rare class threshold
- Generates images for underrepresented classes

### 3. Evaluation
- Per-class accuracy calculation (IoU-based matching)
- Confusion matrix visualization
- Detailed error statistics (TP, FN, FP breakdown)
- Generates SVG plots for analysis

### 4. Class Imbalance Analysis
- Analyzes distribution of classes in the dataset
- Generates bar charts showing instance counts and percentages
- Calculates imbalance metrics (ratio, coefficient of variation)

## Requirements

```bash
torch
transformers
pycocotools
pillow
matplotlib
tqdm
tensorboard
diffusers
opencv-python
numpy
```

## Installation

1. Clone the repository and navigate to the project directory:
```bash
cd /home/alisa/cv_hw2
```

2. Install dependencies:
```bash
pip install torch transformers pycocotools pillow matplotlib tqdm tensorboard diffusers opencv-python numpy
```

3. Download COCO dataset and place it in the expected directory structure:
```
data/coco/
├── train2017/
├── val2017/
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

## Usage

### Training

#### Base Training
```bash
python train.py
```

This will:
- Load the pre-trained DETR model
- Train on COCO training set
- Save checkpoints to `ckpts/`
- Log metrics to TensorBoard in `runs/`

#### Training with Augmented Data

1. First, generate synthetic images:
```bash
python augment.py
```

2. Then train with augmented data:
```bash
python train_augmented.py
```

This uses the augmented annotation file (`instances_augmented.json`) which includes both real and synthetic images.

### Evaluation

Evaluate a trained model:
```bash
python eval.py \
    --checkpoint /home/alisa/cv_hw2/ckpts/detr_epoch_10.pt \
    --data_dir /home/alisa/homework2/data/coco \
    --output_svg per_class_accuracy.svg \
    --output_confusion_all_svg confusion_all_classes.svg
```

Arguments:
- `--checkpoint`: Path to model checkpoint
- `--data_dir`: Path to COCO data directory
- `--batch_size`: Batch size for evaluation (default: 8)
- `--device`: Computation device (default: cuda if available)
- `--output_svg`: Output path for per-class accuracy plot
- `--output_confusion_all_svg`: Output path for confusion matrix plot

### Class Imbalance Analysis

Analyze class distribution in the dataset:
```bash
python classes_imbalance.py \
    --data_dir /home/alisa/homework2/data/coco \
    --split train \
    --both_splits
```

This generates:
- `classes_imbalance_train.svg`: Bar chart of instance counts
- `classes_imbalance_percentage_train.svg`: Percentage distribution

## Configuration

### Training Parameters

In `train.py` and `train_augmented.py`, you can modify:
- `batch_size`: Batch size (default: 16)
- `epochs`: Number of training epochs
- `checkpoint_dir`: Directory for saving checkpoints
- `log_dir`: Directory for TensorBoard logs

### Trainer Parameters

In `trainer.py`, the `Trainer` class accepts:
- `lr`: Learning rate for detection head (default: 1e-4)
- `backbone_lr`: Learning rate for backbone (default: 1e-5)
- `weight_decay`: Weight decay (default: 1e-4)
- `save_freq`: Checkpoint save frequency in epochs (default: 5)
- `freeze_backbone`: Whether to freeze backbone parameters

### Augmentation Parameters

In `augment.py`, you can configure:
- `AUG_FACTOR`: Number of synthetic images per real image for rare classes
- `RARE_THRESHOLD`: Minimum number of images below which a class is considered rare
- `TARGET_CLASSES`: List of classes to augment

## Model Architecture

The project uses **DETR (Detection Transformer)**:
- Backbone: ResNet-50
- Transformer encoder-decoder architecture
- 10 object classes + 1 "no object" class
- End-to-end object detection without anchor boxes or NMS

## Evaluation Metrics

The evaluation script computes:
- **Per-class Accuracy**: Percentage of ground-truth boxes correctly detected (IoU ≥ 0.5)
- **Confusion Matrix**: Shows class-wise prediction errors
- **Error Statistics**:
  - TP (True Positives): Correct detections
  - FN_missed: Ground-truth objects with no predictions
  - FN_loc: Predictions with IoU < threshold but correct class
  - FN_misclass: Predictions with IoU ≥ threshold but wrong class
  - FP (False Positives): Predictions with no matching ground-truth

## Output Files

### Training Outputs
- Checkpoints: `ckpts/detr_epoch_{N}.pt` or `ckpts_augmented/detr_epoch_{N}.pt`
- TensorBoard logs: `runs/` or `runs_augmented/`

### Evaluation Outputs
- `per_class_accuracy.svg`: Bar chart of per-class detection accuracy
- `confusion_all_classes.svg`: Normalized confusion matrix heatmap

### Analysis Outputs
- `classes_imbalance_train.svg`: Class distribution bar chart
- `classes_imbalance_percentage_train.svg`: Percentage distribution chart

## Notes

- The model expects images and annotations in COCO format
- Synthetic images are stored separately in `train2017_synthetic/` but referenced in the augmented annotation file
- The dataset loader automatically handles both real and synthetic images based on file path prefixes
- Training uses mixed precision (FP16) for faster training and lower memory usage
- Gradient clipping (max norm: 0.1) is applied to stabilize training

## Troubleshooting

1. **CUDA out of memory**: Reduce `batch_size` in training scripts
2. **Missing annotations**: Ensure COCO dataset is properly downloaded and paths are correct
3. **Synthetic images not loading**: Check that `augment.py` has been run and images are in `train2017_synthetic/`

## License

This project is for educational purposes (homework assignment).

