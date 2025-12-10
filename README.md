# AICUP2025_II_TEAM8011

**Team 8011's solution for AICUP 2025 aortic-valve-detection**

This repository contains the implementation of a YOLO-based object detection system using cross-validation training for the 2025 AI CUP Competition Track II.

## Overview

This project implements a robust object detection pipeline using YOLOv11 with 5-fold cross-validation to improve model generalization and performance. The solution includes custom data preprocessing, stratified k-fold splitting, and optimized training configurations.

## Repository Structure

AICUP2025_II_TEAM8011/
├── datasets/
│   ├── fixed_yolo.py          # YOLO dataset format correction utilities
│   ├── pixel_distribution.py   # Pixel-level data analysis tools
│   ├── spilit_fold.py         # K-fold cross-validation data splitting
│   └── yolo/                  # YOLO format dataset directory
├── ultralytics/               # Custom Ultralytics YOLO implementation
├── upload/                    # Model outputs and submission files
└── train_yolo_fold.py        # Main training script for k-fold CV


## Features

- **YOLOv11 Architecture**: State-of-the-art object detection model (yolo11s)
- **5-Fold Cross-Validation**: Stratified k-fold splitting for robust evaluation
- **Single-Class Detection**: Optimized for single object class scenarios
- **Data Augmentation**: Comprehensive augmentation pipeline including:
    - Translation (0.1)
    - Scale variation (0.5)
    - Horizontal flipping (0.5)
    - Mosaic augmentation
    - Dropout regularization (0.1)
- **Training Optimizations**:
    - Cosine learning rate scheduling
    - Model compilation for faster inference
    - Dataset caching for improved I/O
    - Deterministic seed control per fold


## Requirements

bash
pip install ultralytics

Additional dependencies:

- Python >= 3.8
- PyTorch >= 1.8
- CUDA-compatible GPU (recommended)


## Dataset Preparation

### 1. Data Analysis

Analyze pixel distribution and dataset statistics:

bash
python datasets/pixel_distribution.py


### 2. Format Correction

Fix YOLO annotation format if needed:

bash
python datasets/fixed_yolo.py


### 3. Create K-Fold Splits

Generate 5-fold cross-validation splits:

bash
python datasets/spilit_fold.py

This creates the following structure:

datasets/all_yolo_data/yolo_5fold/
├── fold_1/
│   └── baseline.yaml
├── fold_2/
│   └── baseline.yaml
├── fold_3/
│   └── baseline.yaml
├── fold_4/
│   └── baseline.yaml
└── fold_5/
    └── baseline.yaml


## Training

### Quick Start

Train all folds sequentially:

bash
python train_yolo_fold.py


### Training Configuration

The training script uses the following hyperparameters:

- **Model**: YOLOv11s pre-trained weights
- **Image Size**: 768×768
- **Batch Size**: 32
- **Epochs**: 100
- **Device**: GPU 0
- **Augmentation**:
    - Translation: 10%
    - Scale: 50%
    - Horizontal flip: 50%
    - Mosaic: enabled
    - Dropout: 10%


### Custom Training

Modify train_yolo_fold.py to adjust hyperparameters:

python
results = model.train(
    data=f"./datasets/all_yolo_data/yolo_5fold/fold_{fold}/baseline.yaml",
    epochs=100,
    batch=32,
    imgsz=768,
    # ... other parameters
)


## Model Outputs

Trained models are saved to:

/root/aicup2025_II/upload/model/detect/all/yolo11s_fold{1-4}/

Each fold directory contains:

- weights/best.pt - Best checkpoint based on validation metrics
- weights/last.pt - Final epoch checkpoint
- Training logs and metrics
- Validation predictions


## Evaluation

The 5-fold cross-validation provides robust performance estimation. Average metrics across folds for final evaluation.

## Inference

python
from ultralytics import YOLO

# Load trained model
model = YOLO('path/to/best.pt')

# Run inference
results = model.predict(
    source='path/to/test/images',
    imgsz=768,
    conf=0.25,  # confidence threshold
    iou=0.45    # NMS IoU threshold
)
