# Deep Learning Features in MLCourseGUI

This document outlines the features and implementation of the Deep Learning tab within the MLCourseGUI application. This section is enhanced to enable interactive construction, training, evaluation, and visualization of neural network architectures without requiring manual scripting.

## Overview

The Deep Learning tab provides a dynamic and flexible interface for users to build and train neural networks through a modular GUI. It supports configurable layer structures, popular optimizers, learning rate schedules, regularization techniques, data augmentation, and visual diagnostics to facilitate experimentation and learning.

## Features by Module

### 1. Layer Configuration

**Description**: Sequential layer builder supporting dense, convolutional, recurrent, and normalization layers.

**Supported Layers**:
- Dense
- Conv2D
- MaxPooling2D
- Flatten
- Dropout
- BatchNormalization
- LSTM
- GRU

**Options**:
- Activation functions: ReLU, Sigmoid, Tanh
- Adjustable units, filters, kernel sizes
- Return sequences toggle for RNNs

**Usage**:
1. Select a layer type from the dialog
2. Configure parameters (e.g., units, activation)
3. Dynamically add, remove, or reset layers

### 2. Training Configuration

**Description**: Adjustable training hyperparameters and strategies for controlling the learning process.

**Parameters**:
- Epochs
- Batch size
- Optimizer: Adam, SGD, RMSprop
- Learning Rate
- Loss Function: sparse_categorical_crossentropy, categorical_crossentropy, MSE, MAE
- Regularization: Dropout rate, L2 weight penalty
- Early Stopping toggle
- Learning Rate Scheduling: Step Decay, Exponential Decay

**Usage**:
1. Set training hyperparameters in the configuration panel
2. Enable optional early stopping or learning rate scheduling
3. Train the model and monitor progress

### 3. Model Management

**Description**: Save and reload model architectures for reuse and evaluation.

**Options**:
- Save model as .h5
- Load existing .h5 model
- Reset architecture configuration

**Usage**:
1. Train a model
2. Save it for later use
3. Load saved models for additional evaluation or further training

### 4. Pre-trained Model Integration

**Description**: Import pre-trained models and fine-tune them on custom data.

**Supported Architectures**:
- VGG16
- ResNet50
- MobileNet

**Options**:
- Freeze base layers
- Add custom classifier head

**Usage**:
1. Select a pre-trained base model from the dropdown
2. Choose to freeze or fine-tune base layers
3. Train with custom dataset input

### 5. Data Augmentation

**Description**: Image augmentation tools to improve generalization.

**Techniques**:
- Random Rotation
- Horizontal Flipping
- Zooming

**Usage**:
1. Enable desired augmentation options before training
2. Use with image datasets like MNIST

### 6. Visualization and Metrics

**Description**: Real-time feedback on training progress and model quality.

**Visuals**:
- Loss and Accuracy plots (Train vs. Validation)
- Weight Gradient Histograms (optional)
- Final test metrics: Accuracy, F1-Score, Parameter Count

**Usage**:
1. Train the model
2. Observe live training curves
3. Analyze final performance summary

## How to Use the Deep Learning Tab

1. Load a dataset (e.g., MNIST) from the Data Management section
2. Design your model by sequentially adding layers (Conv2D, MaxPooling2D, Dropout, Dense, etc.)
3. Configure training settings (epochs, optimizer, loss, etc.)
4. Train the model and monitor real-time metrics
5. Save the model or compare against pre-trained alternatives

## Example Workflows

### 1. Custom CNN + Optimizer Comparison
1. Load MNIST dataset
2. Add Conv2D, MaxPooling, Flatten, Dense, Dropout layers
3. Train with Adam (learning rate=0.0001) and observe performance
4. Switch optimizer to SGD (learning rate=0.001), retrain, and compare
5. Try RMSprop and compare test metrics across runs

### 2. Pre-trained VGG16 Fine-Tuning
1. Select VGG16 from pre-trained models
2. Freeze base layers and add custom head
3. Resize MNIST input to (224, 224, 3)
4. Train for 8 epochs using Adam optimizer
5. Observe performance and compare with custom CNN

## Setup Instructions

### Requirements
- Python >= 3.9
- NumPy
- Pandas
- Matplotlib
- Plotly
- PyQt6
- scikit-learn
- umap-learn
- TensorFlow

### Installation
```bash
# Clone the repository
git clone https://github.com/alp361/MKT3434_2025.git
cd MKT3434_2025

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install numpy pandas matplotlib plotly scikit-learn umap-learn tensorflow pyqt6

# Run the GUI
python 21067611.py