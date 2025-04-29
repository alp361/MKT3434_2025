# Dimensionality Reduction and Validation in MLCourseGUI

This document outlines the features and improvements added to the Dimensionality Reduction tab of the MLCourseGUI application. These additions are designed to support advanced data exploration and model evaluation techniques often required in robotics and mechatronics workflows.

## Overview

The Dimensionality Reduction tab provides a comprehensive toolkit for data visualization, feature extraction, and model validation which are essential capabilities for robotics, mechatronics, and machine learning applications. This module addresses three critical challenges:

1. **Data Compression**: Reduce high-dimensional sensor data to manageable features while preserving meaningful patterns
2. **Visual Exploration**: Intuitively understand complex datasets through 2D or 3D projections
3. **Model Robustness**: Validate algorithms using rigorous statistical methods before deployment

## Features by Module

### 1. Principal Component Analysis (PCA)

**Description:** Linear dimensionality reduction that transforms data to a new coordinate system with maximum variance preservation while decorrelating features.

**Parameters:**
- `n_components`: Number of principal components to retain (int for exact number or float for variance threshold)
- `whiten`: Option to apply whitening to decorrelate features
- `svd_solver`: Chooses decomposition algorithm (auto, full, arpack, or randomized)

**Additional Buttons:**
- Show Explained Variance
- Show Eigenvalue Computation

**Usage:**
1. Set the number of components and solver
2. Optionally apply whitening
3. Train PCA and analyze results

### 2. K-Means Clustering

**Description:** Unsupervised algorithm that partitions data into k clusters by minimizing within-cluster variance using iterative centroid updates.

**Parameters:**
- `n_clusters`: Number of clusters (optimal k can be determined automatically via elbow method)
- `init`: Initialization strategy selection between k-means++, and random.
- `max_iter`: Max number of iterations (typically 300-500 for convergence)

**Additional Buttons:**
- Show Elbow Method
- Run Silhouette Analysis

**Usage:**
1. Set number of clusters, init strategy, and iteration count
2. Train and evaluate with elbow plot and silhouette metrics

### 3. Linear Discriminant Analysis (LDA)

**Description:** Supervised method that projects data to lower dimensions while maximizing class separability through linear combinations of features.

**Parameters:**
- `n_components`: Maximum number of components limited by (number of classes - 1)
- `solver`: Solver method for matrix decomposition (SVD, LSQR, or eigen)

**Additional Button:**
- Show Class Separation Metrics

**Usage:**
1. Set output dimensionality and solver
2. Train LDA and evaluate separation


### 4. t-Distributed Stochastic Neighbor Embedding (t-SNE)

**Description:** Nonlinear technique that preserves local neighborhood relationships in low-dimensional space, ideal for visualizing high-D data.

**Parameters:**
- `n_components`: Projection space (2D or 3D)
- `perplexity`: Balance between global/local aspects (5-50 recommended, scales with dataset size)
- `learning_rate`: Optimization step size (10-1000 range, lower for stability)

**Usage:**
1. Configure projection space, perplexity, and step size.
2. Train t-SNE and visualize the embedding

### 5. Uniform Manifold Approximation and Projection (UMAP)

**Description:** Fast alternative to t-SNE that better preserves global structure while being computationally efficient.

**Parameters:**
- `n_components`: Output dimensions
- `n_neighbors`: Contextual neighborhood size (15-100 typical values)
- `min_dist`: Controls cluster tightness (0.0-1.0, lower for denser packing)

**Usage:**
1. Set UMAP parameters for the selected dataset
2. Run embedding and analyze visually

### 6. Cross-Validation Methods

**Description:** Robust evaluation framework that assesses model generalizability through systematic data splitting and k-fold.

**Parameters:**
- Train/Val/Test Split: Ratio selection for holdout validation (80/10/10, 60/20/20, and 70/15/15)
- K-Fold Value: Number of folds (5-10 typical)
- Model for CV: Choose from trained models such as current model, random forest, etc.
- Evaluation Metric: Accuracy, MSE, RMSE, etc.

**Usage:**
1. Select splitting strategy, number of folds, evaluation metric, and model
2. Evaluate across folds and analyze score mean/std

### 7. Interactive Visualization

**Description:** Dynamic plotting system supporting both static (Matplotlib) and interactive (Plotly) outputs for exploratory analysis.

**Parameter:**
- Visualization Engine: Toggle between Matplotlib and Plotly

**Usage:**
1. Select your preferred engine before projection
2. Supports pan, zoom, hover for cluster and separation insights

## How to Use the Dimensionality Reduction Tab

1. Load or select a dataset from the Data Management section
2. Choose a dimensionality reduction method (PCA, LDA, t-SNE, UMAP)
3. Configure relevant parameters (e.g., number of components, solver, perplexity)
4. Click the appropriate "Train" button
5. Optionally run clustering or visualization enhancements
6. Use cross-validation to evaluate models

## Example Workflows

### 1. PCA + LDA + Cross-Validation

1. Load Breast Cancer dataset
2. Navigate to "Dimensionality Reduction" tab
3. Select "PCA", set `n_components=3`, enable whiten, choose `svd_solver=auto`
4. Train PCA, then visualize explained variance and eigenvalues
5. Switch to "LDA", set `n_components=1`, solver as svd
6. Train LDA and inspect class separation metrics
7. Navigate to Cross-Validation panel, choose 80/10/10 split, k=5, model as "Logistic Regression", metric as Accuracy
8. Run Cross-Validation and analyze mean and std deviation of results

### 2. UMAP + K-Means + Silhouette Evaluation

1. Load Iris dataset
2. Choose "UMAP" from the Dimensionality Reduction methods
3. Set `n_components=2`, `n_neighbors=15`, `min_dist=0.1`
4. Train UMAP and visualize the projection
5. Switch to K-Means Clustering, set `k=3`, `init=k-means++`, `max_iter=300`
6. Train clustering, then run Elbow Method and Silhouette Analysis
7. Evaluate clustering quality and visual separation

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