# Classical Machine Learning Enhancements in MLCourseGUI

This document outlines the new features and improvements added to the "Classical ML" tab of the MLCourseGUI application. These enhancements expand the functionality of classical machine learning algorithms, improve user interaction, and provide more flexibility in evaluation and model training.

## Overview

The "Classical ML" tab provides a user-friendly interface for training and evaluating traditional machine learning models, including regression and classification algorithms. The updated version introduces new algorithm, loss function options, and advanced parameter configurations.

## New Features in the Classical ML Tab

### 1. Support Vector Regression (SVR)

**Description:** A new regression algorithm, Support Vector Regression (SVR), has been added to the "Regression" section.

**Parameters:**
- `kernel`: Choice of kernel function (linear, rbf, poly) to define the feature space transformation.
- `C`: Regularization parameter controlling the trade-off between margin maximization and error.
- `epsilon`: Margin of tolerance for the epsilon-tube within which no penalty is associated.
- `loss_function`: Selectable loss metric (Mean Squared Error (MSE), Mean Absolute Error (MAE), Huber Loss) for evaluation.

**Usage:**
1. Select "Support Vector Regression" from the "Regression" group.
2. Adjust the kernel type, C, and epsilon values using the provided controls.
3. Choose a loss function from the dropdown to evaluate the model's performance.
4. Click "Train Support Vector Regression" to fit the model and visualize results.

### 2. Loss Function Selection

**Description:** Loss function options have been added to classification and regression algorithms to allow users to evaluate model performance using different metrics.

**Regression Loss Options:**
- Mean Squared Error (MSE): Measures the average squared difference between predicted and actual values.
- Mean Absolute Error (MAE): Measures the average absolute difference between predicted and actual values.
- Huber Loss: A robust loss function combining MSE and MAE, less sensitive to outliers (not fully implemented in metrics display yet).

**Classification Loss Options:**
- Cross-Entropy: Measures the performance of classification models with probabilistic outputs.
- Hinge Loss: Used for "maximum-margin" classification, typically with SVMs.

**Affected Algorithms:**
- **Linear Regression:** Added `loss_function` parameter.
- **Logistic Regression:** Added `loss_function` parameter.
- **Support Vector Machine (SVM):** Added `loss_function` parameter.
- **Support Vector Regression (SVR):** Added `loss_function` parameter.

**Usage:**
1. For each supported algorithm, select a loss function from the dropdown menu.
2. Train the model to see the impact of the chosen loss function on the training process and metrics display.

### 3. Enhanced Naive Bayes with Custom Priors

**Description:** The Naive Bayes classifier now supports configurable prior probabilities, enabling users to choose between uniform or custom priors.

**Parameters:**
- `var_smoothing`: Smoothing parameter to handle numerical stability.
- `prior_type`: Choice between Uniform (default equal priors) or User-defined (custom priors).
- `custom_priors`: Text input field for entering comma-separated prior probabilities (e.g., 0.3, 0.3, 0.4), enabled only when User-defined is selected.

**Usage:**
1. Select "Naive Bayes" from the "Classification" group.
2. Adjust `var_smoothing` as needed.
3. Choose `Uniform` for equal priors or `User-defined` to specify custom priors.
4. If `User-defined` is selected, enter prior probabilities in the `custom_priors` text field (e.g., 0.2, 0.5, 0.3 for three classes, ensuring they sum to 1).
5. Click "Train Naive Bayes" to fit the model. The interface validates the priors and displays an error if they are invalid (e.g., mismatch the number of classes or do not sum to 1).

### 4. Improved Parameter Handling in `train_model`

**Description:** The `train_model` method has been enhanced to support the new parameters and loss functions, providing better error handling and flexibility.

**New Functionality:**
- Handles `loss_function` selection for regression models (currently implemented for Linear Regression and SVR with MSE and MAE).
- Processes `prior_type` and `custom_priors` for Naive Bayes, including validation of user-defined priors.
- Supports SVR training with `kernel` and `epsilon` parameters.

**Usage:**
- The method automatically collects parameters from the GUI widgets, applies the selected loss function or priors, and trains the model.
- Results are visualized, and metrics are updated based on the problem type.

## How to Use the Classical ML Tab

### Load Data:
1. Apply scaling and/or preprocessing if needed.
2. Use the "Data Management" section to load a dataset (e.g., Iris, Breast Cancer, or a custom dataset).

### Select an Algorithm:
1. Navigate to the "Classical ML" tab.
2. Choose an algorithm from the "Regression" or "Classification" group.

### Configure Parameters:
1. Adjust algorithm-specific parameters (e.g., C, kernel, max_depth) using the provided controls.
2. For supported algorithms, select a loss function from the dropdown.
3. For Naive Bayes, configure priors if desired.

### Train the Model:
1. Click the "Train [Algorithm Name]" button to fit the model to the loaded data.
2. Monitor the status bar for training completion or error messages.

### View Results:
- The visualization panel displays a scatter plot (regression or classification with PCA if needed).
- The metrics panel shows performance metrics (e.g., MSE, RMSE, R² for regression; accuracy and confusion matrix for classification).

## Example Workflow

### Training SVR with Custom Loss
1. Set `standard scaling` from `Scaling` drop-down menu.
2. Load the "Boston Housing Dataset" (now California Housing Dataset in the updated version).
3. Select "Support Vector Regression" from the "Regression" group.
4. Set `kernel` to `rbf`, `C` to `1.0`, `epsilon` to `0.1`, and `loss_function` to Mean Absolute Error (MAE).
5. Click "Train Support Vector Regression".
6. Observe the scatter plot of predicted vs. actual values and the MAE-based metrics.

### Training Naive Bayes with Custom Priors
1. Load the "Iris Dataset".
2. Select "Naive Bayes" from the "Classification" group.
3. Set `var_smoothing` to `0.1`, `prior_type` to `User-defined`, and `custom_priors` to `0.2, 0.4, 0.4`.
4. Click "Train Naive Bayes".
5. Check the classification visualization and accuracy metrics.


