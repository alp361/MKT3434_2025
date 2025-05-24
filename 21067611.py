import sys
import os
import numpy as np
import pandas as pd
import webbrowser
import umap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTabWidget, QPushButton, QLabel, 
                             QComboBox, QFileDialog, QSpinBox, QDoubleSpinBox,
                             QGroupBox, QScrollArea, QTextEdit, QStatusBar,
                             QProgressBar, QCheckBox, QGridLayout, QMessageBox,
                             QDialog, QLineEdit)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn import datasets, preprocessing, model_selection, impute
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import (accuracy_score, mean_squared_error, 
                              mean_absolute_error, confusion_matrix,
                              mean_absolute_percentage_error, silhouette_score,
                              silhouette_samples, f1_score)

class MLCourseGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Machine Learning Course GUI")
        self.setGeometry(100, 100, 1400, 800)
        
        # Initialize main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)
        
        # Initialize data containers
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.current_model = None
        
        # Neural network configuration
        self.layer_config = []  
        
        self.preprocessing_methods = {
            "None": None,
            "Mean Imputation": "mean",
            "Median Imputation": "median",
            "Most Frequent Imputation": "most_frequent",
            "Forward Fill": "ffill",
            "Backward Fill": "bfill"
        }
        
        # Add loss function options
        self.regression_loss_options = {
            "Mean Squared Error (MSE)": "mse",
            "Mean Absolute Error (MAE)": "mae",
            "Huber Loss": "huber"
        }
        
        self.classification_loss_options = {
            "Cross-Entropy": "categorical_crossentropy",
            "Hinge Loss": "hinge"
        }
        
        # Create components
        self.create_data_section()
        self.create_tabs()
        self.create_visualization()
        self.create_status_bar()
        
    def load_dataset(self):
        """Load selected dataset"""
        try:
            dataset_name = self.dataset_combo.currentText()
            
            if dataset_name == "Load Custom Dataset":
                return
            
            # Load selected dataset
            if dataset_name == "Iris Dataset":
                data = datasets.load_iris()
            elif dataset_name == "Breast Cancer Dataset":
                data = datasets.load_breast_cancer()
            elif dataset_name == "Digits Dataset":
                data = datasets.load_digits()
            # Boston Housing Dataset has been removed, so California Housing Dataset will be used
            elif dataset_name == "Boston Housing Dataset":
                from sklearn.datasets import fetch_california_housing
                data = fetch_california_housing()
            elif dataset_name == "MNIST Dataset":
                (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
                self.X_train, self.X_test = X_train, X_test
                self.y_train, self.y_test = y_train, y_test
                self.status_bar.showMessage(f"Loaded {dataset_name}")
                return
            
            # Split data
            test_size = self.split_spin.value()
            self.X_train, self.X_test, self.y_train, self.y_test = \
                model_selection.train_test_split(data.data, data.target, 
                                              test_size=test_size, 
                                              random_state=42)
            
            # Apply scaling
            self.apply_scaling()
            
            self.status_bar.showMessage(f"Loaded {dataset_name}")
            
        except Exception as e:
            self.show_error(f"Error loading dataset: {str(e)}")
            
    def select_target_column(self, columns):
        """Dialog to select target column from dataset"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Target Column")
        layout = QVBoxLayout(dialog)
        
        # Label to explain target column selection
        info_label = QLabel("Select the column you want to predict (target variable):")
        layout.addWidget(info_label)
        
        # Store data as an attribute of the dialog
        dialog.data = self.current_data    
        
        # Combo box for column selection
        combo = QComboBox()
        combo.addItems(columns)
        layout.addWidget(combo)
        
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        layout.addWidget(info_text)
    
        def update_column_info():
            col = combo.currentText()
            # Show column information
            info = f"Column: {col}\n"
            info += f"Unique Values: {len(dialog.data[col].unique())}\n"
            info += f"Data Type: {dialog.data[col].dtype}\n"
            info += f"Sample Values: {list(dialog.data[col].unique()[:5])}"
            info_text.setText(info)
    
        combo.currentIndexChanged.connect(update_column_info)
        update_column_info()  # Initial update
        
        btn = QPushButton("Select")
        btn.clicked.connect(dialog.accept)
        layout.addWidget(btn)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return combo.currentText()
        return None        
    
    def visualize_missing_data_impact(self, missing_info):
        """Visualize the impact of missing data in the dataset"""
        try:
            # Clear previous figure
            self.figure.clear()
            
            # Create bar plot of missing values
            ax = self.figure.add_subplot(111)
            
            # Filter out columns with zero missing values
            missing_columns = missing_info[missing_info > 0]
            
            if len(missing_columns) > 0:
                missing_columns.plot(kind='bar', ax=ax)
                ax.set_title('Missing Values by Column')
                ax.set_xlabel('Columns')
                ax.set_ylabel('Number of Missing Values')
                ax.tick_params(axis='x', rotation=45)
            else:
                ax.text(0.5, 0.5, 'No Missing Values', 
                        horizontalalignment='center', 
                        verticalalignment='center')
                ax.set_title('Missing Data Analysis')
            
            self.figure.tight_layout()
            self.canvas.draw()
        
        except Exception as e:
            self.show_error(f"Error visualizing missing data: {str(e)}")
                
    def load_custom_data(self):
        """Enhanced custom data loading with preprocessing options and categorical encoding"""
        try:
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "Load Dataset",
                "",
                "CSV files (*.csv)"
            )
            
            if file_name:
                # Load data and store it as an instance attribute
                self.current_data = pd.read_csv(file_name)
                data = self.current_data
                # Preprocess categorical vars
                def preprocess_categorical_data(df):
                    # Identifying categorical columns
                    categorical_columns = df.select_dtypes(include=['object']).columns
                
                    # One-hot encoding for categorical vars
                    for col in categorical_columns:
                        
                        one_hot = pd.get_dummies(df[col], prefix=col)
                    
                        # Drop original categorical column and concat one-hot encoded columns
                        df = pd.concat([df.drop(col, axis=1), one_hot], axis=1)
                
                    return df
            
                # Applying categorical preprocessing
                data = preprocess_categorical_data(data)
            
                # Check and report missing values
                missing_info = data.isnull().sum()
                print("Missing Values:\n", missing_info)
            
                # Preprocessing method selection
                preprocessing_method = self.preprocessing_combo.currentText()
                
                if preprocessing_method != "None":
                    if preprocessing_method in ["Mean Imputation", "Median Imputation", "Most Frequent Imputation"]:
                        # Numerical columns imputation
                        imputer = impute.SimpleImputer(
                            strategy=self.preprocessing_methods[preprocessing_method]
                        )
                        # Separate numerical columns
                        numeric_columns = data.select_dtypes(include=[np.number]).columns
                    
                        # Impute numerical columns
                        if len(numeric_columns) > 0:
                            data[numeric_columns] = imputer.fit_transform(data[numeric_columns])
                    elif preprocessing_method in ["Forward Fill", "Backward Fill"]:
                        # Time series style filling
                        fill_method = "ffill" if preprocessing_method == "Forward Fill" else "bfill"
                        data = data.fillna(method=fill_method)
                else:
                    # If preprocessing is none, drop rows with NaN
                    data = data.dropna()
                    
                # Ask user to select target column
                target_col = self.select_target_column(data.columns)
                
                if target_col:
                    X = data.drop(target_col, axis=1)
                    y = data[target_col]
                    
                    # Split data
                    test_size = self.split_spin.value()
                    self.X_train, self.X_test, self.y_train, self.y_test = \
                        model_selection.train_test_split(X, y, 
                                                      test_size=test_size, 
                                                      random_state=42)
                    
                    # Apply scaling
                    self.apply_scaling()
                    
                    # Visualize missing data impact
                    self.visualize_missing_data_impact(missing_info)
                    
                    self.status_bar.showMessage(f"Loaded custom dataset: {file_name}")
        
        except Exception as e:
            self.show_error(f"Error loading custom dataset: {str(e)}")
    
    
    def apply_scaling(self):
        """Apply selected scaling method to the data"""
        scaling_method = self.scaling_combo.currentText()
        
        if scaling_method != "No Scaling":
            try:
                if scaling_method == "Standard Scaling":
                    scaler = preprocessing.StandardScaler()
                elif scaling_method == "Min-Max Scaling":
                    scaler = preprocessing.MinMaxScaler()
                elif scaling_method == "Robust Scaling":
                    scaler = preprocessing.RobustScaler()
                
                self.X_train = scaler.fit_transform(self.X_train)
                self.X_test = scaler.transform(self.X_test)
                
            except Exception as e:
                self.show_error(f"Error applying scaling: {str(e)}")
    
    def create_data_section(self):
        """Create the data loading and preprocessing section"""
        data_group = QGroupBox("Data Management")
        data_layout = QHBoxLayout()
        
        # Dataset selection
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems([
            "Load Custom Dataset",
            "Iris Dataset",
            "Breast Cancer Dataset",
            "Digits Dataset",
            "Boston Housing Dataset",
            "MNIST Dataset"
        ])
        self.dataset_combo.currentIndexChanged.connect(self.load_dataset)
        
        # Data loading button
        self.load_btn = QPushButton("Load Data")
        self.load_btn.clicked.connect(self.load_custom_data)
        
        # Preprocessing options
        self.scaling_combo = QComboBox()
        self.scaling_combo.addItems([
            "No Scaling",
            "Standard Scaling",
            "Min-Max Scaling",
            "Robust Scaling"
        ])
        
        # Train-test split options
        self.split_spin = QDoubleSpinBox()
        self.split_spin.setRange(0.1, 0.9)
        self.split_spin.setValue(0.2)
        self.split_spin.setSingleStep(0.1)
        
        # Add widgets to layout
        data_layout.addWidget(QLabel("Dataset:"))
        data_layout.addWidget(self.dataset_combo)
        data_layout.addWidget(self.load_btn)
        data_layout.addWidget(QLabel("Scaling:"))
        data_layout.addWidget(self.scaling_combo)
        data_layout.addWidget(QLabel("Test Split:"))
        data_layout.addWidget(self.split_spin)
        
        data_group.setLayout(data_layout)
        self.layout.addWidget(data_group)
        
        # Add preprocessing method dropdown
        self.preprocessing_combo = QComboBox()
        self.preprocessing_combo.addItems(list(self.preprocessing_methods.keys()))
        
        # Modify layout to include preprocessing
        data_layout.addWidget(QLabel("Preprocessing:"))
        data_layout.addWidget(self.preprocessing_combo)
        
    def create_tabs(self):
        """Create tabs for different ML topics"""
        self.tab_widget = QTabWidget()
        
        # Create individual tabs
        tabs = [
            ("Classical ML", self.create_classical_ml_tab),
            ("Deep Learning", self.create_deep_learning_tab),
            ("Dimensionality Reduction", self.create_dim_reduction_tab),
            ("Reinforcement Learning", self.create_rl_tab)
        ]
        
        for tab_name, create_func in tabs:
            scroll = QScrollArea()
            tab_widget = create_func()
            scroll.setWidget(tab_widget)
            scroll.setWidgetResizable(True)
            self.tab_widget.addTab(scroll, tab_name)
        
        self.layout.addWidget(self.tab_widget)
    
    def create_classical_ml_tab(self):
        """Create the classical machine learning algorithms tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # Regression section
        regression_group = QGroupBox("Regression")
        regression_layout = QVBoxLayout()
        
        # Linear Regression with loss func
        lr_group = self.create_algorithm_group(
            "Linear Regression",
            {"fit_intercept": "checkbox",
             "loss_function": list(self.regression_loss_options.keys())}
        )
        regression_layout.addWidget(lr_group)
        
        # Logistic Regression with loss func
        logistic_group = self.create_algorithm_group(
            "Logistic Regression",
            {"C": "double",
             "max_iter": "int",
             "multi_class": ["ovr", "multinomial"],
             "loss_function": list(self.classification_loss_options.keys())}
        )
        regression_layout.addWidget(logistic_group)
        
        # Support Vector Regression (SVR) with loss func
        svr_group = self.create_algorithm_group(
            "Support Vector Regression",
            {"kernel": ["linear", "rbf", "poly"],
             "C": "double",
             "epsilon": "double",
             "loss_function": list(self.regression_loss_options.keys())}
        )
        regression_layout.addWidget(svr_group)
        
        regression_group.setLayout(regression_layout)
        layout.addWidget(regression_group, 0, 0)
        
        # Classification section with loss func selection
        classification_group = QGroupBox("Classification")
        classification_layout = QVBoxLayout()
        
        # Updated Naive Bayes with custom_priors
        nb_group = self.create_algorithm_group(
            "Naive Bayes",
            {"var_smoothing": "double",
             "prior_type": ["Uniform", "User-defined"],
             "custom_priors": "text"}
        )
        classification_layout.addWidget(nb_group)
        
        # SVM with kernel and loss selection
        svm_group = self.create_algorithm_group(
            "Support Vector Machine",
            {"C": "double",
             "kernel": ["linear", "rbf", "poly"],
             "degree": "int",
             "loss_function": list(self.classification_loss_options.keys())}
        )
        classification_layout.addWidget(svm_group)
        
        # Decision Trees
        dt_group = self.create_algorithm_group(
            "Decision Tree",
            {"max_depth": "int",
             "min_samples_split": "int",
             "criterion": ["gini", "entropy"]}
        )
        classification_layout.addWidget(dt_group)
        
        # Random Forest
        rf_group = self.create_algorithm_group(
            "Random Forest",
            {"n_estimators": "int",
             "max_depth": "int",
             "min_samples_split": "int"}
        )
        classification_layout.addWidget(rf_group)
        
        # KNN
        knn_group = self.create_algorithm_group(
            "K-Nearest Neighbors",
            {"n_neighbors": "int",
             "weights": ["uniform", "distance"],
             "metric": ["euclidean", "manhattan"]}
        )
        classification_layout.addWidget(knn_group)
        
        classification_group.setLayout(classification_layout)
        layout.addWidget(classification_group, 0, 1)
        
        return widget
    
    def train_model(self, model_name, param_widgets):
        """Enhanced model training method supporting both classical ML and dimensionality reduction"""
        try:
            # Check if this is a dimensionality reduction algorithm
            dim_reduction_models = ["PCA", "LDA", "K-Means", "t-SNE", "UMAP"]
            
            if model_name in dim_reduction_models:
                # Apply dimensionality reduction based on the selected algorithm
                if model_name == "PCA":
                    self.train_pca(param_widgets)
                elif model_name == "LDA":
                    self.train_lda(param_widgets)
                elif model_name == "K-Means":
                    self.train_kmeans(param_widgets)
                elif model_name == "t-SNE":
                    self.train_tsne(param_widgets)
                elif model_name == "UMAP":
                    self.train_umap(param_widgets)
                
                return  # Exit early for dimensionality reduction methods
            
            # Training implementation for classical ML models
            model_params = {}
            loss_function = None
            prior_type = None
            custom_priors = None
            
            for param_name, widget in param_widgets.items():
                if param_name == "loss_function":
                    loss_function = widget.currentText()
                    continue
                
                if param_name == "prior_type":
                    prior_type = widget.currentText()
                    continue
                
                if param_name == "custom_priors":
                    custom_priors = widget.text()
                    continue
                
                if isinstance(widget, QSpinBox):
                    model_params[param_name] = widget.value()
                elif isinstance(widget, QDoubleSpinBox):
                    model_params[param_name] = widget.value()
                elif isinstance(widget, QCheckBox):
                    model_params[param_name] = widget.isChecked()
                elif isinstance(widget, QComboBox):
                    model_params[param_name] = widget.currentText()
            
            # Select and train appropriate model
            if model_name == "Linear Regression":
                # Remove 'normalize' parameter, because it's no longer supported
                if 'normalize' in model_params:
                    del model_params['normalize']
                
                self.current_model = LinearRegression(**model_params)
                self.current_model.fit(self.X_train, self.y_train)
                y_pred = self.current_model.predict(self.X_test)
                
                # Calculate loss based on selection
                if loss_function == "Mean Squared Error (MSE)":
                    loss = mean_squared_error(self.y_test, y_pred)
                elif loss_function == "Mean Absolute Error (MAE)":
                    loss = mean_absolute_error(self.y_test, y_pred)
                
            elif model_name == "Logistic Regression":
                self.current_model = LogisticRegression(**model_params)
                self.current_model.fit(self.X_train, self.y_train)
                y_pred = self.current_model.predict(self.X_test)
            
            elif model_name == "Support Vector Machine":
                # Adjust parameters for SVM
                kernel = model_params.pop('kernel', 'rbf')
                self.current_model = SVC(kernel=kernel, **model_params)
                self.current_model.fit(self.X_train, self.y_train)
                y_pred = self.current_model.predict(self.X_test)
            
            elif model_name == "Support Vector Regression":
                kernel = model_params.pop('kernel', 'rbf')
                self.current_model = SVR(kernel=kernel, **model_params)
                self.current_model.fit(self.X_train, self.y_train)
                y_pred = self.current_model.predict(self.X_test)
            
            elif model_name == "Naive Bayes":
                # Setting var_smoothing
                var_smoothing = model_params.get('var_smoothing', 1e-9)
                
                # Handle prior probabilities
                if prior_type == "Uniform":
                    priors = None
                elif prior_type == "User-defined":
                    # Parse custom priors from text input
                    try:
                        # Split by comma and convert to float
                        priors = [float(p.strip()) for p in custom_priors.split(',')]
                        
                        # Validate priors
                        if not np.isclose(sum(priors), 1.0):
                            raise ValueError("Prior probabilities must sum to 1")
                        
                        unique_classes = np.unique(self.y_train)
                        if len(priors) != len(unique_classes):
                            raise ValueError(f"Number of priors must match number of classes ({len(unique_classes)})")
                    
                    except Exception as e:
                        self.show_error(f"Invalid prior probabilities: {str(e)}")
                        return
                
                # Create Naive Bayes model
                self.current_model = GaussianNB(
                    var_smoothing=var_smoothing,
                    priors=priors if prior_type == "User-defined" else None
                )
                
                self.current_model.fit(self.X_train, self.y_train)
                y_pred = self.current_model.predict(self.X_test)
            
            # Update visualization and metrics
            self.update_visualization(y_pred)
            self.update_metrics(y_pred)
            
            self.status_bar.showMessage(f"{model_name} Training Complete")
                
        except Exception as e:
            self.show_error(f"Error training {model_name}: {str(e)}")
            
    def create_dim_reduction_tab(self):
        """Create the dimensionality reduction and validation tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # PCA section
        pca_group = QGroupBox("Principal Component Analysis (PCA)")
        pca_layout = QVBoxLayout()
        
        pca_params = self.create_algorithm_group(
            "PCA",
            {"n_components": "int",
            "whiten": "checkbox",
            "svd_solver": ["auto", "full", "arpack", "randomized"]}
        )
        pca_layout.addWidget(pca_params)
        
        # Explained variance visualization option
        variance_btn = QPushButton("Show Explained Variance")
        variance_btn.clicked.connect(lambda: self.show_explained_variance())
        pca_layout.addWidget(variance_btn)
        
        # Eigenvalue computation button
        eigenvalue_btn = QPushButton("Show Eigenvalue Computation")
        eigenvalue_btn.clicked.connect(lambda: self.show_eigenvalue_computation())
        pca_layout.addWidget(eigenvalue_btn)
        
        pca_group.setLayout(pca_layout)
        layout.addWidget(pca_group, 0, 0)
        
        # LDA section
        lda_group = QGroupBox("Linear Discriminant Analysis (LDA)")
        lda_layout = QVBoxLayout()
        
        lda_params = self.create_algorithm_group(
            "LDA",
            {"n_components": "int",
            "solver": ["svd", "lsqr", "eigen"]}
        )
        lda_layout.addWidget(lda_params)
        
        # Class separation metrics visualization
        separation_btn = QPushButton("Show Class Separation Metrics")
        separation_btn.clicked.connect(lambda: self.show_class_separation())
        lda_layout.addWidget(separation_btn)
        
        lda_group.setLayout(lda_layout)
        layout.addWidget(lda_group, 0, 1)
        
        # K-Means section
        kmeans_group = QGroupBox("K-Means Clustering")
        kmeans_layout = QVBoxLayout()
        
        kmeans_params = self.create_algorithm_group(
            "K-Means",
            {"n_clusters": "int",
            "init": ["k-means++", "random"],
            "max_iter": "int"}
        )
        kmeans_layout.addWidget(kmeans_params)
        
        # Elbow method visualization
        elbow_btn = QPushButton("Show Elbow Method")
        elbow_btn.clicked.connect(lambda: self.show_elbow_method())
        kmeans_layout.addWidget(elbow_btn)
        
        # Silhouette analysis button
        silhouette_btn = QPushButton("Run Silhouette Analysis")
        silhouette_btn.clicked.connect(lambda: self.run_silhouette_analysis())
        kmeans_layout.addWidget(silhouette_btn)
        
        kmeans_group.setLayout(kmeans_layout)
        layout.addWidget(kmeans_group, 1, 0)
        
        # t-SNE section
        tsne_group = QGroupBox("t-SNE")
        tsne_layout = QVBoxLayout()
        
        # Modified n_components limited to 2 or 3 only
        tsne_params = self.create_algorithm_group(
            "t-SNE",
            {"n_components (projection)": ["2", "3"],
            "perplexity": "double",
            "learning_rate": "double"}
        )
        tsne_layout.addWidget(tsne_params)
        tsne_group.setLayout(tsne_layout)
        layout.addWidget(tsne_group, 1, 1)
        
        # UMAP section
        umap_group = QGroupBox("UMAP")
        umap_layout = QVBoxLayout()
        
        umap_params = self.create_algorithm_group(
            "UMAP",
            {"n_components": "int",
            "n_neighbors": "int",
            "min_dist": "double"}
        )
        umap_layout.addWidget(umap_params)
        
        umap_group.setLayout(umap_layout)
        layout.addWidget(umap_group, 2, 0, 1, 2)  # Span 2 columns
        
        # Validation Methods
        validation_group = QGroupBox("Cross-Validation Methods")
        validation_layout = QVBoxLayout()
        
        # Dataset split options
        split_layout = QHBoxLayout()
        split_layout.addWidget(QLabel("Train/Val/Test Split:"))
        self.split_combo = QComboBox()
        self.split_combo.addItems(["70/15/15", "60/20/20", "80/10/10"])
        split_layout.addWidget(self.split_combo)
        self.apply_split_btn = QPushButton("Apply Split")
        self.apply_split_btn.clicked.connect(lambda: self.apply_dataset_split())
        split_layout.addWidget(self.apply_split_btn)
        validation_layout.addLayout(split_layout)
        
        # K-fold options
        kfold_layout = QHBoxLayout()
        kfold_layout.addWidget(QLabel("K-Fold Value:"))
        self.kfold_spin = QSpinBox()
        self.kfold_spin.setRange(2, 20)
        self.kfold_spin.setValue(5)
        kfold_layout.addWidget(self.kfold_spin)
        validation_layout.addLayout(kfold_layout)
        
        # Model selection for CV
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model for CV:"))
        self.cv_model_combo = QComboBox()
        self.cv_model_combo.addItems(["Current Model", "Random Forest", "SVM", "LogisticRegression"])
        model_layout.addWidget(self.cv_model_combo)
        validation_layout.addLayout(model_layout)
        
        # Metrics selection
        metrics_layout = QHBoxLayout()
        metrics_layout.addWidget(QLabel("Evaluation Metric:"))
        self.metrics_combo = QComboBox()
        self.metrics_combo.addItems(["Accuracy", "MSE", "RMSE", "R²", "F1-Score"])
        metrics_layout.addWidget(self.metrics_combo)
        validation_layout.addLayout(metrics_layout)
        
        # Run validation button
        validation_btn = QPushButton("Run Cross-Validation")
        validation_btn.clicked.connect(lambda: self.run_cross_validation())
        validation_layout.addWidget(validation_btn)
        
        validation_group.setLayout(validation_layout)
        layout.addWidget(validation_group, 3, 0, 1, 2)  # Span 2 columns
        
        plotly_group = QGroupBox("Interactive Visualization")
        plotly_layout = QVBoxLayout()
        
        # Toggle between Matplotlib and Plotly
        viz_layout = QHBoxLayout()
        viz_layout.addWidget(QLabel("Visualization Engine:"))
        self.viz_engine_combo = QComboBox()
        self.viz_engine_combo.addItems(["Matplotlib", "Plotly"])
        self.viz_engine_combo.currentTextChanged.connect(self.update_visualization_plotly)
        viz_layout.addWidget(self.viz_engine_combo)
        plotly_layout.addLayout(viz_layout)
        
        plotly_group.setLayout(plotly_layout)
        layout.addWidget(plotly_group, 4, 0, 1, 2)  # Span 2 columns
        
        return widget
    
    def create_rl_tab(self):
        """Create the reinforcement learning tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # Environment selection
        env_group = QGroupBox("Environment")
        env_layout = QVBoxLayout()
        
        self.env_combo = QComboBox()
        self.env_combo.addItems([
            "CartPole-v1",
            "MountainCar-v0",
            "Acrobot-v1"
        ])
        env_layout.addWidget(self.env_combo)
        
        env_group.setLayout(env_layout)
        layout.addWidget(env_group, 0, 0)
        
        # RL Algorithm selection
        algo_group = QGroupBox("RL Algorithm")
        algo_layout = QVBoxLayout()
        
        self.rl_algo_combo = QComboBox()
        self.rl_algo_combo.addItems([
            "Q-Learning",
            "SARSA",
            "DQN"
        ])
        algo_layout.addWidget(self.rl_algo_combo)
        
        algo_group.setLayout(algo_layout)
        layout.addWidget(algo_group, 0, 1)
        
        return widget
    
    def create_visualization(self):
        """Create the visualization section"""
        viz_group = QGroupBox("Visualization")
        viz_layout = QHBoxLayout()
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        viz_layout.addWidget(self.canvas)
        
        # Metrics display
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        viz_layout.addWidget(self.metrics_text)
        
        viz_group.setLayout(viz_layout)
        self.layout.addWidget(viz_group)
    
    def create_status_bar(self):
        """Create the status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Add progress bar
        self.progress_bar = QProgressBar()
        self.status_bar.addPermanentWidget(self.progress_bar)
    
    def create_algorithm_group(self, name, params):
        """Helper method to create algorithm parameter groups"""
        group = QGroupBox(name)
        layout = QVBoxLayout()
        
        # Create parameter inputs
        param_widgets = {}
        prior_type_widget = None
        custom_priors_widget = None
        
        for param_name, param_type in params.items():
            param_layout = QHBoxLayout()
            param_layout.addWidget(QLabel(f"{param_name}:"))
            
            if param_type == "int":
                widget = QSpinBox()
                widget.setRange(1, 1000)
            elif param_type == "double":
                widget = QDoubleSpinBox()
                widget.setRange(0.0001, 1000.0)
                widget.setSingleStep(0.1)
            elif param_type == "checkbox":
                widget = QCheckBox()
            elif isinstance(param_type, list):
                widget = QComboBox()
                widget.addItems(param_type)
                
                if param_name == "prior_type":
                    prior_type_widget = widget
            elif param_type == "text":
                widget = QLineEdit()
                widget.setPlaceholderText("Enter comma-separated probabilities")
                widget.setMaximumWidth(150)
                widget.setEnabled(False)
                custom_priors_widget = widget

            param_layout.addWidget(widget)
            param_widgets[param_name] = widget
            layout.addLayout(param_layout)
        
        # Add train button
        train_btn = QPushButton(f"Train {name}")
        train_btn.clicked.connect(lambda: self.train_model(name, param_widgets))
        layout.addWidget(train_btn)
        
        def toggle_custom_priors():
            """Dynamic enabling/disabling of custom priors section"""
            if prior_type_widget and custom_priors_widget:
                prior_type = prior_type_widget.currentText()
                
                # Enable/disable based on prior type
                custom_priors_widget.setEnabled(prior_type == "User-defined")
                
                # Clear the input if switching to Uniform
                if prior_type == "Uniform":
                    custom_priors_widget.clear()
        
        # Connect the toggle function to prior_type combo box
        if prior_type_widget and custom_priors_widget:
            prior_type_widget.currentIndexChanged.connect(toggle_custom_priors)
        
            # Initial state
            prior_type_widget.currentIndexChanged.emit(prior_type_widget.currentIndex())

        
        group.setLayout(layout)
        return group
    
        
    def show_error(self, message):
        """Show error message dialog"""
        QMessageBox.critical(self, "Error", message)
       
    def create_deep_learning_tab(self):
        """Create the improved deep learning tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # Main controls section
        controls_group = QGroupBox("Neural Network Architecture")
        controls_layout = QVBoxLayout()
        
        # Layer configuration
        layer_section = QHBoxLayout()
        layer_btn = QPushButton("Add Layer")
        layer_btn.clicked.connect(self.add_layer_dialog)
        remove_layer_btn = QPushButton("Remove Last Layer")
        remove_layer_btn.clicked.connect(self.remove_last_layer)
        clear_layers_btn = QPushButton("Clear All Layers")
        clear_layers_btn.clicked.connect(self.clear_all_layers)
        
        layer_section.addWidget(layer_btn)
        layer_section.addWidget(remove_layer_btn)
        layer_section.addWidget(clear_layers_btn)
        controls_layout.addLayout(layer_section)
        
        # Layer display
        self.layer_display = QTextEdit()
        self.layer_display.setReadOnly(True)
        self.layer_display.setMaximumHeight(100)
        controls_layout.addWidget(QLabel("Current Architecture:"))
        controls_layout.addWidget(self.layer_display)
        
        # Training parameters
        training_params = self.create_enhanced_training_params()
        controls_layout.addWidget(training_params)
        
        # Model management
        model_mgmt = self.create_model_management_section()
        controls_layout.addWidget(model_mgmt)
        
        # Train button
        train_btn = QPushButton("Train Neural Network")
        train_btn.clicked.connect(self.train_neural_network)
        controls_layout.addWidget(train_btn)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group, 0, 0)
        
        # Pre-trained models section
        pretrained_group = self.create_pretrained_section()
        layout.addWidget(pretrained_group, 0, 1)
        
        # Advanced options
        advanced_group = self.create_advanced_options()
        layout.addWidget(advanced_group, 1, 0, 1, 2)
        
        return widget
    
    def add_layer_dialog(self):
        """Enhanced layer addition dialog"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Neural Network Layer")
        dialog.setFixedSize(400, 300)
        layout = QVBoxLayout(dialog)
        
        # Layer type
        type_layout = QHBoxLayout()
        type_combo = QComboBox()
        type_combo.addItems(["Dense", "Conv2D", "MaxPooling2D", "LSTM", "GRU", "Dropout", "Flatten", "BatchNormalization"])
        type_layout.addWidget(QLabel("Layer Type:"))
        type_layout.addWidget(type_combo)
        layout.addLayout(type_layout)
        
        # Dynamic parameters
        self.param_widget = QWidget()
        self.param_layout = QVBoxLayout(self.param_widget)
        layout.addWidget(self.param_widget)
        
        def update_params():
            # Clear existing widgets
            while self.param_layout.count():
                child = self.param_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
                elif child.layout():
                    self.clearLayout(child.layout())
            
            layer_type = type_combo.currentText()
            self.current_params = {}
            
            if layer_type == "Dense":
                self.add_param_input("Units", "int", 64, 1, 1000)
                self.add_param_combo("Activation", ["relu", "sigmoid", "tanh", "softmax", "linear"])
            elif layer_type == "Conv2D":
                self.add_param_input("Filters", "int", 32, 1, 512)
                self.add_param_input("Kernel Size", "int", 3, 1, 10)
                self.add_param_combo("Activation", ["relu", "sigmoid", "tanh"])
                self.add_param_combo("Padding", ["valid", "same"])
            elif layer_type == "MaxPooling2D":
                # MaxPooling2D doesn't need parameters
                pass
            elif layer_type == "LSTM":
                self.add_param_input("Units", "int", 64, 1, 500)
                self.add_param_check("Return Sequences")
            elif layer_type == "GRU":
                self.add_param_input("Units", "int", 50, 1, 500)
                self.add_param_check("Return Sequences")
            elif layer_type == "Dropout":
                self.add_param_input("Rate", "float", 0.5, 0.0, 1.0)
            elif layer_type == "Flatten":
                # Flatten doesn't need parameters
                pass
            elif layer_type == "BatchNormalization":
                # BatchNormalization can work with default parameters
                pass
        
        type_combo.currentIndexChanged.connect(update_params)
        update_params()
        
        # Buttons
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(lambda: self.add_layer_to_config(type_combo.currentText()) or dialog.accept())
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        dialog.exec()
    
    def add_param_input(self, name, param_type, default, min_val, max_val):
        """Add parameter input widget"""
        layout = QHBoxLayout()
        layout.addWidget(QLabel(f"{name}:"))
        
        if param_type == "int":
            widget = QSpinBox()
            widget.setRange(min_val, max_val)
            widget.setValue(default)
        else:  # float
            widget = QDoubleSpinBox()
            widget.setRange(min_val, max_val)
            widget.setValue(default)
            widget.setDecimals(3)
        
        layout.addWidget(widget)
        self.param_layout.addLayout(layout)
        self.current_params[name.lower().replace(" ", "_")] = widget

    def add_param_combo(self, name, options):
        """Add parameter combo widget"""
        layout = QHBoxLayout()
        layout.addWidget(QLabel(f"{name}:"))
        widget = QComboBox()
        widget.addItems(options)
        layout.addWidget(widget)
        self.param_layout.addLayout(layout)
        self.current_params[name.lower().replace(" ", "_")] = widget

    def add_param_check(self, name):
        """Add parameter checkbox widget"""
        widget = QCheckBox(name)
        self.param_layout.addWidget(widget)
        self.current_params[name.lower().replace(" ", "_")] = widget

    def add_layer_to_config(self, layer_type):
        """Add layer to configuration"""
        params = {}
        for param_name, widget in self.current_params.items():
            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                params[param_name] = widget.value()
            elif isinstance(widget, QComboBox):
                params[param_name] = widget.currentText()
            elif isinstance(widget, QCheckBox):
                params[param_name] = widget.isChecked()
        
        self.layer_config.append({"type": layer_type, "params": params})
        self.update_layer_display()

    def remove_last_layer(self):
        """Remove the last added layer"""
        if self.layer_config:
            self.layer_config.pop()
            self.update_layer_display()

    def clear_all_layers(self):
        """Clear all layers"""
        self.layer_config = []
        self.update_layer_display()

    def clearLayout(self, layout):
        """Clear a layout"""
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
            elif child.layout():
                self.clearLayout(child.layout())

    def update_layer_display(self):
        """Update the layer display text"""
        display_text = ""
        for i, layer in enumerate(self.layer_config):
            display_text += f"{i+1}. {layer['type']}"
            if layer['params']:
                params_str = ", ".join([f"{k}={v}" for k, v in layer['params'].items()])
                display_text += f" ({params_str})"
            display_text += "\n"
        self.layer_display.setText(display_text)
    
    def create_enhanced_training_params(self):
        """Create enhanced training parameters section"""
        group = QGroupBox("Training Configuration")
        layout = QGridLayout()
        
        # Basic parameters
        layout.addWidget(QLabel("Epochs:"), 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(10)
        layout.addWidget(self.epochs_spin, 0, 1)
        
        layout.addWidget(QLabel("Batch Size:"), 0, 2)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 512)
        self.batch_size_spin.setValue(32)
        layout.addWidget(self.batch_size_spin, 0, 3)
        
        # Optimizer selection
        layout.addWidget(QLabel("Optimizer:"), 1, 0)
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["Adam", "SGD", "RMSprop"])
        layout.addWidget(self.optimizer_combo, 1, 1)
        
        layout.addWidget(QLabel("Learning Rate:"), 1, 2)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 1.0)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setDecimals(4)
        layout.addWidget(self.lr_spin, 1, 3)
        
        # Loss function selection
        layout.addWidget(QLabel("Loss Function:"), 2, 0)
        self.loss_combo = QComboBox()
        self.loss_combo.addItems(["categorical_crossentropy", "sparse_categorical_crossentropy", "mse", "mae"])
        layout.addWidget(self.loss_combo, 2, 1)
        
        # Regularization
        layout.addWidget(QLabel("L2 Regularization:"), 2, 2)
        self.l2_spin = QDoubleSpinBox()
        self.l2_spin.setRange(0.0, 1.0)
        self.l2_spin.setValue(0.0)
        self.l2_spin.setDecimals(4)
        layout.addWidget(self.l2_spin, 2, 3)
        
        # Early stopping
        self.early_stopping_check = QCheckBox("Early Stopping")
        self.early_stopping_check.setChecked(True)
        layout.addWidget(self.early_stopping_check, 3, 0)
        
        # Learning rate scheduling
        self.lr_schedule_check = QCheckBox("LR Decay")
        layout.addWidget(self.lr_schedule_check, 3, 1)
        
        group.setLayout(layout)
        return group
    
    def create_model_management_section(self):
        """Create model save and load section"""
        group = QGroupBox("Model Management")
        layout = QHBoxLayout()
        
        save_btn = QPushButton("Save Model")
        save_btn.clicked.connect(self.save_model)
        load_btn = QPushButton("Load Model")
        load_btn.clicked.connect(self.load_model)
        
        layout.addWidget(save_btn)
        layout.addWidget(load_btn)
        
        group.setLayout(layout)
        return group

    def create_pretrained_section(self):
        """Create pre-trained models section"""
        group = QGroupBox("Pre-trained Models")
        layout = QVBoxLayout()
        
        self.pretrained_combo = QComboBox()
        self.pretrained_combo.addItems(["None", "VGG16", "ResNet50", "MobileNet"])
        layout.addWidget(QLabel("Base Model:"))
        layout.addWidget(self.pretrained_combo)
        
        self.freeze_check = QCheckBox("Freeze Base Layers")
        self.freeze_check.setChecked(True)
        layout.addWidget(self.freeze_check)
        
        finetune_btn = QPushButton("Load & Fine-tune")
        finetune_btn.clicked.connect(self.load_pretrained_model)
        layout.addWidget(finetune_btn)
        
        group.setLayout(layout)
        return group

    def create_advanced_options(self):
        """Create advanced training options"""
        group = QGroupBox("Advanced Options")
        layout = QHBoxLayout()
        
        # Data augmentation options
        aug_group = QGroupBox("Data Augmentation")
        aug_layout = QVBoxLayout()
        self.rotation_check = QCheckBox("Rotation")
        self.flip_check = QCheckBox("Horizontal Flip")
        self.zoom_check = QCheckBox("Zoom")
        aug_layout.addWidget(self.rotation_check)
        aug_layout.addWidget(self.flip_check)
        aug_layout.addWidget(self.zoom_check)
        aug_group.setLayout(aug_layout)
        
        # Visualization options
        viz_group = QGroupBox("Visualization")
        viz_layout = QVBoxLayout()
        self.plot_gradients_check = QCheckBox("Plot Gradients")
        self.plot_weights_check = QCheckBox("Plot Weight Histograms")
        viz_layout.addWidget(self.plot_gradients_check)
        viz_layout.addWidget(self.plot_weights_check)
        viz_group.setLayout(viz_layout)
        
        layout.addWidget(aug_group)
        layout.addWidget(viz_group)
        group.setLayout(layout)
        return group
    
    def train_neural_network(self):
        """Enhanced neural network training with error handling"""
        if not self.layer_config:
            self.show_error("Please add at least one layer to the network")
            return
        
        if self.X_train is None:
            self.show_error("Please load a dataset first")
            return
        
        try:
            # Create model with proper input validation
            model = self.create_neural_network()
            
            # Prepare data
            X_train, y_train = self.prepare_training_data()
            X_test, y_test = self.prepare_test_data()
            
            # Determine appropriate loss function based on data
            loss = self.loss_combo.currentText()
            if len(np.unique(self.y_train)) == 2:  # Binary classification
                if len(y_train.shape) == 1 or y_train.shape[1] == 1:
                    loss = 'binary_crossentropy'
                else:
                    loss = 'categorical_crossentropy'
            elif len(np.unique(self.y_train)) > 2:  # Multi-class
                if len(y_train.shape) == 1:
                    loss = 'sparse_categorical_crossentropy'
                else:
                    loss = 'categorical_crossentropy'
            
            # Compile model
            optimizer = self.get_optimizer()
            metrics = ['accuracy'] if 'crossentropy' in loss else ['mae']
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
            
            # Get callbacks
            callbacks = self.get_callbacks()
            
            # Display model summary
            print("Model Architecture:")
            model.summary()
            
            # Train model
            history = model.fit(
                X_train, y_train,
                batch_size=self.batch_size_spin.value(),
                epochs=self.epochs_spin.value(),
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                verbose=1
            )
            
            self.current_model = model
            self.plot_training_history(history)
            self.update_metrics_display(model, X_test, y_test)
            
            if self.plot_weights_check.isChecked():
                self.plot_weight_histograms(model)
            
            self.status_bar.showMessage("Neural Network Training Complete")
            
        except Exception as e:
            error_msg = f"Error training neural network: {str(e)}"
            print(error_msg)  # Print to console for debugging
            self.show_error(error_msg)
    
    def create_neural_network(self):
        """Create neural network with enhanced architecture"""
        model = models.Sequential()
        
        # Determine input shape based on data type
        input_shape = None
        data_type = "tabular"
        
        if hasattr(self, 'X_train') and self.X_train is not None:
            if len(self.X_train.shape) == 4:  # Image data
                input_shape = self.X_train.shape[1:]
                data_type = "image"
            elif len(self.X_train.shape) == 3:  # Sequence data or grayscale images
                # Check if it's actually image data like MNIST
                if self.X_train.shape[1] == self.X_train.shape[2]:  # Square dimensions suggest images
                    # For LSTM usage, keep 3D shape
                    # Check if user wants to use LSTM/GRU layers
                    has_rnn_layers = any(layer['type'] in ['LSTM', 'GRU'] for layer in self.layer_config)
                    
                    if has_rnn_layers:
                        # For RNN: treat each row of the image as a time step
                        # Input shape: (batch, timesteps=28, features=28) for MNIST
                        input_shape = (self.X_train.shape[1], self.X_train.shape[2])
                        data_type = "sequence"
                    else:
                        # For CNN: add channel dimension
                        self.X_train = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], self.X_train.shape[2], 1)
                        self.X_test = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1], self.X_test.shape[2], 1)
                        input_shape = self.X_train.shape[1:]
                        data_type = "image"
                else:
                    # True sequence data
                    input_shape = self.X_train.shape[1:]
                    data_type = "sequence"
            else:  # Tabular data
                input_shape = (self.X_train.shape[1],)
                data_type = "tabular"
        else:
            # Default shape if no data loaded
            input_shape = (10,)  # Default for tabular data
            data_type = "tabular"
        
        first_layer = True
        has_conv_layers = any(layer['type'] == 'Conv2D' for layer in self.layer_config)
        has_rnn_layers = any(layer['type'] in ['LSTM', 'GRU'] for layer in self.layer_config)
        
        # Validate architecture based on data type
        if has_conv_layers and data_type not in ["image"]:
            # Provide specific error messages based on data type
            if data_type == "tabular":
                raise ValueError("Conv2D layers require image data (4D input). Current data is tabular (2D). Please use Dense layers for tabular data or load image data (like MNIST).")
            elif data_type == "sequence":
                raise ValueError("Conv2D layers require image data (4D input). Current data appears to be sequence data (3D). For sequence data, use LSTM/GRU layers.")
        
        for i, layer_config in enumerate(self.layer_config):
            layer_type = layer_config["type"]
            params = layer_config["params"].copy()
            
            # Add input shape to first layer
            if first_layer and layer_type in ["Dense", "Conv2D", "LSTM", "GRU"]:
                if layer_type == "Dense":
                    if data_type == "image" and not any(prev_layer['type'] == 'Flatten' for prev_layer in self.layer_config[:i]):
                        # If we have image data but no Flatten layer before Dense, we need to flatten automatically
                        pass
                    else:
                        params['input_shape'] = input_shape
                elif layer_type == "Conv2D":
                    if data_type != "image":
                        raise ValueError(f"Conv2D layer requires 4D image data, but data type is {data_type}")
                    params['input_shape'] = input_shape
                elif layer_type in ["LSTM", "GRU"]:
                    if data_type == "tabular":
                        # For tabular data, reshape for sequence processing
                        params['input_shape'] = (1, input_shape[0])  # (timesteps, features)
                    elif data_type == "sequence":
                        params['input_shape'] = input_shape
                    elif data_type == "image":
                        # For image data, use the sequence input shape
                        params['input_shape'] = input_shape
                    else:
                        raise ValueError(f"{layer_type} layer configuration error for data type {data_type}")
                first_layer = False
            
            # Handle kernel size for Conv2D
            if layer_type == "Conv2D" and "kernel_size" in params:
                kernel_size = params['kernel_size']
                if isinstance(kernel_size, int):
                    params['kernel_size'] = (kernel_size, kernel_size)
            
            # Add regularization if specified
            l2_reg = self.l2_spin.value()
            if l2_reg > 0 and layer_type in ["Dense", "Conv2D"]:
                params['kernel_regularizer'] = tf.keras.regularizers.l2(l2_reg)
            
            # Create layer
            try:
                if layer_type == "Dense":
                    # Check if we need to flatten before Dense layer for image data
                    if (data_type == "image" and i == 0 and 
                        not any(prev_layer['type'] in ['Flatten', 'GlobalAveragePooling2D'] for prev_layer in self.layer_config[:i])):
                        model.add(layers.Flatten(input_shape=input_shape))
                        model.add(layers.Dense(**{k:v for k,v in params.items() if k != 'input_shape'}))
                    else:
                        model.add(layers.Dense(**params))
                        
                elif layer_type == "Conv2D":
                    model.add(layers.Conv2D(**params))
                    
                elif layer_type == "MaxPooling2D":
                    pool_size = params.get('pool_size', 2)
                    if isinstance(pool_size, int):
                        pool_size = (pool_size, pool_size)
                    model.add(layers.MaxPooling2D(pool_size=pool_size))
                    
                elif layer_type == "LSTM":
                    model.add(layers.LSTM(**params))
                    
                elif layer_type == "GRU":
                    model.add(layers.GRU(**params))
                    
                elif layer_type == "Dropout":
                    rate = params.get('rate', 0.5)
                    model.add(layers.Dropout(rate))
                    
                elif layer_type == "Flatten":
                    model.add(layers.Flatten())
                    
                elif layer_type == "BatchNormalization":
                    model.add(layers.BatchNormalization())
                    
            except Exception as e:
                raise ValueError(f"Error creating {layer_type} layer: {str(e)}")
        
        # Add output layer
        if hasattr(self, 'y_train') and self.y_train is not None:
            if len(self.y_train.shape) > 1 and self.y_train.shape[1] > 1:
                # One-hot encoded output
                num_classes = self.y_train.shape[1]
            else:
                # Integer labels
                num_classes = len(np.unique(self.y_train))
        else:
            num_classes = 2  # Default binary classification
        
        # Add a flattened layer before the output if needed
        if (data_type == "image" and len(self.layer_config) > 0 and 
            self.layer_config[-1]['type'] not in ['Dense', 'Flatten', 'GlobalAveragePooling2D'] and
            not has_rnn_layers):
            model.add(layers.GlobalAveragePooling2D())
        
        # Add appropriate output layer
        if num_classes > 2:
            model.add(layers.Dense(num_classes, activation='softmax'))
        else:
            model.add(layers.Dense(1, activation='sigmoid'))
        
        return model

    def get_optimizer(self):
        """Get configured optimizer"""
        lr = self.lr_spin.value()
        optimizer_name = self.optimizer_combo.currentText()
        
        if optimizer_name == "Adam":
            return optimizers.Adam(learning_rate=lr)
        elif optimizer_name == "SGD":
            return optimizers.SGD(learning_rate=lr)
        elif optimizer_name == "RMSprop":
            return optimizers.RMSprop(learning_rate=lr)

    def get_callbacks(self):
        """Get training callbacks"""
        callbacks = []
        
        # Early stopping
        if self.early_stopping_check.isChecked():
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True))
        
        # Learning rate scheduling
        if self.lr_schedule_check.isChecked():
            callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5))
        
        # Custom callback for progress and gradient tracking
        if self.plot_gradients_check.isChecked():
            callbacks.append(self.GradientCallback(self))
        
        return callbacks

    class GradientCallback(tf.keras.callbacks.Callback):
        """Custom callback for gradient monitoring"""
        def __init__(self, parent):
            super().__init__()
            self.parent = parent
            self.gradients = []
        
        def on_epoch_end(self, epoch, logs=None):
            # Update progress
            progress = int(((epoch + 1) / self.params['epochs']) * 100)
            self.parent.progress_bar.setValue(progress)
            
            # Collect gradients if requested
            if epoch % 10 == 0:  # Every 10 epochs
                with tf.GradientTape() as tape:
                    predictions = self.model(self.parent.X_train[:100])  # Sample
                    loss = self.model.compiled_loss(self.parent.y_train[:100], predictions)
                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.gradients.append([tf.norm(g).numpy() for g in gradients if g is not None])
                
    def update_visualization(self, y_pred):
        """Update the visualization with current results"""
        self.figure.clear()
        
        # Create appropriate visualization based on data
        if len(np.unique(self.y_test)) > 10:  # Regression
            ax = self.figure.add_subplot(111)
            ax.scatter(self.y_test, y_pred)
            ax.plot([self.y_test.min(), self.y_test.max()],
                   [self.y_test.min(), self.y_test.max()],
                   'r--', lw=2)
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            
        else:  # Classification
            if self.X_train.shape[1] > 2:  # Use PCA for visualization
                pca = PCA(n_components=2)
                X_test_2d = pca.fit_transform(self.X_test)
                
                ax = self.figure.add_subplot(111)
                scatter = ax.scatter(X_test_2d[:, 0], X_test_2d[:, 1],
                                   c=y_pred, cmap='viridis')
                self.figure.colorbar(scatter)
                
            else:  # Direct 2D visualization
                ax = self.figure.add_subplot(111)
                scatter = ax.scatter(self.X_test[:, 0], self.X_test[:, 1],
                                   c=y_pred, cmap='viridis')
                self.figure.colorbar(scatter)
        
        self.canvas.draw()
        
    def update_metrics(self, y_pred):
        """Update metrics display"""
        metrics_text = "Model Performance Metrics:\n\n"
        
        # Calculate appropriate metrics based on problem type
        if len(np.unique(self.y_test)) > 10:  # Regression
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = self.current_model.score(self.X_test, self.y_test)
            
            metrics_text += f"Mean Squared Error: {mse:.4f}\n"
            metrics_text += f"Root Mean Squared Error: {rmse:.4f}\n"
            metrics_text += f"R² Score: {r2:.4f}"
            
        else:  # Classification
            accuracy = accuracy_score(self.y_test, y_pred)
            conf_matrix = confusion_matrix(self.y_test, y_pred)
            
            metrics_text += f"Accuracy: {accuracy:.4f}\n\n"
            metrics_text += "Confusion Matrix:\n"
            metrics_text += str(conf_matrix)
        
        self.metrics_text.setText(metrics_text)
    
    def prepare_training_data(self):
        """Prepare training data with proper shape handling and augmentation"""
        if self.X_train is None:
            raise ValueError("No training data available. Please load a dataset first.")
        
        X_train = self.X_train.copy()
        y_train = self.y_train.copy()
        
        # Handle different data types
        if len(X_train.shape) == 4:  # Image data
            # Normalize image data
            X_train = X_train.astype('float32') / 255.0
            
            # Apply data augmentation for image data
            if any([self.rotation_check.isChecked(), 
                    self.flip_check.isChecked(), 
                    self.zoom_check.isChecked()]):
                datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                    rotation_range=20 if self.rotation_check.isChecked() else 0,
                    horizontal_flip=self.flip_check.isChecked(),
                    zoom_range=0.2 if self.zoom_check.isChecked() else 0
                )
                return datagen.flow(X_train, y_train, batch_size=self.batch_size_spin.value())
        
        elif len(X_train.shape) == 2:  # Tabular data
            # Apply scaling if selected
            scaling_method = self.scaling_combo.currentText()
            if scaling_method != "No Scaling":
                if scaling_method == "Standard Scaling":
                    scaler = preprocessing.StandardScaler()
                elif scaling_method == "Min-Max Scaling":
                    scaler = preprocessing.MinMaxScaler()
                elif scaling_method == "Robust Scaling":
                    scaler = preprocessing.RobustScaler()
                
                X_train = scaler.fit_transform(X_train)
        
        # Handle target variable encoding
        if len(np.unique(y_train)) > 2 and len(y_train.shape) == 1:
            # Convert to categorical for multi-class
            y_train = tf.keras.utils.to_categorical(y_train)
        
        return X_train, y_train

    def prepare_test_data(self):
        """Prepare test data with proper shape handling"""
        if self.X_test is None:
            raise ValueError("No test data available.")
        
        X_test = self.X_test.copy()
        y_test = self.y_test.copy()
        
        # Handle different data types
        if len(X_test.shape) == 4:  # Image data
            X_test = X_test.astype('float32') / 255.0
        elif len(X_test.shape) == 2:  # Tabular data
            # Apply same scaling as training data
            scaling_method = self.scaling_combo.currentText()
            if scaling_method != "No Scaling":
                if scaling_method == "Standard Scaling":
                    scaler = preprocessing.StandardScaler()
                elif scaling_method == "Min-Max Scaling":
                    scaler = preprocessing.MinMaxScaler()
                elif scaling_method == "Robust Scaling":
                    scaler = preprocessing.RobustScaler()
                
                # Fit on training data, transform test data
                scaler.fit(self.X_train)
                X_test = scaler.transform(X_test)
        
        # Handle target variable encoding
        if len(np.unique(self.y_train)) > 2 and len(y_test.shape) == 1:
            y_test = tf.keras.utils.to_categorical(y_test)
        
        return X_test, y_test

    def save_model(self):
        """Save trained model"""
        if self.current_model is None:
            self.show_error("No trained model to save")
            return
        
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Model", "", "H5 files (*.h5)")
        if file_name:
            self.current_model.save(file_name)
            self.status_bar.showMessage(f"Model saved to {file_name}")

    def load_model(self):
        """Load saved model"""
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Model", "", "H5 files (*.h5)")
        if file_name:
            self.current_model = tf.keras.models.load_model(file_name)
            self.status_bar.showMessage(f"Model loaded from {file_name}")

    def load_pretrained_model(self):
        """Load and configure pre-trained model"""
        model_name = self.pretrained_combo.currentText()
        if model_name == "None":
            return
        
        try:
            input_shape = (224, 224, 3)  # Standard input for pre-trained models
            
            if model_name == "VGG16":
                base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
            elif model_name == "ResNet50":
                base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
            elif model_name == "MobileNet":
                base_model = tf.keras.applications.MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)
            
            if self.freeze_check.isChecked():
                base_model.trainable = False
            
            # Create new model
            model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(len(np.unique(self.y_train)), activation='softmax')
            ])
            
            self.current_model = model
            self.status_bar.showMessage(f"Loaded {model_name} for fine-tuning")
            
        except Exception as e:
            self.show_error(f"Error loading pre-trained model: {str(e)}")

    def plot_weight_histograms(self, model):
        """Plot weight histograms"""
        self.figure.clear()
        weights = []
        for layer in model.layers:
            if hasattr(layer, 'get_weights') and layer.get_weights():
                weights.extend([w.flatten() for w in layer.get_weights()])
        
        if weights:
            ax = self.figure.add_subplot(111)
            ax.hist(np.concatenate(weights), bins=50, alpha=0.7)
            ax.set_title('Weight Distribution')
            ax.set_xlabel('Weight Value')
            ax.set_ylabel('Frequency')
            self.figure.tight_layout()
            self.canvas.draw()

    def update_metrics_display(self, model, X_test, y_test):
        """Update metrics display with comprehensive results"""
        y_pred = model.predict(X_test)
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test
        else:
            y_pred_classes = (y_pred > 0.5).astype(int).flatten()
            y_true = y_test
        
        accuracy = accuracy_score(y_true, y_pred_classes)
        f1 = f1_score(y_true, y_pred_classes, average='weighted')
        
        metrics_text = f"Test Accuracy: {accuracy:.4f}\n"
        metrics_text += f"F1-Score: {f1:.4f}\n"
        metrics_text += f"Model Parameters: {model.count_params():,}\n"
        
        self.metrics_text.setText(metrics_text)
    
    def plot_training_history(self, history):
        """Plot neural network training history"""
        self.figure.clear()
        
        # Plot training & validation accuracy
        ax1 = self.figure.add_subplot(211)
        ax1.plot(history.history['accuracy'])
        ax1.plot(history.history['val_accuracy'])
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend(['Train', 'Test'])
        
        # Plot training & validation loss
        ax2 = self.figure.add_subplot(212)
        ax2.plot(history.history['loss'])
        ax2.plot(history.history['val_loss'])
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend(['Train', 'Test'])
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def update_visualization_plotly(self):
        """Update visualization based on selected engine (Matplotlib or Plotly)"""
        if not hasattr(self, 'current_model'):
            return
            
        engine = self.viz_engine_combo.currentText()
        
        if engine == "Plotly":
            self.plot_with_plotly()
        else:
            # Re-plot with matplotlib based on current model type
            if isinstance(self.current_model, PCA):
                self.plot_dimensionality_reduction("PCA")
            elif isinstance(self.current_model, LinearDiscriminantAnalysis):
                self.plot_dimensionality_reduction("LDA")
            elif isinstance(self.current_model, KMeans):
                self.plot_clustering_results("K-Means")
            elif isinstance(self.current_model, TSNE):
                self.plot_dimensionality_reduction("t-SNE")
            elif hasattr(self.current_model, 'embedding_') and 'UMAP' in str(type(self.current_model)):
                self.plot_dimensionality_reduction("UMAP")
    
    def plot_with_plotly(self):
        """Create interactive visualization with Plotly"""
        try:
            # Create a figure based on the type of data
            if hasattr(self, 'X_transformed'):
                data = self.X_transformed
                
                # Create a dataframe with the transformed data
                df = pd.DataFrame(data, columns=[f'Component {i+1}' for i in range(data.shape[1])])
                
                # Add target variable if available for coloring
                if hasattr(self, 'y_train') and self.y_train is not None:
                    df['Target'] = self.y_train
                    color_col = 'Target'
                else:
                    color_col = None
                
                # Create appropriate plot based on dimensions
                if data.shape[1] == 1:
                    # 1D data - create histogram
                    fig = px.histogram(df, x='Component 1', color=color_col)
                    title = "1D Projection"
                elif data.shape[1] == 2:
                    # 2D data - create scatter plot
                    fig = px.scatter(df, x='Component 1', y='Component 2', color=color_col)
                    title = "2D Projection"
                else:
                    # 3D data - create 3D scatter
                    fig = px.scatter_3d(df, x='Component 1', y='Component 2', z='Component 3', 
                                    color=color_col)
                    title = "3D Projection"
                    
                # Handle specific model types
                if isinstance(self.current_model, PCA):
                    title = f"PCA {title}"
                elif isinstance(self.current_model, LinearDiscriminantAnalysis):
                    title = f"LDA {title}"
                elif isinstance(self.current_model, TSNE):
                    title = f"t-SNE {title}"
                elif hasattr(self.current_model, 'embedding_'):
                    title = f"UMAP {title}"
                    
            elif hasattr(self, 'cluster_labels'):
                # For clustering results, apply PCA for visualization
                if self.X_train.shape[1] > 2:
                    pca = PCA(n_components=2)
                    data = pca.fit_transform(self.X_train)
                else:
                    data = self.X_train
                    
                # Create a dataframe with the data
                df = pd.DataFrame(data, columns=['Component 1', 'Component 2'])
                df['Cluster'] = self.cluster_labels
                
                # Create scatter plot
                fig = px.scatter(df, x='Component 1', y='Component 2', color='Cluster')
                title = "K-Means Clustering"
            else:
                return
                
            # Update plot layout
            fig.update_layout(
                title=title,
                template="plotly_white",
                height=600
            )
            
            # Save the plot to a temporary HTML file and display it
            temp_file = "temp_plot.html"
            plot(fig, filename=temp_file, auto_open=False)
            
            # Show the plot in a browser
            webbrowser.open('file://' + os.path.realpath(temp_file))
            
            self.status_bar.showMessage(f"Interactive Plotly visualization created")
            
        except ImportError:
            self.show_error("Plotly is not installed. Please install it using 'pip install plotly pandas'")
        except Exception as e:
            self.show_error(f"Error creating Plotly visualization: {str(e)}")
    
    def train_pca(self, param_widgets):
        """Train PCA model with parameters"""
        if self.X_train is None:
            self.show_error("Please load a dataset first")
            return
        
        # Get parameters
        n_components = param_widgets["n_components"].value()
        whiten = param_widgets["whiten"].isChecked()
        svd_solver = param_widgets["svd_solver"].currentText()
        
        # Create and fit PCA model
        pca = PCA(n_components=n_components, whiten=whiten, svd_solver=svd_solver)
        X_transformed = pca.fit_transform(self.X_train)
        
        # Store the model
        self.current_model = pca
        self.X_transformed = X_transformed
        
        # Update visualization
        self.plot_dimensionality_reduction("PCA")
        self.status_bar.showMessage(f"PCA training complete")

    def train_lda(self, param_widgets):
        """Train LDA model with parameters"""
        if self.X_train is None or self.y_train is None:
            self.show_error("Please load a dataset first")
            return
        
        # Get parameters
        n_components = param_widgets["n_components"].value()
        solver = param_widgets["solver"].currentText()
        
        # Create and fit LDA model
        lda = LinearDiscriminantAnalysis(n_components=n_components, solver=solver)
        X_transformed = lda.fit_transform(self.X_train, self.y_train)
        
        # Store the model
        self.current_model = lda
        self.X_transformed = X_transformed
        
        # Update visualization
        self.plot_dimensionality_reduction("LDA")
        self.status_bar.showMessage(f"LDA training complete")

    def train_kmeans(self, param_widgets):
        """Train K-Means model with parameters"""
        if self.X_train is None:
            self.show_error("Please load a dataset first")
            return
        
        # Get parameters
        n_clusters = param_widgets["n_clusters"].value()
        init = param_widgets["init"].currentText()
        max_iter = param_widgets["max_iter"].value()
        
        # Create and fit K-Means model
        kmeans = KMeans(n_clusters=n_clusters, init=init, max_iter=max_iter, random_state=42)
        cluster_labels = kmeans.fit_predict(self.X_train)
        
        # Store the model
        self.current_model = kmeans
        self.cluster_labels = cluster_labels
        
        # Update visualization
        self.plot_clustering_results("K-Means")
        self.status_bar.showMessage(f"K-Means clustering complete")

    def train_tsne(self, param_widgets):
        """Train t-SNE model with parameters"""
        if self.X_train is None:
            self.show_error("Please load a dataset first")
            return
        
        # Get parameters
        n_components = int(param_widgets["n_components (projection)"].currentText())  # Changed from value() to currentText()
        perplexity = param_widgets["perplexity"].value()
        learning_rate = param_widgets["learning_rate"].value()
        
        # Create and fit t-SNE model
        tsne = TSNE(n_components=n_components, perplexity=perplexity, 
                    learning_rate=learning_rate, random_state=42)
        X_transformed = tsne.fit_transform(self.X_train)
        
        # Store the model
        self.current_model = tsne
        self.X_transformed = X_transformed
        
        # Update visualization
        self.plot_dimensionality_reduction("t-SNE")
        self.status_bar.showMessage(f"t-SNE training complete")

    def train_umap(self, param_widgets):
        """Train UMAP model with parameters"""
        if self.X_train is None:
            self.show_error("Please load a dataset first")
            return
        
        try:
            # Get parameters
            n_components = param_widgets["n_components"].value()
            n_neighbors = param_widgets["n_neighbors"].value()
            min_dist = param_widgets["min_dist"].value()
            
            # Create and fit UMAP model
            reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, 
                            min_dist=min_dist, random_state=42)
            X_transformed = reducer.fit_transform(self.X_train)
            
            # Store the model
            self.current_model = reducer
            self.X_transformed = X_transformed
            
            # Update visualization
            self.plot_dimensionality_reduction("UMAP")
            self.status_bar.showMessage(f"UMAP training complete")
        except ImportError:
            self.show_error("UMAP is not installed. Please install it using 'pip install umap-learn'")

    def plot_dimensionality_reduction(self, method_name):
        """Plot dimensionality reduction results"""
        self.figure.clear()
        
        # Check dimensionality of transformed data
        if self.X_transformed.shape[1] < 2:
            # 1D projection
            ax = self.figure.add_subplot(111)
            if self.y_train is not None:
                # For classification, show different classes
                for cls in np.unique(self.y_train):
                    mask = self.y_train == cls
                    ax.hist(self.X_transformed[mask, 0], alpha=0.5, label=f'Class {cls}')
                ax.legend()
            else:
                # For unsupervised methods
                ax.hist(self.X_transformed[:, 0])
            ax.set_title(f"{method_name} 1D Projection")
            ax.set_xlabel("Component 1")
        
        elif self.X_transformed.shape[1] == 2:
            # 2D projection
            ax = self.figure.add_subplot(111)
            if hasattr(self, 'y_train') and self.y_train is not None:
                scatter = ax.scatter(self.X_transformed[:, 0], self.X_transformed[:, 1],
                                c=self.y_train, cmap='viridis', alpha=0.8)
                legend = ax.legend(*scatter.legend_elements(), title="Classes")
                ax.add_artist(legend)
            else:
                ax.scatter(self.X_transformed[:, 0], self.X_transformed[:, 1], alpha=0.8)
            ax.set_title(f"{method_name} 2D Projection")
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
        
        else:
            # 3D projection
            ax = self.figure.add_subplot(111, projection='3d')
            if hasattr(self, 'y_train') and self.y_train is not None:
                scatter = ax.scatter(self.X_transformed[:, 0], self.X_transformed[:, 1], 
                                self.X_transformed[:, 2], c=self.y_train, cmap='viridis', alpha=0.8)
                legend = ax.legend(*scatter.legend_elements(), title="Classes")
                ax.add_artist(legend)
            else:
                ax.scatter(self.X_transformed[:, 0], self.X_transformed[:, 1], 
                        self.X_transformed[:, 2], alpha=0.8)
            ax.set_title(f"{method_name} 3D Projection")
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.set_zlabel("Component 3")
        
        self.canvas.draw()
        
        # Update metrics text with additional information
        metrics_text = f"{method_name} Results:\n\n"
        
        if method_name == "PCA" and hasattr(self, 'current_model'):
            explained_var = self.current_model.explained_variance_ratio_
            metrics_text += "Explained Variance Ratio:\n"
            for i, var in enumerate(explained_var):
                metrics_text += f"Component {i+1}: {var:.4f}\n"
            metrics_text += f"\nTotal Variance Explained: {sum(explained_var):.4f}"
        
        elif method_name == "LDA" and hasattr(self, 'current_model'):
            metrics_text += "Class Separation Analysis:\n"
            # Show discriminant coefficients if available
            if hasattr(self.current_model, 'coef_'):
                metrics_text += "Discriminant Coefficients:\n"
                for i, coef in enumerate(self.current_model.coef_):
                    metrics_text += f"Class {i+1}: {np.array2string(coef, precision=4)}\n"
        
        self.metrics_text.setText(metrics_text)

    def plot_clustering_results(self, method_name):
        """Plot clustering results"""
        self.figure.clear()
        
        if not hasattr(self, 'cluster_labels'):
            return
        
        # Apply PCA if data dimensionality is high
        if self.X_train.shape[1] > 2:
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(self.X_train)
        else:
            X_2d = self.X_train
        
        # Plot clustering results
        ax = self.figure.add_subplot(111)
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=self.cluster_labels, cmap='viridis')
        ax.set_title(f"{method_name} Clustering Results")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        legend = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend)
        
        self.canvas.draw()
        
        # Update metrics text
        metrics_text = f"{method_name} Clustering Results:\n\n"
        
        if method_name == "K-Means" and hasattr(self, 'current_model'):
            metrics_text += f"Number of Clusters: {self.current_model.n_clusters}\n"
            metrics_text += f"Inertia (Sum of Squared Distances): {self.current_model.inertia_:.4f}\n"
            
            # Calculate silhouette score
            try:
                silhouette = silhouette_score(self.X_train, self.cluster_labels)
                metrics_text += f"Silhouette Score: {silhouette:.4f}"
            except:
                pass
        
        self.metrics_text.setText(metrics_text)

    def show_explained_variance(self):
        """Show PCA explained variance plot"""
        if not hasattr(self, 'current_model') or not isinstance(self.current_model, PCA):
            self.show_error("Please train a PCA model first")
            return
        
        # Extract explained variance information
        explained_var = self.current_model.explained_variance_ratio_
        cum_explained_var = np.cumsum(explained_var)
        
        # Plot explained variance
        self.figure.clear()
        ax1 = self.figure.add_subplot(111)
        
        components = range(1, len(explained_var) + 1)
        ax1.bar(components, explained_var, alpha=0.5, label='Individual')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_xlabel('Principal Components')
        ax1.set_title('Explained Variance by Components')
        
        ax2 = ax1.twinx()
        ax2.step(components, cum_explained_var, where='mid', color='r', label='Cumulative')
        ax2.set_ylabel('Cumulative Explained Variance')
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        self.canvas.draw()

    def show_class_separation(self):
        """Show LDA class separation metrics"""
        if not hasattr(self, 'current_model') or not isinstance(self.current_model, LinearDiscriminantAnalysis):
            self.show_error("Please train an LDA model first")
            return
        
        try:
            # Make predictions
            y_pred = self.current_model.predict(self.X_test)
            
            # Calculate confusion matrix
            cm = confusion_matrix(self.y_test, y_pred)
            
            # Plot confusion matrix
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            cax = ax.matshow(cm, cmap='Blues')
            self.figure.colorbar(cax)
            
            # Set labels
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            
            # Add text
            classes = np.unique(self.y_test)
            for i in range(len(classes)):
                for j in range(len(classes)):
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center')
            
            self.canvas.draw()
            
            # Update metrics text
            accuracy = accuracy_score(self.y_test, y_pred)
            try:
                f1 = f1_score(self.y_test, y_pred, average='weighted')
                metrics_text = f"LDA Class Separation Metrics:\n\n"
                metrics_text += f"Accuracy: {accuracy:.4f}\n"
                metrics_text += f"F1 Score: {f1:.4f}"
                self.metrics_text.setText(metrics_text)
            except:
                metrics_text = f"LDA Class Separation Metrics:\n\n"
                metrics_text += f"Accuracy: {accuracy:.4f}"
                self.metrics_text.setText(metrics_text)
            
        except Exception as e:
            self.show_error(f"Error calculating metrics: {str(e)}")

    def show_elbow_method(self):
        """Show K-Means elbow method for determining optimal k"""
        if self.X_train is None:
            self.show_error("Please load a dataset first")
            return
        
        try:
            # Calculate inertia for different k values
            inertias = []
            k_range = range(1, 11)
            
            # Show progress
            self.progress_bar.setValue(0)
            self.status_bar.showMessage("Running elbow method analysis...")
            
            for i, k in enumerate(k_range):
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(self.X_train)
                inertias.append(kmeans.inertia_)
                self.progress_bar.setValue(int((i+1)/len(k_range) * 100))
            
            # Plot elbow curve
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.plot(k_range, inertias, 'bo-')
            ax.set_xlabel('Number of Clusters (k)')
            ax.set_ylabel('Inertia (Sum of Squared Distances)')
            ax.set_title('Elbow Method for Optimal k')
            ax.grid(True)
            
            self.canvas.draw()
            self.status_bar.showMessage("Elbow method analysis complete")
            
        except Exception as e:
            self.show_error(f"Error in elbow method: {str(e)}")

    def run_silhouette_analysis(self):
        """Run silhouette analysis for clustering quality"""
        if self.X_train is None:
            self.show_error("Please load a dataset first")
            return
        
        try:
            # Apply PCA
            if self.X_train.shape[1] > 2:
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(self.X_train)
            else:
                X_pca = self.X_train
            
            # Initialize plot
            self.figure.clear()
            
            # Test a range of cluster numbers
            n_clusters_range = range(2, 6)
            n_clusters = len(n_clusters_range)
            
            # Create a subplot grid
            rows = int(np.ceil(n_clusters / 2))
            cols = 2 if n_clusters > 1 else 1
            
            # Calculate silhouette scores
            for i, n_clusters in enumerate(n_clusters_range):
                # Create subplot
                ax = self.figure.add_subplot(rows, cols, i+1)
                
                # Run K-means
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(self.X_train)
                
                # Calculate silhouette scores
                silhouette_avg = silhouette_score(self.X_train, cluster_labels)
                
                # Plot silhouette
                ax.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
                ax.set_title(f'Clusters: {n_clusters}, Silhouette: {silhouette_avg:.3f}')
                ax.set_xlabel('Component 1')
                ax.set_ylabel('Component 2')
            
            self.figure.tight_layout()
            self.canvas.draw()
            
            self.status_bar.showMessage("Silhouette analysis complete")
            
        except Exception as e:
            self.show_error(f"Error in silhouette analysis: {str(e)}")

    def apply_dataset_split(self):
        """Apply selected train/val/test split to the dataset"""
        if self.X_train is None or self.y_train is None:
            self.show_error("Please load a dataset first")
            return
        
        split_option = self.split_combo.currentText()
        
        # Extract split values
        train_size = float(split_option.split('/')[0]) / 100
        val_size = float(split_option.split('/')[1]) / 100
        
        try:
            # Combine existing train and test data back into full dataset
            X_combined = np.vstack((self.X_train, self.X_test)) if self.X_test is not None else self.X_train
            y_combined = np.concatenate((self.y_train, self.y_test)) if self.y_test is not None else self.y_train
            
            # Calculate test_size based on what's left after train and validation
            test_size = 1.0 - train_size
            
            # First split to get train and temp
            X_train, X_temp, y_train, y_temp = train_test_split(
                X_combined, y_combined, train_size=train_size, random_state=42)
            
            # Then split temp into val and test
            val_relative_size = val_size / (val_size + (1 - train_size - val_size))
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, train_size=val_relative_size, random_state=42)
            
            # Update the class attributes
            self.X_train = X_train
            self.y_train = y_train
            self.X_val = X_val
            self.y_val = y_val
            self.X_test = X_test
            self.y_test = y_test
            
            self.status_bar.showMessage(f"Applied {split_option} split to dataset")
            
        except Exception as e:
            self.show_error(f"Error applying dataset split: {str(e)}")
    
    def run_cross_validation(self):
        """Run k-fold cross-validation with improved model selection"""
        if self.X_train is None or self.y_train is None:
            self.show_error("Please load a dataset first")
            return
        
        try:
            # Get parameters
            k_folds = self.kfold_spin.value()
            metric = self.metrics_combo.currentText().lower()
            model_choice = self.cv_model_combo.currentText()
            
            # Convert metric to scikit-learn scoring parameter
            if metric == "accuracy":
                scoring = "accuracy"
            elif metric == "mse":
                scoring = "neg_mean_squared_error"
            elif metric == "rmse":
                scoring = "neg_root_mean_squared_error"
            elif metric == "r²":
                scoring = "r2"
            elif metric == "f1-score":
                scoring = "f1_weighted"
            else:
                scoring = "accuracy"
            
            # Select model for cross-validation
            if model_choice == "Current Model" and hasattr(self, 'current_model'):
                model = self.current_model
            elif model_choice == "Random Forest":
                model = RandomForestClassifier(random_state=42)
            elif model_choice == "SVM":
                model = SVC(random_state=42)
            elif model_choice == "LogisticRegression":
                model = LogisticRegression(random_state=42)
            else:
                # Default to Random Forest if no valid selection
                model = RandomForestClassifier(random_state=42)
            
            # Skip dimensionality reduction models that don't have predict method
            if hasattr(model, 'predict') == False:
                self.show_error("Selected model does not support prediction. Please select a different model.")
                return
            
            # Run k-fold cross-validation
            kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            
            # Show progress
            self.progress_bar.setValue(0)
            self.status_bar.showMessage(f"Running {k_folds}-fold cross-validation...")
            
            scores = cross_val_score(model, self.X_train, self.y_train, 
                                cv=kf, scoring=scoring)
            
            # Reset progress bar
            self.progress_bar.setValue(100)
            
            # Plot results
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            fold_indices = np.arange(1, k_folds + 1)
            
            # Handle negative scores for MSE/RMSE
            if "neg_" in scoring:
                scores = -scores
            
            # Plot individual fold scores
            ax.bar(fold_indices, scores)
            ax.axhline(y=np.mean(scores), color='r', linestyle='-', label=f'Mean: {np.mean(scores):.4f}')
            ax.axhline(y=np.mean(scores) + np.std(scores), color='g', linestyle='--', 
                    label=f'Std: {np.std(scores):.4f}')
            ax.axhline(y=np.mean(scores) - np.std(scores), color='g', linestyle='--')
            
            ax.set_xlabel('Fold')
            ax.set_ylabel(f'{metric.upper()} Score')
            ax.set_title(f'{k_folds}-Fold Cross-Validation Results')
            ax.set_xticks(fold_indices)
            ax.legend()
            
            self.canvas.draw()
            
            # Update metrics text
            metrics_text = f"Cross-Validation Results ({k_folds}-Fold):\n\n"
            metrics_text += f"Model: {model_choice}\n"
            metrics_text += f"Metric: {metric.upper()}\n"
            metrics_text += f"Mean Score: {np.mean(scores):.4f}\n"
            metrics_text += f"Std Dev: {np.std(scores):.4f}\n\n"
            metrics_text += "Fold Scores:\n"
            
            for i, score in enumerate(scores):
                metrics_text += f"Fold {i+1}: {score:.4f}\n"
            
            self.metrics_text.setText(metrics_text)
            self.status_bar.showMessage(f"{k_folds}-fold cross-validation complete")
            
        except Exception as e:
            self.show_error(f"Error in cross-validation: {str(e)}")

    def show_eigenvalue_computation(self):
        """Show eigenvalue computation for PCA"""
        if not hasattr(self, 'current_model') or not isinstance(self.current_model, PCA):
            self.show_error("Please train a PCA model first")
            return
        
        try:
            # Get the eigenvalues from the model
            eigenvalues = self.current_model.explained_variance_
            
            # Create a heatmap of covariance matrix
            cov_matrix = np.cov(self.X_train, rowvar=False)
            
            # Plot eigenvalues and covariance matrix
            self.figure.clear()
            
            # Create 2x1 subplot grid
            gs = self.figure.add_gridspec(2, 1, height_ratios=[1, 1.5])
            
            # Plot eigenvalues
            ax1 = self.figure.add_subplot(gs[0])
            components = range(1, len(eigenvalues) + 1)
            ax1.bar(components, eigenvalues)
            ax1.set_title('Eigenvalues (Explained Variance)')
            ax1.set_xlabel('Principal Components')
            ax1.set_ylabel('Eigenvalue')
            
            # Plot covariance matrix heatmap
            ax2 = self.figure.add_subplot(gs[1])
            im = ax2.imshow(cov_matrix, cmap='viridis')
            ax2.set_title('Covariance Matrix')
            
            # Add colorbar
            self.figure.colorbar(im, ax=ax2)
            
            # Adjust layout
            self.figure.tight_layout()
            self.canvas.draw()
            
            # Update metrics text
            metrics_text = "Eigenvalue Analysis:\n\n"
            metrics_text += "Top 5 Eigenvalues:\n"
            
            for i, val in enumerate(eigenvalues[:5] if len(eigenvalues) >= 5 else eigenvalues):
                metrics_text += f"Component {i+1}: {val:.4f}\n"
            
            metrics_text += "\nCovariance Matrix Size: {0}x{0}".format(cov_matrix.shape[0])
            
            self.metrics_text.setText(metrics_text)
            self.status_bar.showMessage("Eigenvalue computation complete")
            
        except Exception as e:
            self.show_error(f"Error computing eigenvalues: {str(e)}")
    
    def show_error(self, message):
        """Show error message dialog"""
        QMessageBox.critical(self, "Error", message)
    
def main():
    """Main function to start the application"""
    app = QApplication(sys.argv)
    window = MLCourseGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()