"""
Support Vector Machine model implementation for calcium imaging data.
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict, Any, Optional, Tuple, List, Union

logger = logging.getLogger(__name__)


class SVMModel:
    """
    Support Vector Machine model for decoding behavior from calcium imaging signals.

    This class implements an SVM classifier with PCA preprocessing for dimensionality
    reduction, which is essential for handling high-dimensional calcium imaging data.
    """

    def __init__(self,
                 C: float = 1.0,
                 kernel: str = 'rbf',
                 gamma: str = 'scale',
                 class_weight: Optional[str] = 'balanced',
                 probability: bool = True,
                 random_state: int = 42,
                 n_components: Optional[float] = 0.95,
                 optimize_hyperparams: bool = False,
                 use_pca: bool = True):
        """
        Initialize an SVM model.
        """
        # Store hyperparameters
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.class_weight = class_weight
        self.probability = probability
        self.random_state = random_state
        self.n_components = n_components
        self.optimize_hyperparams = optimize_hyperparams
        self.use_pca = use_pca

        # Initialize SVM model
        self.svm = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            class_weight=class_weight,
            probability=probability,
            random_state=random_state
        )

        # Initialize pipeline with optional PCA
        if use_pca:
            self.model = Pipeline([
                ('scaler', StandardScaler()),  # Added scaler for better PCA performance
                ('pca', PCA(n_components=n_components, random_state=random_state)),
                ('svm', self.svm)
            ])
            logger.info(f"Initialized SVM model with PCA (n_components={n_components})")
        else:
            self.model = Pipeline([
                ('scaler', StandardScaler()),  # Always scale for SVM
                ('svm', self.svm)
            ])
            logger.info("Initialized SVM model without PCA")

    def _prepare_data(self, X, y=None):
        """
        Prepare the data for the model.
        """
        # Convert torch tensors to numpy arrays if needed
        if hasattr(X, 'numpy'):
            X = X.numpy()
        if y is not None and hasattr(y, 'numpy'):
            y = y.numpy()

        # Reshape X to 2D if needed (n_samples, window_size * n_neurons)
        if X.ndim == 3:
            n_samples, window_size, n_neurons = X.shape
            X = X.reshape(n_samples, window_size * n_neurons)

        return X, y

    def optimize_hyperparameters(self, X_train, y_train, cv: int = 3, n_iter: int = 15):
        """
        Optimize model hyperparameters using RandomizedSearchCV.
        """
        logger.info("Optimizing SVM hyperparameters")

        # Prepare data
        X_train, y_train = self._prepare_data(X_train, y_train)

        # Define parameter grid - simplified for calcium imaging data
        if self.use_pca:
            param_grid = {
                'pca__n_components': [0.85, 0.9, 0.95, 0.99],  # Focused range
                'svm__C': [0.1, 1, 10, 100],
                'svm__gamma': ['scale', 'auto', 0.001, 0.01],
                'svm__kernel': ['rbf', 'linear']  # Removed 'poly' - rarely needed
            }
        else:
            param_grid = {
                'svm__C': [0.1, 1, 10, 100],
                'svm__gamma': ['scale', 'auto', 0.001, 0.01],
                'svm__kernel': ['rbf', 'linear']
            }

        # Initialize RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring='balanced_accuracy',  # Better for imbalanced data
            verbose=1,
            random_state=self.random_state,
            n_jobs=-1
        )

        # Fit RandomizedSearchCV
        random_search.fit(X_train, y_train)

        # Get best parameters
        best_params = random_search.best_params_
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best cross-validation score: {random_search.best_score_:.4f}")

        # Update model parameters
        if self.use_pca:
            self.n_components = best_params.get('pca__n_components', self.n_components)
            self.C = best_params.get('svm__C', self.C)
            self.gamma = best_params.get('svm__gamma', self.gamma)
            self.kernel = best_params.get('svm__kernel', self.kernel)

            # Reinitialize pipeline with best parameters
            self.svm = SVC(
                C=self.C,
                kernel=self.kernel,
                gamma=self.gamma,
                class_weight=self.class_weight,
                probability=self.probability,
                random_state=self.random_state
            )

            self.model = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=self.n_components, random_state=self.random_state)),
                ('svm', self.svm)
            ])
        else:
            self.C = best_params.get('svm__C', self.C)
            self.gamma = best_params.get('svm__gamma', self.gamma)
            self.kernel = best_params.get('svm__kernel', self.kernel)

            # Reinitialize SVM with best parameters
            self.svm = SVC(
                C=self.C,
                kernel=self.kernel,
                gamma=self.gamma,
                class_weight=self.class_weight,
                probability=self.probability,
                random_state=self.random_state
            )

            self.model = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', self.svm)
            ])

        return self

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the SVM model.

        Parameters
        ----------
        X_train : torch.Tensor or np.ndarray
            Training features
        y_train : torch.Tensor or np.ndarray
            Training labels
        X_val : torch.Tensor or np.ndarray, optional
            Validation features, by default None
        y_val : torch.Tensor or np.ndarray, optional
            Validation labels, by default None

        Returns
        -------
        self
            The trained model
        """
        logger.info("Training SVM model")

        # Prepare data
        X_train, y_train = self._prepare_data(X_train, y_train)

        # Optimize hyperparameters if requested
        if self.optimize_hyperparams:
            self.optimize_hyperparameters(X_train, y_train)

        # Train the model
        self.model.fit(X_train, y_train)

        logger.info("SVM model training complete")

        # Log PCA explained variance if applicable
        if self.use_pca:
            pca = self.model.named_steps['pca']
            explained_variance = pca.explained_variance_ratio_.sum()
            n_components = pca.n_components_
            logger.info(f"PCA: {n_components} components explain {explained_variance:.2%} of variance")

        # If validation data is provided, report validation score
        if X_val is not None and y_val is not None:
            X_val, y_val = self._prepare_data(X_val, y_val)
            val_score = self.model.score(X_val, y_val)
            logger.info(f"Validation accuracy: {val_score:.4f}")

        return self

    def predict(self, X):
        """
        Make predictions with the trained model.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray
            Input features

        Returns
        -------
        np.ndarray
            Predicted labels
        """
        # Prepare data
        X, _ = self._prepare_data(X)

        # Make predictions
        predictions = self.model.predict(X)

        return predictions

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray
            Input features

        Returns
        -------
        np.ndarray
            Predicted class probabilities
        """
        # Prepare data
        X, _ = self._prepare_data(X)

        # Predict probabilities
        probabilities = self.model.predict_proba(X)

        return probabilities


    
