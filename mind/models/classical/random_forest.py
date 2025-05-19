"""
Improved Random Forest model with preprocessing for raw calcium signals.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging
from typing import Dict, Any, Optional, Tuple, List, Union

logger = logging.getLogger(__name__)


class RandomForestModel:
    """
    Enhanced Random Forest model with preprocessing for calcium imaging signals.
    """

    def __init__(self,
                 n_estimators: int = 300,  # Increased for better ensemble
                 max_depth: int = None,  # Let trees grow deeper
                 min_samples_split: int = 10,  # Increased to reduce overfitting
                 min_samples_leaf: int = 5,  # Increased for generalization
                 max_features: str = 'log2',  # Different feature sampling
                 class_weight: str = 'balanced',  # Better for standard RF
                 n_jobs: int = -1,
                 random_state: int = 42,
                 criterion: str = 'entropy',  # Try entropy for better splits
                 bootstrap: bool = True,
                 use_pca: bool = True,
                 pca_variance: float = 0.95,
                 optimize_hyperparams: bool = False):
        """
        Initialize improved Random Forest with preprocessing.

        Key improvements:
        - Added PCA for dimensionality reduction
        - Changed to entropy criterion
        - Adjusted tree parameters for better generalization
        - Added data standardization
        """
        # Store parameters
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.criterion = criterion
        self.bootstrap = bootstrap
        self.use_pca = use_pca
        self.pca_variance = pca_variance
        self.optimize_hyperparams = optimize_hyperparams

        # Initialize preprocessing components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=pca_variance, random_state=random_state) if use_pca else None

        # Initialize Random Forest
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            n_jobs=n_jobs,
            random_state=random_state,
            criterion=criterion,
            bootstrap=bootstrap,
            oob_score=bootstrap
        )

        logger.info(f"Initialized Improved Random Forest with {n_estimators} trees and preprocessing")

    def _prepare_data(self, X, y=None):
        """Prepare data with proper reshaping."""
        if hasattr(X, 'numpy'):
            X = X.numpy()
        if y is not None and hasattr(y, 'numpy'):
            y = y.numpy()

        # Reshape if needed
        if X.ndim == 3:
            n_samples, window_size, n_neurons = X.shape
            # Create additional features from temporal statistics
            X_flat = X.reshape(n_samples, window_size * n_neurons)

            # Add temporal features (mean, std, max across time for each neuron)
            X_mean = X.mean(axis=1)
            X_std = X.std(axis=1)
            X_max = X.max(axis=1)

            # Concatenate all features
            X = np.hstack([X_flat, X_mean, X_std, X_max])

        return X, y

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model with preprocessing."""
        logger.info("Training Improved Random Forest with preprocessing")

        # Prepare data
        X_train, y_train = self._prepare_data(X_train, y_train)

        # Apply preprocessing
        X_train_scaled = self.scaler.fit_transform(X_train)

        if self.use_pca:
            X_train_processed = self.pca.fit_transform(X_train_scaled)
            logger.info(f"PCA reduced dimensions from {X_train_scaled.shape[1]} to {X_train_processed.shape[1]}")
        else:
            X_train_processed = X_train_scaled

        # Train model
        self.model.fit(X_train_processed, y_train)

        # Log OOB score
        if hasattr(self.model, 'oob_score_'):
            logger.info(f"Out-of-bag score: {self.model.oob_score_:.4f}")

        # Validate if data provided
        if X_val is not None and y_val is not None:
            X_val, y_val = self._prepare_data(X_val, y_val)
            X_val_scaled = self.scaler.transform(X_val)

            if self.use_pca:
                X_val_processed = self.pca.transform(X_val_scaled)
            else:
                X_val_processed = X_val_scaled

            val_score = self.model.score(X_val_processed, y_val)
            logger.info(f"Validation accuracy: {val_score:.4f}")

        return self

    def predict(self, X):
        """Make predictions with preprocessing."""
        X, _ = self._prepare_data(X)
        X_scaled = self.scaler.transform(X)

        if self.use_pca:
            X_processed = self.pca.transform(X_scaled)
        else:
            X_processed = X_scaled

        return self.model.predict(X_processed)

    def predict_proba(self, X):
        """Predict probabilities with preprocessing."""
        X, _ = self._prepare_data(X)
        X_scaled = self.scaler.transform(X)

        if self.use_pca:
            X_processed = self.pca.transform(X_scaled)
        else:
            X_processed = X_scaled

        return self.model.predict_proba(X_processed)

    def get_feature_importance(self, window_size: int, n_neurons: int) -> np.ndarray:
        """Get feature importance (note: will be in PCA space if PCA is used)."""
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model must be trained before getting feature importance")

        # Return approximate importance in original space
        importance = self.model.feature_importances_

        if self.use_pca and hasattr(self.pca, 'components_'):
            # Transform back from PCA space (approximate)
            importance_original = np.abs(self.pca.components_.T @ importance[:self.pca.n_components_])
            # Take first window_size * n_neurons features (the flattened window)
            importance_window = importance_original[:window_size * n_neurons]
            return importance_window.reshape(window_size, n_neurons)
        else:
            # Direct mapping for non-PCA case
            n_features = window_size * n_neurons
            return importance[:n_features].reshape(window_size, n_neurons)

