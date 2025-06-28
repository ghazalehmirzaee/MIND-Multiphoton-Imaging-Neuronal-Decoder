"""
Random Forest model implementation for calcium imaging data.
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
    Optimized Random Forest model for neural decoding.
    """

    def __init__(self,
                 n_estimators: int = 300,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 5,
                 min_samples_leaf: int = 2,
                 max_features: str = 'sqrt',
                 class_weight: str = 'balanced_subsample',
                 n_jobs: int = -1,
                 random_state: int = 42,
                 criterion: str = 'gini',
                 bootstrap: bool = True,
                 use_pca: bool = False,  # Changed to False by default
                 pca_variance: float = 0.95,
                 optimize_hyperparams: bool = False):
        """
        Initialize Random Forest model with preprocessing options.
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

        logger.info(f"Initialized Random Forest with {n_estimators} trees and PCA={use_pca}")

    def _prepare_data(self, X, y=None):
        """
        Prepare data for model training or inference.
        """
        # Convert torch tensors to numpy if needed
        if hasattr(X, 'numpy'):
            X = X.numpy()
        if y is not None and hasattr(y, 'numpy'):
            y = y.numpy()

        # Reshape if needed (without adding potentially noisy features)
        if X.ndim == 3:
            n_samples, window_size, n_neurons = X.shape
            X = X.reshape(n_samples, window_size * n_neurons)

        return X, y

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the Random Forest model.
        """
        logger.info("Training Random Forest")

        # Prepare data
        X_train, y_train = self._prepare_data(X_train, y_train)

        # Apply preprocessing - standardization
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Apply PCA if requested
        if self.use_pca:
            n_components = min(self.pca.n_components, X_train_scaled.shape[1])
            self.pca.n_components = n_components
            X_train_processed = self.pca.fit_transform(X_train_scaled)
            logger.info(f"PCA reduced dimensions from {X_train_scaled.shape[1]} to {X_train_processed.shape[1]} "
                        f"({self.pca.explained_variance_ratio_.sum():.2%} explained variance)")
        else:
            X_train_processed = X_train_scaled

        # Train model
        self.model.fit(X_train_processed, y_train)

        # Log OOB score if available
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
        """
        Make predictions with the trained model.
        """
        # Prepare data
        X, _ = self._prepare_data(X)

        # Apply preprocessing
        X_scaled = self.scaler.transform(X)

        if self.use_pca:
            X_processed = self.pca.transform(X_scaled)
        else:
            X_processed = X_scaled

        # Make predictions
        return self.model.predict(X_processed)

    def predict_proba(self, X):
        """
        Predict class probabilities.
        """
        # Prepare data
        X, _ = self._prepare_data(X)

        # Apply preprocessing
        X_scaled = self.scaler.transform(X)

        if self.use_pca:
            X_processed = self.pca.transform(X_scaled)
        else:
            X_processed = X_scaled

        # Predict probabilities
        return self.model.predict_proba(X_processed)

    def get_feature_importance(self, window_size: int, n_neurons: int) -> np.ndarray:
        """
        Get feature importance matrix.

        This method extracts feature importance from the trained model and reshapes
        it to a matrix of shape (window_size, n_neurons).
        """
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model must be trained before getting feature importance")

        # Get feature importances
        importances = self.model.feature_importances_

        if self.use_pca:
            # When using PCA, we can't directly map back to original features
            # Create an approximate mapping using PCA components
            try:
                # Get PCA components
                components = self.pca.components_  # shape: (n_components, n_features)

                # Weight components by explained variance ratio
                weighted_components = components.T * self.pca.explained_variance_ratio_

                # Sum across components to get importance for original features
                original_importances = np.abs(weighted_components).sum(axis=1)

                # Normalize
                original_importances = original_importances / original_importances.sum()

                # Reshape to (window_size, n_neurons)
                importance_matrix = original_importances[:window_size * n_neurons].reshape(window_size, n_neurons)

                return importance_matrix
            except:
                # Fallback: use equal importance
                logger.warning("Could not map PCA feature importance back to original space")
                importance_matrix = np.ones((window_size, n_neurons)) / (window_size * n_neurons)
                return importance_matrix
        else:
            # Direct mapping for non-PCA case
            # Take only the first window_size * n_neurons features
            n_features = min(len(importances), window_size * n_neurons)
            importance_matrix = importances[:n_features].reshape(window_size, n_neurons)

            return importance_matrix

