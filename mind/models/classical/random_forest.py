"""
Random Forest model implementation for calcium imaging data.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import logging
from typing import Dict, Any, Optional, Tuple, List, Union

logger = logging.getLogger(__name__)


class RandomForestModel:
    """
    Random Forest model for decoding behavior from calcium imaging signals.

    This class implements a Random Forest classifier with hyperparameter optimization
    for decoding mouse forelimb movements from calcium imaging data.
    """

    def __init__(self,
                 n_estimators: int = 300,
                 max_depth: int = 20,  # Reduced from 30 - more appropriate for calcium imaging
                 min_samples_split: int = 5,
                 min_samples_leaf: int = 2,
                 max_features: str = 'sqrt',
                 class_weight: Optional[str] = 'balanced',
                 n_jobs: int = -1,
                 random_state: int = 42,
                 optimize_hyperparams: bool = False):
        """
        Initialize a Random Forest model.

        Parameters
        ----------
        n_estimators : int, optional
            Number of trees in the forest, by default 300
        max_depth : int, optional
            Maximum depth of trees, by default 20
        min_samples_split : int, optional
            Minimum samples required to split a node, by default 5
        min_samples_leaf : int, optional
            Minimum samples required at a leaf node, by default 2
        max_features : str, optional
            Number of features to consider for best split, by default 'sqrt'
        class_weight : Optional[str], optional
            Weights for imbalanced classes, by default 'balanced'
        n_jobs : int, optional
            Number of parallel jobs, by default -1 (all CPUs)
        random_state : int, optional
            Random seed for reproducibility, by default 42
        optimize_hyperparams : bool, optional
            Whether to optimize hyperparameters, by default False
        """
        # Store hyperparameters
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.optimize_hyperparams = optimize_hyperparams

        # Initialize the model
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            n_jobs=n_jobs,
            random_state=random_state
        )

        logger.info(f"Initialized Random Forest model with {n_estimators} estimators")

    def _prepare_data(self, X, y=None):
        """
        Prepare the data for the model.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray
            Input features, shape (n_samples, window_size, n_neurons)
        y : torch.Tensor or np.ndarray, optional
            Target labels, shape (n_samples,)

        Returns
        -------
        Tuple[np.ndarray, Optional[np.ndarray]]
            Prepared X and y (if provided)
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

        Parameters
        ----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
        cv : int, optional
            Number of cross-validation folds, by default 3
        n_iter : int, optional
            Number of parameter settings sampled, by default 15

        Returns
        -------
        self
            The model with optimized hyperparameters
        """
        logger.info("Optimizing Random Forest hyperparameters")

        # Prepare data
        X_train, y_train = self._prepare_data(X_train, y_train)

        # Define parameter grid - focused on reasonable ranges for calcium imaging
        param_grid = {
            'n_estimators': [100, 200, 300, 400],  # Removed 500 - diminishing returns
            'max_depth': [10, 15, 20, 25, None],  # Focused range, removed very deep trees
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']  # Removed 'None' - too many features
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
            n_jobs=self.n_jobs
        )

        # Fit RandomizedSearchCV
        random_search.fit(X_train, y_train)

        # Get best parameters
        best_params = random_search.best_params_
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best cross-validation score: {random_search.best_score_:.4f}")

        # Update model with best parameters
        self.n_estimators = best_params.get('n_estimators', self.n_estimators)
        self.max_depth = best_params.get('max_depth', self.max_depth)
        self.min_samples_split = best_params.get('min_samples_split', self.min_samples_split)
        self.min_samples_leaf = best_params.get('min_samples_leaf', self.min_samples_leaf)
        self.max_features = best_params.get('max_features', self.max_features)

        # Reinitialize model with best parameters
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            class_weight=self.class_weight,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )

        return self

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the Random Forest model.

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
        logger.info("Training Random Forest model")

        # Prepare data
        X_train, y_train = self._prepare_data(X_train, y_train)

        # Optimize hyperparameters if requested
        if self.optimize_hyperparams:
            self.optimize_hyperparameters(X_train, y_train)

        # Train the model
        self.model.fit(X_train, y_train)

        logger.info("Random Forest model training complete")

        # Report feature importance statistics
        feature_importances = self.model.feature_importances_
        logger.info(f"Feature importance stats: min={feature_importances.min():.5f}, "
                    f"max={feature_importances.max():.5f}, mean={feature_importances.mean():.5f}")

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

    def get_feature_importance(self, window_size: int, n_neurons: int) -> np.ndarray:
        """
        Get feature importance scores.

        Parameters
        ----------
        window_size : int
            Size of the sliding window
        n_neurons : int
            Number of neurons

        Returns
        -------
        np.ndarray
            Feature importance scores, shape (window_size, n_neurons)
        """
        # Make sure the model is trained
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model must be trained before getting feature importance")

        # Get feature importance
        feature_importances = self.model.feature_importances_

        # Reshape to (window_size, n_neurons)
        feature_importances = feature_importances.reshape(window_size, n_neurons)

        return feature_importances

