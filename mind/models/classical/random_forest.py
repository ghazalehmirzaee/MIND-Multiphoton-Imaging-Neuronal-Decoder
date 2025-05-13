# """
# Random Forest model implementation for calcium imaging data.
# """
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import RandomizedSearchCV
# import logging
# from typing import Dict, Any, Optional, Tuple, List, Union
#
# logger = logging.getLogger(__name__)
#
#
# class RandomForestModel:
#     """
#     Random Forest model for decoding behavior from calcium imaging signals.
#
#     This class implements a Random Forest classifier with hyperparameter optimization
#     for decoding mouse forelimb movements from calcium imaging data.
#     """
#
#     def __init__(self,
#                  n_estimators: int = 300,
#                  max_depth: int = 20,
#                  min_samples_split: int = 5,
#                  min_samples_leaf: int = 2,
#                  max_features: str = 'sqrt',
#                  class_weight: Optional[str] = 'balanced',
#                  n_jobs: int = -1,
#                  random_state: int = 42,
#                  optimize_hyperparams: bool = False,
#                  criterion: str = 'gini',  # Added this parameter
#                  bootstrap: bool = True,  # Added this parameter
#                  min_weight_fraction_leaf: float = 0.0):  # Added this parameter
#         """
#         Initialize a Random Forest model.
#
#         Parameters
#         ----------
#         n_estimators : int, optional
#             Number of trees in the forest, by default 300
#         max_depth : int, optional
#             Maximum depth of trees, by default 20
#         min_samples_split : int, optional
#             Minimum samples required to split a node, by default 5
#         min_samples_leaf : int, optional
#             Minimum samples required at a leaf node, by default 2
#         max_features : str, optional
#             Number of features to consider for best split, by default 'sqrt'
#         class_weight : Optional[str], optional
#             Weights for imbalanced classes, by default 'balanced'
#         n_jobs : int, optional
#             Number of parallel jobs, by default -1 (all CPUs)
#         random_state : int, optional
#             Random seed for reproducibility, by default 42
#         optimize_hyperparams : bool, optional
#             Whether to optimize hyperparameters, by default False
#         criterion : str, optional
#             Function to measure quality of split, by default 'gini'
#         bootstrap : bool, optional
#             Whether to use bootstrap samples, by default True
#         min_weight_fraction_leaf : float, optional
#             Minimum weighted fraction of samples at leaf, by default 0.0
#         """
#         # Store all hyperparameters
#         self.n_estimators = n_estimators
#         self.max_depth = max_depth
#         self.min_samples_split = min_samples_split
#         self.min_samples_leaf = min_samples_leaf
#         self.max_features = max_features
#         self.class_weight = class_weight
#         self.n_jobs = n_jobs
#         self.random_state = random_state
#         self.optimize_hyperparams = optimize_hyperparams
#         self.criterion = criterion
#         self.bootstrap = bootstrap
#         self.min_weight_fraction_leaf = min_weight_fraction_leaf
#
#         # Initialize the model with all parameters
#         self.model = RandomForestClassifier(
#             n_estimators=n_estimators,
#             max_depth=max_depth,
#             min_samples_split=min_samples_split,
#             min_samples_leaf=min_samples_leaf,
#             max_features=max_features,
#             class_weight=class_weight,
#             n_jobs=n_jobs,
#             random_state=random_state,
#             criterion=criterion,
#             bootstrap=bootstrap,
#             min_weight_fraction_leaf=min_weight_fraction_leaf
#         )
#
#         logger.info(f"Initialized Random Forest model with {n_estimators} estimators")
#
#     def _prepare_data(self, X, y=None):
#         """
#         Prepare the data for the model.
#
#         Parameters
#         ----------
#         X : torch.Tensor or np.ndarray
#             Input features, shape (n_samples, window_size, n_neurons)
#         y : torch.Tensor or np.ndarray, optional
#             Target labels, shape (n_samples,)
#
#         Returns
#         -------
#         Tuple[np.ndarray, Optional[np.ndarray]]
#             Prepared X and y (if provided)
#         """
#         # Convert torch tensors to numpy arrays if needed
#         if hasattr(X, 'numpy'):
#             X = X.numpy()
#         if y is not None and hasattr(y, 'numpy'):
#             y = y.numpy()
#
#         # Reshape X to 2D if needed (n_samples, window_size * n_neurons)
#         if X.ndim == 3:
#             n_samples, window_size, n_neurons = X.shape
#             X = X.reshape(n_samples, window_size * n_neurons)
#
#         return X, y
#
#     def optimize_hyperparameters(self, X_train, y_train, cv: int = 3, n_iter: int = 15):
#         """
#         Optimize model hyperparameters using RandomizedSearchCV.
#
#         Parameters
#         ----------
#         X_train : np.ndarray
#             Training features
#         y_train : np.ndarray
#             Training labels
#         cv : int, optional
#             Number of cross-validation folds, by default 3
#         n_iter : int, optional
#             Number of parameter settings sampled, by default 15
#
#         Returns
#         -------
#         self
#             The model with optimized hyperparameters
#         """
#         logger.info("Optimizing Random Forest hyperparameters")
#
#         # Prepare data
#         X_train, y_train = self._prepare_data(X_train, y_train)
#
#         # Define parameter grid - focused on reasonable ranges for calcium imaging
#         param_grid = {
#             'n_estimators': [100, 200, 300, 400],
#             'max_depth': [10, 15, 20, 25, None],
#             'min_samples_split': [2, 5, 10],
#             'min_samples_leaf': [1, 2, 4],
#             'max_features': ['sqrt', 'log2'],
#             'criterion': ['gini', 'entropy'],
#             'bootstrap': [True, False],
#             'class_weight': ['balanced', 'balanced_subsample', None]
#         }
#
#         # Initialize RandomizedSearchCV
#         random_search = RandomizedSearchCV(
#             estimator=self.model,
#             param_distributions=param_grid,
#             n_iter=n_iter,
#             cv=cv,
#             scoring='balanced_accuracy',
#             verbose=1,
#             random_state=self.random_state,
#             n_jobs=self.n_jobs
#         )
#
#         # Fit RandomizedSearchCV
#         random_search.fit(X_train, y_train)
#
#         # Get best parameters
#         best_params = random_search.best_params_
#         logger.info(f"Best parameters: {best_params}")
#         logger.info(f"Best cross-validation score: {random_search.best_score_:.4f}")
#
#         # Update model with best parameters
#         self.n_estimators = best_params.get('n_estimators', self.n_estimators)
#         self.max_depth = best_params.get('max_depth', self.max_depth)
#         self.min_samples_split = best_params.get('min_samples_split', self.min_samples_split)
#         self.min_samples_leaf = best_params.get('min_samples_leaf', self.min_samples_leaf)
#         self.max_features = best_params.get('max_features', self.max_features)
#         self.criterion = best_params.get('criterion', self.criterion)
#         self.bootstrap = best_params.get('bootstrap', self.bootstrap)
#         self.class_weight = best_params.get('class_weight', self.class_weight)
#
#         # Reinitialize model with best parameters
#         self.model = RandomForestClassifier(
#             n_estimators=self.n_estimators,
#             max_depth=self.max_depth,
#             min_samples_split=self.min_samples_split,
#             min_samples_leaf=self.min_samples_leaf,
#             max_features=self.max_features,
#             class_weight=self.class_weight,
#             n_jobs=self.n_jobs,
#             random_state=self.random_state,
#             criterion=self.criterion,
#             bootstrap=self.bootstrap,
#             min_weight_fraction_leaf=self.min_weight_fraction_leaf
#         )
#
#         return self
#
#     def fit(self, X_train, y_train, X_val=None, y_val=None):
#         """
#         Train the Random Forest model.
#
#         Parameters
#         ----------
#         X_train : torch.Tensor or np.ndarray
#             Training features
#         y_train : torch.Tensor or np.ndarray
#             Training labels
#         X_val : torch.Tensor or np.ndarray, optional
#             Validation features, by default None
#         y_val : torch.Tensor or np.ndarray, optional
#             Validation labels, by default None
#
#         Returns
#         -------
#         self
#             The trained model
#         """
#         logger.info("Training Random Forest model")
#
#         # Prepare data
#         X_train, y_train = self._prepare_data(X_train, y_train)
#
#         # Optimize hyperparameters if requested
#         if self.optimize_hyperparams:
#             self.optimize_hyperparameters(X_train, y_train)
#
#         # Train the model
#         self.model.fit(X_train, y_train)
#
#         logger.info("Random Forest model training complete")
#
#         # Report feature importance statistics
#         feature_importances = self.model.feature_importances_
#         logger.info(f"Feature importance stats: min={feature_importances.min():.5f}, "
#                     f"max={feature_importances.max():.5f}, mean={feature_importances.mean():.5f}")
#
#         # If validation data is provided, report validation score
#         if X_val is not None and y_val is not None:
#             X_val, y_val = self._prepare_data(X_val, y_val)
#             val_score = self.model.score(X_val, y_val)
#             logger.info(f"Validation accuracy: {val_score:.4f}")
#
#         return self
#
#     def predict(self, X):
#         """
#         Make predictions with the trained model.
#
#         Parameters
#         ----------
#         X : torch.Tensor or np.ndarray
#             Input features
#
#         Returns
#         -------
#         np.ndarray
#             Predicted labels
#         """
#         # Prepare data
#         X, _ = self._prepare_data(X)
#
#         # Make predictions
#         predictions = self.model.predict(X)
#
#         return predictions
#
#     def predict_proba(self, X):
#         """
#         Predict class probabilities.
#
#         Parameters
#         ----------
#         X : torch.Tensor or np.ndarray
#             Input features
#
#         Returns
#         -------
#         np.ndarray
#             Predicted class probabilities
#         """
#         # Prepare data
#         X, _ = self._prepare_data(X)
#
#         # Predict probabilities
#         probabilities = self.model.predict_proba(X)
#
#         return probabilities
#
#     def get_feature_importance(self, window_size: int, n_neurons: int) -> np.ndarray:
#         """
#         Get feature importance scores.
#
#         Parameters
#         ----------
#         window_size : int
#             Size of the sliding window
#         n_neurons : int
#             Number of neurons
#
#         Returns
#         -------
#         np.ndarray
#             Feature importance scores, shape (window_size, n_neurons)
#         """
#         # Make sure the model is trained
#         if not hasattr(self.model, 'feature_importances_'):
#             raise ValueError("Model must be trained before getting feature importance")
#
#         # Get feature importance
#         feature_importances = self.model.feature_importances_
#
#         # Reshape to (window_size, n_neurons)
#         feature_importances = feature_importances.reshape(window_size, n_neurons)
#
#         return feature_importances
#
#
# """
# Fixed Random Forest model implementation for calcium imaging data.
# """
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.utils.class_weight import compute_class_weight
# import logging
# from typing import Dict, Any, Optional, Tuple, List, Union
#
# logger = logging.getLogger(__name__)
#
#
# class RandomForestModel:
#     """
#     Fixed Random Forest model with better handling of imbalanced data.
#     """
#
#     def __init__(self,
#                  n_estimators: int = 200,
#                  max_depth: int = 15,
#                  min_samples_split: int = 5,
#                  min_samples_leaf: int = 2,
#                  max_features: str = 'sqrt',
#                  class_weight: Optional[str] = 'balanced_subsample',
#                  n_jobs: int = -1,
#                  random_state: int = 42,
#                  criterion: str = 'gini',
#                  bootstrap: bool = True,
#                  optimize_hyperparams: bool = False):
#         """
#         Initialize Random Forest with parameters optimized for calcium imaging data.
#
#         Parameters
#         ----------
#         n_estimators : int, optional
#             Number of trees in the forest, by default 200
#         max_depth : int, optional
#             Maximum depth of trees, by default 15
#         min_samples_split : int, optional
#             Minimum samples required to split a node, by default 5
#         min_samples_leaf : int, optional
#             Minimum samples required at a leaf node, by default 2
#         max_features : str, optional
#             Number of features to consider for best split, by default 'sqrt'
#         class_weight : Optional[str], optional
#             Weights for imbalanced classes, by default 'balanced_subsample'
#         n_jobs : int, optional
#             Number of parallel jobs, by default -1 (all CPUs)
#         random_state : int, optional
#             Random seed for reproducibility, by default 42
#         criterion : str, optional
#             Function to measure split quality, by default 'gini'
#         bootstrap : bool, optional
#             Whether bootstrap samples are used for building trees, by default True
#         optimize_hyperparams : bool, optional
#             Whether to optimize hyperparameters, by default False
#
#         Changed parameters:
#         - n_estimators: Reduced to 200 for faster training
#         - max_depth: Reduced to 15 to avoid overfitting
#         - class_weight: Changed to 'balanced_subsample' for better imbalance handling
#         """
#         self.n_estimators = n_estimators
#         self.max_depth = max_depth
#         self.min_samples_split = min_samples_split
#         self.min_samples_leaf = min_samples_leaf
#         self.max_features = max_features
#         self.class_weight = class_weight
#         self.n_jobs = n_jobs
#         self.random_state = random_state
#         self.criterion = criterion
#         self.bootstrap = bootstrap
#         self.optimize_hyperparams = optimize_hyperparams
#
#         # Initialize model with balanced_subsample for better performance
#         self.model = RandomForestClassifier(
#             n_estimators=n_estimators,
#             max_depth=max_depth,
#             min_samples_split=min_samples_split,
#             min_samples_leaf=min_samples_leaf,
#             max_features=max_features,
#             class_weight=class_weight,
#             n_jobs=n_jobs,
#             random_state=random_state,
#             criterion=criterion,
#             bootstrap=bootstrap,
#             oob_score=bootstrap  # Only use OOB when bootstrap is True
#         )
#
#         logger.info(f"Initialized Random Forest with {n_estimators} trees and balanced subsample")
#
#     def _prepare_data(self, X, y=None):
#         """Prepare data for the model."""
#         if hasattr(X, 'numpy'):
#             X = X.numpy()
#         if y is not None and hasattr(y, 'numpy'):
#             y = y.numpy()
#
#         # Reshape if needed
#         if X.ndim == 3:
#             n_samples, window_size, n_neurons = X.shape
#             X = X.reshape(n_samples, window_size * n_neurons)
#
#         return X, y
#
#     def fit(self, X_train, y_train, X_val=None, y_val=None):
#         """Train the Random Forest model."""
#         logger.info("Training Random Forest model with balanced subsampling")
#
#         # Prepare data
#         X_train, y_train = self._prepare_data(X_train, y_train)
#
#         # Log class distribution
#         unique, counts = np.unique(y_train, return_counts=True)
#         logger.info(f"Training class distribution: {dict(zip(unique, counts))}")
#
#         # Train model
#         self.model.fit(X_train, y_train)
#
#         # Log OOB score as internal validation
#         if hasattr(self.model, 'oob_score_'):
#             logger.info(f"Out-of-bag score: {self.model.oob_score_:.4f}")
#
#         # Validate if data provided
#         if X_val is not None and y_val is not None:
#             X_val, y_val = self._prepare_data(X_val, y_val)
#             val_score = self.model.score(X_val, y_val)
#             logger.info(f"Validation accuracy: {val_score:.4f}")
#
#         return self
#
#     def predict(self, X):
#         """Make predictions."""
#         X, _ = self._prepare_data(X)
#         return self.model.predict(X)
#
#     def predict_proba(self, X):
#         """Predict class probabilities."""
#         X, _ = self._prepare_data(X)
#         return self.model.predict_proba(X)
#
#     def get_feature_importance(self, window_size: int, n_neurons: int) -> np.ndarray:
#         """Get feature importance scores reshaped to (window_size, n_neurons)."""
#         if not hasattr(self.model, 'feature_importances_'):
#             raise ValueError("Model must be trained before getting feature importance")
#
#         importance = self.model.feature_importances_
#         return importance.reshape(window_size, n_neurons)
#


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
