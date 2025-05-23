"""
Random Forest model optimized for calcium imaging neural decoding.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class RandomForestModel:
    """Random Forest model with feature importance extraction."""

    def __init__(self, **kwargs):
        """Initialize with configuration parameters."""
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=kwargs.get('n_estimators', 300),
            max_depth=kwargs.get('max_depth', 20),
            min_samples_split=kwargs.get('min_samples_split', 5),
            min_samples_leaf=kwargs.get('min_samples_leaf', 2),
            class_weight=kwargs.get('class_weight', 'balanced_subsample'),
            random_state=kwargs.get('random_state', 42),
            n_jobs=-1
        )
        logger.info(f"Initialized Random Forest with {self.model.n_estimators} trees")

    def _prepare_data(self, X, y=None):
        """Convert and reshape data for sklearn."""
        if hasattr(X, 'numpy'):
            X = X.numpy()
        if y is not None and hasattr(y, 'numpy'):
            y = y.numpy()

        # Flatten 3D to 2D: (samples, window_size * n_neurons)
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)

        return X, y

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model."""
        X_train, y_train = self._prepare_data(X_train, y_train)
        X_train = self.scaler.fit_transform(X_train)

        self.model.fit(X_train, y_train)

        if X_val is not None and y_val is not None:
            X_val, y_val = self._prepare_data(X_val, y_val)
            X_val = self.scaler.transform(X_val)
            score = self.model.score(X_val, y_val)
            logger.info(f"Validation accuracy: {score:.4f}")

        return self

    def predict(self, X):
        """Make predictions."""
        X, _ = self._prepare_data(X)
        X = self.scaler.transform(X)
        return self.model.predict(X)

    def predict_proba(self, X):
        """Get prediction probabilities."""
        X, _ = self._prepare_data(X)
        X = self.scaler.transform(X)
        return self.model.predict_proba(X)

    def get_feature_importance(self, window_size: int, n_neurons: int) -> np.ndarray:
        """Extract feature importance matrix."""
        importance = self.model.feature_importances_
        return importance.reshape(window_size, n_neurons)

