# mind/models/classical/random_forest.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import joblib
from typing import Dict, Any, Optional


class RandomForestModel:
    """
    Random Forest model for neural signal classification.
    """

    def __init__(self,
                 n_estimators: int = 200,
                 max_depth: Optional[int] = 30,
                 min_samples_split: int = 5,
                 min_samples_leaf: int = 2,
                 max_features: str = 'sqrt',
                 class_weight: Optional[str] = 'balanced',
                 random_state: int = 42,
                 n_jobs: int = -1,
                 optimize_hyperparams: bool = False):
        """
        Initialize Random Forest model.

        Parameters
        ----------
        n_estimators : int, optional
            Number of trees in the forest
        max_depth : int or None, optional
            Maximum depth of the trees
        min_samples_split : int, optional
            Minimum samples required to split a node
        min_samples_leaf : int, optional
            Minimum samples required at a leaf node
        max_features : str, optional
            Number of features to consider for best split
        class_weight : str or None, optional
            Weights associated with classes
        random_state : int, optional
            Random seed for reproducibility
        n_jobs : int, optional
            Number of jobs to run in parallel
        optimize_hyperparams : bool, optional
            Whether to optimize hyperparameters using randomized search
        """
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'class_weight': class_weight,
            'random_state': random_state,
            'n_jobs': n_jobs
        }
        self.optimize_hyperparams = optimize_hyperparams
        self.model = None
        self.feature_importances_ = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train the Random Forest model.

        Parameters
        ----------
        X_train : np.ndarray
            Training data
        y_train : np.ndarray
            Training labels
        X_val : np.ndarray, optional
            Validation data
        y_val : np.ndarray, optional
            Validation labels

        Returns
        -------
        Dict[str, Any]
            Dictionary containing training metrics
        """
        if self.optimize_hyperparams and X_val is not None and y_val is not None:
            print("Optimizing hyperparameters...")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [20, 30, 40, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }

            rf = RandomForestClassifier(random_state=self.params['random_state'],
                                        n_jobs=self.params['n_jobs'],
                                        class_weight=self.params['class_weight'])

            search = RandomizedSearchCV(
                rf, param_grid, n_iter=20, cv=3, random_state=self.params['random_state'],
                scoring='f1_weighted', n_jobs=self.params['n_jobs']
            )

            search.fit(X_train, y_train)
            self.model = search.best_estimator_
            print(f"Best hyperparameters: {search.best_params_}")
        else:
            print("Training Random Forest model with fixed hyperparameters...")
            self.model = RandomForestClassifier(**self.params)
            self.model.fit(X_train, y_train)

        # Extract feature importances
        self.feature_importances_ = self.model.feature_importances_

        # Compute training and validation metrics
        train_acc = self.model.score(X_train, y_train)
        metrics = {'train_accuracy': train_acc}

        if X_val is not None and y_val is not None:
            val_acc = self.model.score(X_val, y_val)
            metrics['val_accuracy'] = val_acc

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the trained model.

        Parameters
        ----------
        X : np.ndarray
            Input data

        Returns
        -------
        np.ndarray
            Predicted labels
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.

        Parameters
        ----------
        X : np.ndarray
            Input data

        Returns
        -------
        np.ndarray
            Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        return self.model.predict_proba(X)

    def save(self, file_path: str) -> None:
        """
        Save the trained model.

        Parameters
        ----------
        file_path : str
            Path to save the model
        """
        if self.model is None:
            raise ValueError("No trained model to save.")

        joblib.dump(self.model, file_path)
        print(f"Model saved to {file_path}")

    def load(self, file_path: str) -> None:
        """
        Load a trained model.

        Parameters
        ----------
        file_path : str
            Path to the saved model
        """
        self.model = joblib.load(file_path)
        self.feature_importances_ = self.model.feature_importances_
        print(f"Model loaded from {file_path}")

    def get_feature_importances(self) -> np.ndarray:
        """
        Get feature importances from the trained model.

        Returns
        -------
        np.ndarray
            Feature importances
        """
        if self.feature_importances_ is None:
            raise ValueError("Feature importances not available. Train the model first.")

        return self.feature_importances_

