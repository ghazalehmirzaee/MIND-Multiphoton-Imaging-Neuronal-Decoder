# mind/models/classical/svm.py
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
import joblib
from typing import Dict, Any, Optional


class SVMModel:
    """
    Support Vector Machine model for neural signal classification.
    """

    def __init__(self,
                 C: float = 1.0,
                 kernel: str = 'rbf',
                 gamma: str = 'scale',
                 class_weight: Optional[str] = 'balanced',
                 random_state: int = 42,
                 apply_pca: bool = True,
                 pca_components: float = 0.95,
                 optimize_hyperparams: bool = False):
        """
        Initialize SVM model.

        Parameters
        ----------
        C : float, optional
            Regularization parameter
        kernel : str, optional
            Kernel type
        gamma : str or float, optional
            Kernel coefficient
        class_weight : str or None, optional
            Weights associated with classes
        random_state : int, optional
            Random seed for reproducibility
        apply_pca : bool, optional
            Whether to apply PCA for dimensionality reduction
        pca_components : float, optional
            Number of components or variance ratio to keep in PCA
        optimize_hyperparams : bool, optional
            Whether to optimize hyperparameters using randomized search
        """
        self.params = {
            'C': C,
            'kernel': kernel,
            'gamma': gamma,
            'class_weight': class_weight,
            'random_state': random_state,
            'probability': True  # Enable probability estimates
        }
        self.apply_pca = apply_pca
        self.pca_components = pca_components
        self.optimize_hyperparams = optimize_hyperparams
        self.model = None
        self.pca = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train the SVM model.

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
        # Apply PCA if specified
        if self.apply_pca:
            print(f"Applying PCA (components={self.pca_components})...")
            self.pca = PCA(n_components=self.pca_components, random_state=self.params['random_state'])
            X_train_pca = self.pca.fit_transform(X_train)
            X_val_pca = self.pca.transform(X_val) if X_val is not None else None
            print(f"  X_train: {X_train.shape} -> {X_train_pca.shape}")
            if X_val is not None:
                print(f"  X_val: {X_val.shape} -> {X_val_pca.shape}")
        else:
            X_train_pca = X_train
            X_val_pca = X_val

        if self.optimize_hyperparams and X_val_pca is not None and y_val is not None:
            print("Optimizing hyperparameters...")
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.01, 0.1, 1],
                'kernel': ['rbf', 'poly', 'sigmoid']
            }

            svm = SVC(probability=True, random_state=self.params['random_state'],
                      class_weight=self.params['class_weight'])

            search = RandomizedSearchCV(
                svm, param_grid, n_iter=10, cv=3, random_state=self.params['random_state'],
                scoring='f1_weighted', n_jobs=-1
            )

            search.fit(X_train_pca, y_train)
            self.model = search.best_estimator_
            print(f"Best hyperparameters: {search.best_params_}")
        else:
            print("Training SVM model with fixed hyperparameters...")
            self.model = SVC(**self.params)
            self.model.fit(X_train_pca, y_train)

        # Compute training and validation metrics
        train_acc = self.model.score(X_train_pca, y_train)
        metrics = {'train_accuracy': train_acc}

        if X_val_pca is not None and y_val is not None:
            val_acc = self.model.score(X_val_pca, y_val)
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

        # Apply PCA if it was used during training
        if self.pca is not None:
            X = self.pca.transform(X)

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

        # Apply PCA if it was used during training
        if self.pca is not None:
            X = self.pca.transform(X)

        return self.model.predict_proba(X)

    def save(self, file_path: str) -> None:
        """
        Save the trained model and PCA transformer.

        Parameters
        ----------
        file_path : str
            Path to save the model
        """
        if self.model is None:
            raise ValueError("No trained model to save.")

        model_data = {
            'model': self.model,
            'pca': self.pca
        }

        joblib.dump(model_data, file_path)
        print(f"Model saved to {file_path}")

    def load(self, file_path: str) -> None:
        """
        Load a trained model and PCA transformer.

        Parameters
        ----------
        file_path : str
            Path to the saved model
        """
        model_data = joblib.load(file_path)
        self.model = model_data['model']
        self.pca = model_data['pca']
        print(f"Model loaded from {file_path}")

