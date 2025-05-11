# mind/models/classical/mlp.py
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
import joblib
from typing import Dict, Any, Optional, List, Union


class MLPModel:
    """
    Multi-Layer Perceptron model for neural signal classification.
    """

    def __init__(self,
                 hidden_layer_sizes: tuple = (64, 128, 32),
                 activation: str = 'relu',
                 solver: str = 'adam',
                 alpha: float = 0.0001,
                 batch_size: Union[int, str] = 'auto',
                 learning_rate: str = 'adaptive',
                 learning_rate_init: float = 0.001,
                 max_iter: int = 500,
                 early_stopping: bool = True,
                 validation_fraction: float = 0.1,
                 random_state: int = 42,
                 optimize_hyperparams: bool = False):
        """
        Initialize MLP model.

        Parameters
        ----------
        hidden_layer_sizes : tuple, optional
            Number of neurons in each hidden layer
        activation : str, optional
            Activation function
        solver : str, optional
            Solver for weight optimization
        alpha : float, optional
            L2 penalty parameter
        batch_size : int or 'auto', optional
            Size of minibatches
        learning_rate : str, optional
            Learning rate schedule
        learning_rate_init : float, optional
            Initial learning rate
        max_iter : int, optional
            Maximum number of iterations
        early_stopping : bool, optional
            Whether to use early stopping
        validation_fraction : float, optional
            Fraction of training data to use for validation
        random_state : int, optional
            Random seed for reproducibility
        optimize_hyperparams : bool, optional
            Whether to optimize hyperparameters using randomized search
        """
        self.params = {
            'hidden_layer_sizes': hidden_layer_sizes,
            'activation': activation,
            'solver': solver,
            'alpha': alpha,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'learning_rate_init': learning_rate_init,
            'max_iter': max_iter,
            'early_stopping': early_stopping,
            'validation_fraction': validation_fraction,
            'random_state': random_state
        }
        self.optimize_hyperparams = optimize_hyperparams
        self.model = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train the MLP model.

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
                'hidden_layer_sizes': [(64,), (128,), (64, 32), (64, 128, 32)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01, 0.1]
            }

            # We'll use a different setup when optimizing hyperparameters
            temp_params = self.params.copy()
            # Turn off early stopping to use custom validation set
            temp_params['early_stopping'] = False

            mlp = MLPClassifier(**temp_params)

            search = RandomizedSearchCV(
                mlp, param_grid, n_iter=10, cv=3, random_state=self.params['random_state'],
                scoring='f1_weighted', n_jobs=-1
            )

            # Combine X_train and X_val, y_train and y_val for cross-validation
            if X_val is not None and y_val is not None:
                X_combined = np.vstack((X_train, X_val))
                y_combined = np.hstack((y_train, y_val))
                search.fit(X_combined, y_combined)
            else:
                search.fit(X_train, y_train)

            self.model = search.best_estimator_
            print(f"Best hyperparameters: {search.best_params_}")
        else:
            print("Training MLP model with fixed hyperparameters...")

            # Adjust parameters if we have a separate validation set
            if X_val is not None and y_val is not None and self.params['early_stopping']:
                # If we're using a separate validation set, we'll use it instead of built-in early stopping
                temp_params = self.params.copy()
                temp_params['early_stopping'] = False
                self.model = MLPClassifier(**temp_params)

                print("Using separate validation set for early stopping...")
                best_val_loss = float('inf')
                best_iter = 0
                patience = 10
                no_improve_count = 0

                # Fit model with warm_start=True to enable incremental fitting
                self.model.warm_start = True
                self.model.max_iter = 50  # Reduce iteration count to check validation more frequently

                for i in range(20):  # Maximum 20 chunks of training
                    self.model.fit(X_train, y_train)
                    val_loss = -self.model.score(X_val, y_val)  # Negative score as loss

                    print(f"  Epoch {(i + 1) * 50}, validation loss: {val_loss:.4f}")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_iter = i
                        no_improve_count = 0
                    else:
                        no_improve_count += 1

                    if no_improve_count >= patience:
                        print(f"  Early stopping at epoch {(i + 1) * 50}")
                        break

                print(f"Best validation loss: {best_val_loss:.4f} at epoch {(best_iter + 1) * 50}")
            else:
                # Use standard training with built-in early stopping
                self.model = MLPClassifier(**self.params)
                self.model.fit(X_train, y_train)

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
        print(f"Model loaded from {file_path}")

        