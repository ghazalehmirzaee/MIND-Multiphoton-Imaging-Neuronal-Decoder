"""
Fully Connected Neural Network model implementation for calcium imaging data.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List, Union

logger = logging.getLogger(__name__)


class FCNNModel(nn.Module):
    """
    Fully Connected Neural Network model for decoding behavior from calcium imaging signals.

    This module implements a multi-layer fully connected neural network with batch
    normalization, ReLU activation, and dropout for regularization.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [256, 128, 64],
                 output_dim: int = 2,
                 dropout_rate: float = 0.4):
        """
        Initialize a Fully Connected Neural Network model.
        """
        super(FCNNModel, self).__init__()

        # Store parameters
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

        # Create layers list to store all layers
        layers = []

        # Input layer
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            # Add linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Add batch normalization
            layers.append(nn.BatchNorm1d(hidden_dim))

            # Add ReLU activation
            layers.append(nn.ReLU())

            # Add dropout for regularization
            # Use smaller dropout rate for the last hidden layer
            current_dropout = dropout_rate if i < len(hidden_dims) - 1 else dropout_rate * 0.75
            layers.append(nn.Dropout(current_dropout))

            # Update previous dimension
            prev_dim = hidden_dim

        # Create sequential model for all hidden layers
        self.hidden_layers = nn.Sequential(*layers)

        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

        logger.info(f"Initialized FCNN model with hidden dims {hidden_dims}")

    def forward(self, x):
        """
        Forward pass through the network.
        """
        # Flatten input if needed
        batch_size = x.size(0)
        if x.dim() > 2:
            x = x.view(batch_size, -1)

        # Pass through hidden layers
        x = self.hidden_layers(x)

        # Pass through output layer
        x = self.output_layer(x)

        return x

    def get_feature_importance(self, window_size: int, n_neurons: int) -> np.ndarray:
        """
        Estimate feature importance using first layer weights.
        """
        # Get the weights of the first layer
        first_layer = None
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                first_layer = layer
                break

        if first_layer is None:
            raise ValueError("Could not find a linear layer in the model")

        # Get the weights
        weights = first_layer.weight.data.cpu().numpy()  # Shape: (hidden_dim, input_dim)

        # Calculate feature importance as the sum of absolute weights
        feature_importance = np.abs(weights).sum(axis=0)  # Shape: (input_dim,)

        # Normalize feature importance
        feature_importance = feature_importance / feature_importance.sum()

        # Reshape to (window_size, n_neurons)
        feature_importance = feature_importance.reshape(window_size, n_neurons)

        return feature_importance


class FCNNWrapper:
    """
    Wrapper for the FCNN model with training and inference functionality.

    This wrapper provides a sklearn-like interface for the FCNN model, making it
    easier to use with the rest of the codebase.
    """

    def __init__(self,
                 input_dim: Optional[int] = None,
                 hidden_dims: List[int] = [256, 128, 64],
                 output_dim: int = 2,
                 dropout_rate: float = 0.4,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5,
                 batch_size: int = 32,
                 num_epochs: int = 100,
                 patience: int = 15,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 random_state: int = 42):
        """
        Initialize the FCNN wrapper.
        """
        # Set random seed for reproducibility
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)

        # Store parameters
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience
        self.device = device
        self.random_state = random_state

        # Model will be initialized during fit
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        logger.info(f"Initialized FCNN wrapper (device={device})")

    def _prepare_data(self, X, y=None):
        """
        Prepare the data for the model.
        """
        # Convert numpy arrays to torch tensors if needed
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
        if y is not None and isinstance(y, np.ndarray):
            y = torch.LongTensor(y)

        # Move tensors to device
        X = X.to(self.device)
        if y is not None:
            y = y.to(self.device)

        return X, y

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the FCNN model.
        """
        logger.info("Training FCNN model")

        # Initialize input dimension if not provided
        if self.input_dim is None:
            if X_train.ndim == 3:
                self.input_dim = X_train.shape[1] * X_train.shape[2]
            else:
                self.input_dim = X_train.shape[1]
            logger.info(f"Input dimension inferred as {self.input_dim}")

        # Initialize model
        self.model = FCNNModel(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
            dropout_rate=self.dropout_rate
        ).to(self.device)

        # Initialize optimizer with AdamW (includes weight decay)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Initialize learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # Initialize loss function with class weighting
        if hasattr(y_train, 'numpy'):
            y_np = y_train.numpy()
        else:
            y_np = y_train

        # Calculate class weights for balanced training
        classes, counts = np.unique(y_np, return_counts=True)
        class_weights = 1.0 / counts
        class_weights = class_weights / class_weights.sum() * len(classes)
        class_weights = torch.FloatTensor(class_weights).to(self.device)

        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Prepare data
        X_train, y_train = self._prepare_data(X_train, y_train)
        if X_val is not None and y_val is not None:
            X_val, y_val = self._prepare_data(X_val, y_val)
            has_validation = True
        else:
            has_validation = False

        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        if has_validation:
            val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False
            )

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_X, batch_y in train_loader:
                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                # Accumulate loss
                train_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()

            # Calculate average training loss and accuracy
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total

            # Validation phase
            if has_validation:
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.model(batch_X)
                        loss = self.criterion(outputs, batch_y)

                        val_loss += loss.item()

                        _, predicted = torch.max(outputs.data, 1)
                        val_total += batch_y.size(0)
                        val_correct += (predicted == batch_y).sum().item()

                val_loss /= len(val_loader)
                val_acc = val_correct / val_total

                # Update learning rate scheduler
                self.scheduler.step(val_loss)

                # Log progress
                logger.info(f"Epoch {epoch + 1}/{self.num_epochs} - "
                           f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                           f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

                # Check for early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save the best model
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        break
            else:
                # Without validation data, just log training metrics
                logger.info(f"Epoch {epoch + 1}/{self.num_epochs} - "
                           f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Load the best model if validation was used
        if has_validation and best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info("Loaded best model based on validation loss")

        logger.info("FCNN model training complete")

        return self

    def predict(self, X):
        """
        Make predictions with the trained model.
        """
        # Ensure model is initialized
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Prepare data
        X, _ = self._prepare_data(X)

        # Set model to evaluation mode
        self.model.eval()

        # Make predictions
        with torch.no_grad():
            outputs = self.model(X)
            _, predicted = torch.max(outputs.data, 1)

            # Convert to numpy array
            predictions = predicted.cpu().numpy()

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
        # Ensure model is initialized
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Prepare data
        X, _ = self._prepare_data(X)

        # Set model to evaluation mode
        self.model.eval()

        # Make predictions
        with torch.no_grad():
            outputs = self.model(X)
            probabilities = F.softmax(outputs, dim=1)

            # Convert to numpy array
            probabilities = probabilities.cpu().numpy()

        return probabilities

    def get_feature_importance(self, window_size: int, n_neurons: int) -> np.ndarray:
        """
        Get feature importance from the model.

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
        # Ensure model is initialized
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        return self.model.get_feature_importance(window_size, n_neurons)

