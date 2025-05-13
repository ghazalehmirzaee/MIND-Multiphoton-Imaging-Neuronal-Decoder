"""
Convolutional Neural Network model implementation for calcium imaging data.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List, Union

logger = logging.getLogger(__name__)


class CNNModel(nn.Module):
    """
    Simplified Convolutional Neural Network model for decoding behavior from calcium imaging signals.

    This model uses 1D convolutions to capture temporal patterns in the calcium
    imaging data, with batch normalization and dropout for regularization.
    """

    def __init__(self,
                 window_size: int,
                 n_neurons: int,
                 n_filters: List[int] = [64, 128, 256],
                 kernel_size: int = 3,
                 output_dim: int = 2,
                 dropout_rate: float = 0.3):
        """
        Initialize a CNN model.

        Parameters
        ----------
        window_size : int
            Size of the sliding window (time dimension)
        n_neurons : int
            Number of neurons (feature dimension)
        n_filters : List[int], optional
            Number of filters in each convolutional layer, by default [64, 128, 256]
        kernel_size : int, optional
            Size of the convolutional kernel, by default 3
        output_dim : int, optional
            Output dimension (number of classes), by default 2
        dropout_rate : float, optional
            Dropout rate for regularization, by default 0.3
        """
        super(CNNModel, self).__init__()

        # Store parameters
        self.window_size = window_size
        self.n_neurons = n_neurons
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

        # Calculate padding to maintain input size
        padding = kernel_size // 2

        # First convolutional layer
        self.conv1 = nn.Conv1d(n_neurons, n_filters[0], kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(n_filters[0])

        # Second convolutional layer
        self.conv2 = nn.Conv1d(n_filters[0], n_filters[1], kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(n_filters[1])

        # Third convolutional layer
        self.conv3 = nn.Conv1d(n_filters[1], n_filters[2], kernel_size=kernel_size, padding=padding)
        self.bn3 = nn.BatchNorm1d(n_filters[2])

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Calculate output size from convolutional layers
        conv_output_size = window_size * n_filters[2]

        # Fully connected layer for classification
        self.fc = nn.Linear(conv_output_size, output_dim)

        logger.info(f"Initialized CNN model with {len(n_filters)} convolutional layers")

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch_size, window_size, n_neurons)

        Returns
        -------
        torch.Tensor
            Output tensor, shape (batch_size, output_dim)
        """
        # Transpose input to format expected by Conv1d: (batch_size, channels, seq_len)
        # In our case, neurons are treated as channels and window steps as sequence length
        x = x.permute(0, 2, 1)  # Shape: (batch_size, n_neurons, window_size)

        # First convolutional block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # Second convolutional block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        # Third convolutional block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        # Apply dropout for regularization
        x = self.dropout(x)

        # Flatten the output of the convolutional layers
        x = x.view(x.size(0), -1)

        # Fully connected layer for classification
        x = self.fc(x)

        return x

    def get_feature_importance(self) -> np.ndarray:
        """
        Estimate feature importance using first layer filter weights.

        Returns
        -------
        np.ndarray
            Feature importance scores, shape (window_size, n_neurons)
        """
        # Get the weights of the first convolutional layer
        # Shape: (n_filters[0], n_neurons, kernel_size)
        weights = self.conv1.weight.data.cpu().numpy()

        # Calculate feature importance as the sum of absolute weights across filters and kernel positions
        feature_importance = np.abs(weights).sum(axis=0).sum(axis=1)  # Shape: (n_neurons,)

        # Normalize feature importance
        feature_importance = feature_importance / feature_importance.sum()

        # Expand to match window_size dimension
        # This is an approximation since CNN importance is inherently distributed across the window
        feature_importance = np.tile(feature_importance, (self.window_size, 1))

        return feature_importance


class CNNWrapper:
    """
    Simplified wrapper for the CNN model with training and inference functionality.

    This wrapper provides a sklearn-like interface for the CNN model with reduced complexity.
    """

    def __init__(self,
                 window_size: Optional[int] = None,
                 n_neurons: Optional[int] = None,
                 n_filters: List[int] = [64, 128, 256],
                 kernel_size: int = 3,
                 output_dim: int = 2,
                 dropout_rate: float = 0.3,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5,
                 batch_size: int = 32,
                 num_epochs: int = 100,
                 patience: int = 15,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 random_state: int = 42):
        """
        Initialize the CNN wrapper with simplified architecture.

        Parameters
        ----------
        window_size : Optional[int], optional
            Size of the sliding window (time dimension), by default None
        n_neurons : Optional[int], optional
            Number of neurons (feature dimension), by default None
        n_filters : List[int], optional
            Number of filters in each convolutional layer, by default [64, 128, 256]
        kernel_size : int, optional
            Size of the convolutional kernel, by default 3
        output_dim : int, optional
            Output dimension (number of classes), by default 2
        dropout_rate : float, optional
            Dropout rate for regularization, by default 0.3
        learning_rate : float, optional
            Learning rate for the optimizer, by default 0.001
        weight_decay : float, optional
            Weight decay for the optimizer, by default 1e-5
        batch_size : int, optional
            Batch size for training, by default 32
        num_epochs : int, optional
            Maximum number of epochs, by default 100
        patience : int, optional
            Patience for early stopping, by default 15
        device : str, optional
            Device for training ('cuda' or 'cpu'), by default 'cuda' if available
        random_state : int, optional
            Random seed for reproducibility, by default 42
        """
        # Set random seed for reproducibility
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)

        # Store parameters
        self.window_size = window_size
        self.n_neurons = n_neurons
        self.n_filters = n_filters
        self.kernel_size = kernel_size
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

        logger.info(f"Initialized CNN wrapper (device={device})")

    def _prepare_data(self, X, y=None):
        """
        Prepare the data for the model.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray
            Input features
        y : torch.Tensor or np.ndarray, optional
            Target labels

        Returns
        -------
        Tuple[torch.Tensor, Optional[torch.Tensor]]
            Prepared X and y (if provided)
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
        Train the CNN model with simplified training loop.

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
        logger.info("Training CNN model")

        # Infer dimensions if not provided
        if self.window_size is None or self.n_neurons is None:
            if X_train.ndim == 3:
                self.window_size = X_train.shape[1]
                self.n_neurons = X_train.shape[2]
            else:
                raise ValueError("X_train must be 3D (samples, window_size, n_neurons)")

            logger.info(f"Inferred dimensions: window_size={self.window_size}, n_neurons={self.n_neurons}")

        # Initialize model
        self.model = CNNModel(
            window_size=self.window_size,
            n_neurons=self.n_neurons,
            n_filters=self.n_filters,
            kernel_size=self.kernel_size,
            output_dim=self.output_dim,
            dropout_rate=self.dropout_rate
        ).to(self.device)

        # Initialize optimizer with AdamW (includes weight decay)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Initialize simple learning rate scheduler (ReduceLROnPlateau)
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

                # Calculate loss
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

                # Update learning rate based on validation loss
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

        logger.info("CNN model training complete")

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

    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance from the model.

        Returns
        -------
        np.ndarray
            Feature importance scores, shape (window_size, n_neurons)
        """
        # Ensure model is initialized
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        return self.model.get_feature_importance()

