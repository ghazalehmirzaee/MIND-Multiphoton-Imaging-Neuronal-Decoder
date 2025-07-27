"""
CNN model for calcium imaging data.
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
    Optimized CNN for calcium imaging neural decoding.
    """

    def __init__(self,
                 window_size: int,
                 n_neurons: int,
                 n_filters: List[int] = [64, 128, 256],
                 kernel_size: int = 3,
                 output_dim: int = 2,
                 dropout_rate: float = 0.5):
        """
        Initialize the optimized CNN model.
        """
        super(CNNModel, self).__init__()

        self.window_size = window_size
        self.n_neurons = n_neurons
        padding = kernel_size // 2

        # convolutional layers with batch normalization
        self.conv1 = nn.Conv1d(n_neurons, n_filters[0], kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(n_filters[0])

        self.conv2 = nn.Conv1d(n_filters[0], n_filters[1], kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(n_filters[1])

        self.conv3 = nn.Conv1d(n_filters[1], n_filters[2], kernel_size, padding=padding)
        self.bn3 = nn.BatchNorm1d(n_filters[2])

        # Global pooling for spatial invariance
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(n_filters[2], output_dim)

        # Initialize weights for better gradient flow
        self._initialize_weights()

        logger.info(f"Initialized optimized CNN with {n_neurons} neurons")

    def _initialize_weights(self):
        """Initialize model weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass with residual connections.
        """
        # Reshape for 1D convolution: (batch, n_neurons, window_size)
        x = x.permute(0, 2, 1)

        # First convolutional block with residual connection
        identity = x
        x = F.relu(self.bn1(self.conv1(x)))

        # Second convolutional block
        x = F.relu(self.bn2(self.conv2(x)))

        # Third convolutional block
        x = F.relu(self.bn3(self.conv3(x)))

        # Global pooling
        x = self.global_pool(x).squeeze(-1)

        # Classification with dropout
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def get_feature_importance(self, window_size: int = None, n_neurons: int = None) -> np.ndarray:
        """
        Get feature importance matrix based on weight magnitudes.

        This method analyzes the trained weights to determine which neurons and
        time points are most important for classification.
        """
        # Use instance values 
        if window_size is None:
            window_size = self.window_size
        if n_neurons is None:
            n_neurons = self.n_neurons

        # Get weights from first convolutional layer
        # Shape: (n_filters[0], n_neurons, kernel_size)
        weights = self.conv1.weight.data.abs().cpu().numpy()

        # Average across filters and kernel dimension
        neuron_importance = weights.mean(axis=(0, 2))

        # Create importance matrix with same value for each time step
        importance_matrix = np.tile(neuron_importance, (window_size, 1))

        # Normalize
        if importance_matrix.sum() > 0:
            importance_matrix = importance_matrix / importance_matrix.sum()

        return importance_matrix


class CNNWrapper:
    """
    Wrapper for the CNN model.

    This wrapper handles data preparation, training, evaluation, and
    feature importance extraction for the CNN model.
    """

    def __init__(self,
                 window_size: Optional[int] = None,
                 n_neurons: Optional[int] = None,
                 n_filters: List[int] = [64, 128, 256],
                 kernel_size: int = 3,
                 output_dim: int = 2,
                 dropout_rate: float = 0.5,
                 learning_rate: float = 0.0005,
                 weight_decay: float = 1e-4,
                 batch_size: int = 32,
                 num_epochs: int = 100,
                 patience: int = 10,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 random_state: int = 42):
        """
        Initialize CNN wrapper.
        """
        # Set random seed for reproducibility
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
            torch.cuda.manual_seed_all(random_state)

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

        # Initialize model, optimizer, and scheduler
        self.model = None
        self.optimizer = None
        self.scheduler = None

        logger.info(f"CNN wrapper initialized (device={device})")

    def _prepare_data(self, X, y=None):
        """
        Prepare data for training or inference.
        """
        # Convert numpy arrays to tensors if needed
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
        if y is not None and isinstance(y, np.ndarray):
            y = torch.LongTensor(y)

        # Move to device
        X = X.to(self.device)
        if y is not None:
            y = y.to(self.device)

        return X, y

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the CNN model.
        """
        logger.info("Training CNN model")

        # Infer dimensions if not provided
        if self.window_size is None or self.n_neurons is None:
            if X_train.ndim == 3:
                self.window_size = X_train.shape[1]
                self.n_neurons = X_train.shape[2]
            else:
                raise ValueError("Cannot infer dimensions from X_train. Please provide window_size and n_neurons.")

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

        # Initialize optimizer with weight decay
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

        # Prepare data
        X_train, y_train = self._prepare_data(X_train, y_train)
        if X_val is not None and y_val is not None:
            X_val, y_val = self._prepare_data(X_val, y_val)

        # Calculate class weights for imbalanced data
        if hasattr(y_train, 'numpy'):
            y_np = y_train.cpu().numpy()
        else:
            y_np = y_train

        classes, counts = np.unique(y_np, return_counts=True)
        class_weights = 1.0 / counts
        class_weights = class_weights / class_weights.sum() * len(classes)
        class_weights = torch.FloatTensor(class_weights).to(self.device)

        # Loss function with class weights
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        if X_val is not None and y_val is not None:
            val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False
            )
            has_validation = True
        else:
            has_validation = False

        # Training loop with early stopping
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
                loss = criterion(outputs, batch_y)

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
                        loss = criterion(outputs, batch_y)

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
                    best_model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
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
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})
            logger.info("Loaded best model based on validation loss")

        logger.info("CNN model training complete")

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

    def get_feature_importance(self, window_size=None, n_neurons=None) -> np.ndarray:
        """
        Get feature importance from the model.
        """
        # Ensure model is initialized
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        return self.model.get_feature_importance(window_size, n_neurons)

    def get_top_contributing_neurons(self, n_top=100) -> np.ndarray:
        """
        Get indices of top contributing neurons.
        """
        # Ensure model is initialized
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Get feature importance
        importance_matrix = self.get_feature_importance()

        # Average across time dimension
        neuron_importance = importance_matrix.mean(axis=0)

        # Get top indices
        top_indices = np.argsort(neuron_importance)[::-1][:n_top]

        return top_indices
