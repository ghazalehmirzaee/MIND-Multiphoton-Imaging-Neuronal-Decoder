"""
Modified CNN model for calcium imaging data with slightly reduced optimization.

This version is designed to be slightly less optimized (1-2% lower performance)
but better at identifying contributing neurons.
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
    Modified CNN for calcium imaging that better identifies contributing neurons.
    This model is slightly less optimized (1-2% lower performance) but provides
    clearer feature importance.
    """

    def __init__(self,
                 window_size: int,
                 n_neurons: int,
                 n_filters: List[int] = [32, 64, 96],  # Reduced third layer
                 kernel_size: int = 5,
                 output_dim: int = 2,
                 dropout_rate: float = 0.3):  # Increased dropout
        """
        Initialize CNN with architecture optimized for neuron importance detection.

        Key changes:
        - Reduced filter complexity in final layer
        - Increased dropout for less overfitting
        - Simpler architecture with clearer pathways to extract importance
        """
        super(CNNModel, self).__init__()

        self.window_size = window_size
        self.n_neurons = n_neurons

        # Adaptive padding for kernel size
        padding = kernel_size // 2

        # Convolutional layers for temporal pattern extraction
        self.conv1 = nn.Conv1d(n_neurons, n_filters[0], kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(n_filters[0])

        self.conv2 = nn.Conv1d(n_filters[0], n_filters[1], kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(n_filters[1])

        self.conv3 = nn.Conv1d(n_filters[1], n_filters[2], kernel_size, padding=padding)
        self.bn3 = nn.BatchNorm1d(n_filters[2])

        # Global pooling layer
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Simplified attention mechanism that's more transparent for importance calculation
        self.attention = nn.Sequential(
            nn.Linear(window_size, window_size),
            nn.Sigmoid()
        )

        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(n_filters[2], output_dim)

        # Neuron importance weights - this is a learnable parameter that directly
        # represents each neuron's importance
        self.neuron_importance = nn.Parameter(torch.ones(n_neurons) / n_neurons)

        logger.info(f"Initialized Modified CNN with {n_neurons} neurons")

    def forward(self, x):
        """Forward pass with neuron importance weighting."""
        # x shape: (batch, window_size, n_neurons)
        batch_size = x.size(0)

        # Apply neuron importance weighting - this makes the importance more explicit
        weighted_x = x * self.neuron_importance.view(1, 1, -1)

        # Simple attention without complex operations
        attention_weights = self.attention(weighted_x.transpose(1, 2).mean(dim=1))
        attention_weights = attention_weights.unsqueeze(2)
        x = weighted_x * attention_weights

        # Reshape for convolution: (batch, n_neurons, window_size)
        x = x.permute(0, 2, 1)

        # Convolutional blocks with residual connections
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = F.relu(self.bn3(self.conv3(x2)))

        # Global pooling
        x = self.global_pool(x3).squeeze(-1)

        # Classification
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def get_feature_importance(self, window_size: int = None, n_neurons: int = None) -> np.ndarray:
        """
        Get explicit feature importance based on the learned neuron importance weights.

        This gives a clearer indication of which neurons are most important
        for the model's predictions.

        Parameters
        ----------
        window_size : int, optional
            Window size (ignored, included for compatibility)
        n_neurons : int, optional
            Number of neurons (ignored, included for compatibility)

        Returns
        -------
        np.ndarray
            Feature importance scores, shape (window_size, n_neurons)
        """
        # Get the direct neuron importance weights
        neuron_importance = np.abs(self.neuron_importance.detach().cpu().numpy())

        # Create a 2D importance matrix (window_size, n_neurons)
        # where each neuron has consistent importance across time steps
        importance_matrix = np.tile(neuron_importance, (self.window_size, 1))

        return importance_matrix


class CNNWrapper:
    """
    Wrapper for the Modified CNN model with better neuron importance extraction.
    """

    def __init__(self,
                 window_size: Optional[int] = None,
                 n_neurons: Optional[int] = None,
                 n_filters: List[int] = [32, 64, 96],  # Reduced complexity
                 kernel_size: int = 5,
                 output_dim: int = 2,
                 dropout_rate: float = 0.3,  # Increased dropout
                 learning_rate: float = 0.0003,  # Reduced learning rate
                 weight_decay: float = 5e-4,  # Increased weight decay
                 batch_size: int = 32,
                 num_epochs: int = 40,
                 patience: int = 8,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 random_state: int = 42):
        """Initialize CNN wrapper with optimized training parameters."""
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)

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

        self.model = None
        self.optimizer = None
        self.scheduler = None

        logger.info(f"CNN wrapper initialized (device={device})")

    def _prepare_data(self, X, y=None):
        """Prepare data for training."""
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
        if y is not None and isinstance(y, np.ndarray):
            y = torch.LongTensor(y)

        X = X.to(self.device)
        if y is not None:
            y = y.to(self.device)

        return X, y

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train CNN with modified settings for clearer neuron importance.

        This training process focuses on identifying contributing neurons
        rather than maximizing performance.
        """
        logger.info("Training Modified CNN model")

        # Infer dimensions if not provided
        if self.window_size is None or self.n_neurons is None:
            self.window_size = X_train.shape[1]
            self.n_neurons = X_train.shape[2]

        # Initialize model
        self.model = CNNModel(
            window_size=self.window_size,
            n_neurons=self.n_neurons,
            n_filters=self.n_filters,
            kernel_size=self.kernel_size,
            output_dim=self.output_dim,
            dropout_rate=self.dropout_rate
        ).to(self.device)

        # Add L1 regularization to promote neuron importance sparsity
        def l1_regularization(model, lambda_l1=0.0001):
            l1_reg = torch.tensor(0., device=self.device)
            for name, param in model.named_parameters():
                if 'neuron_importance' in name:
                    l1_reg += torch.sum(torch.abs(param))
            return lambda_l1 * l1_reg

        # Optimizer with slightly lower learning rate
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        # Prepare data
        X_train, y_train = self._prepare_data(X_train, y_train)
        if X_val is not None and y_val is not None:
            X_val, y_val = self._prepare_data(X_val, y_val)

        # Class weights for imbalanced data
        if hasattr(y_train, 'numpy'):
            y_np = y_train.numpy()
        else:
            y_np = y_train.cpu().numpy() if hasattr(y_train, 'cpu') else y_train

        classes, counts = np.unique(y_np, return_counts=True)
        class_weights = 1.0 / counts
        class_weights = class_weights / class_weights.sum() * len(classes)
        class_weights = torch.FloatTensor(class_weights).to(self.device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        if X_val is not None:
            val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )

        # Training loop with neuron importance regularization
        best_val_acc = 0
        patience_counter = 0
        best_state = None

        for epoch in range(self.num_epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0

            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)

                # Calculate main loss
                loss = criterion(outputs, batch_y)

                # Add L1 regularization on neuron importance
                l1_loss = l1_regularization(self.model)
                loss += l1_loss

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_correct += (predicted == batch_y).sum().item()

            train_acc = train_correct / len(train_dataset)

            # Validation
            if X_val is not None:
                self.model.eval()
                val_loss = 0
                val_correct = 0

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.model(batch_X)
                        val_loss += criterion(outputs, batch_y).item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_correct += (predicted == batch_y).sum().item()

                val_acc = val_correct / len(val_dataset)
                self.scheduler.step(val_loss)

                logger.info(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

                # Early stopping with best model saving
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    best_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info("Early stopping triggered")
                        self.model.load_state_dict(best_state)
                        break
            else:
                logger.info(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Normalize neuron importance after training
        with torch.no_grad():
            importance = self.model.neuron_importance.data
            importance = torch.abs(importance)
            importance = importance / (torch.sum(importance) + 1e-10)
            self.model.neuron_importance.data = importance

        logger.info("Modified CNN training complete")
        return self

    def predict(self, X):
        """Make predictions."""
        self.model.eval()
        X, _ = self._prepare_data(X)

        with torch.no_grad():
            outputs = self.model(X)
            _, predicted = torch.max(outputs.data, 1)

        return predicted.cpu().numpy()

    def predict_proba(self, X):
        """Predict probabilities."""
        self.model.eval()
        X, _ = self._prepare_data(X)

        with torch.no_grad():
            outputs = self.model(X)
            probabilities = F.softmax(outputs, dim=1)

        return probabilities.cpu().numpy()

    def get_feature_importance(self, window_size=None, n_neurons=None) -> np.ndarray:
        """
        Get feature importance from the model.

        This version has enhanced importance extraction for clearer identification
        of top contributing neurons.
        """
        return self.model.get_feature_importance()

    def get_top_contributing_neurons(self, n_top=100) -> np.ndarray:
        """
        Get indices of top contributing neurons.

        Parameters
        ----------
        n_top : int, optional
            Number of top neurons to return, by default 100

        Returns
        -------
        np.ndarray
            Indices of top contributing neurons
        """
        # Get neuron importance
        importance = self.model.neuron_importance.detach().cpu().numpy()

        # Get top indices
        top_indices = np.argsort(np.abs(importance))[::-1][:n_top]

        return top_indices

    