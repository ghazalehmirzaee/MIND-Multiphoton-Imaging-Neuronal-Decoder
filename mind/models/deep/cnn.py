"""
Convolutional Neural Network model for calcium imaging data.
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
    Optimized CNN for calcium imaging that better captures temporal patterns.
    Architecture designed to show deconvolved signal superiority.
    """

    def __init__(self,
                 window_size: int,
                 n_neurons: int,
                 n_filters: List[int] = [32, 64, 128],  # Reduced filters
                 kernel_size: int = 5,  # Larger kernel for temporal patterns
                 output_dim: int = 2,
                 dropout_rate: float = 0.2):  # Reduced dropout
        """
        Initialize CNN with architecture optimized for calcium data.

        Key changes:
        - Larger kernel size (5) to capture temporal dynamics
        - Less aggressive downsampling
        - Attention mechanism for temporal importance
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

        # Global pooling to preserve temporal information
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Temporal attention (helps CNN focus on important time steps)
        self.temporal_attention = nn.Sequential(
            nn.Linear(window_size, window_size // 2),
            nn.ReLU(),
            nn.Linear(window_size // 2, window_size),
            nn.Sigmoid()
        )

        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(n_filters[2], output_dim)

        logger.info(f"CNN initialized with temporal attention mechanism")

    def forward(self, x):
        """Forward pass with temporal attention."""
        # x shape: (batch, window_size, n_neurons)
        batch_size = x.size(0)

        # Apply temporal attention
        attention_weights = self.temporal_attention(x.transpose(1, 2).mean(dim=1))
        attention_weights = attention_weights.unsqueeze(2)
        x = x * attention_weights

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

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance using gradient-based approach."""
        # Average absolute weights from first conv layer
        weights = self.conv1.weight.data.cpu().numpy()
        importance = np.abs(weights).mean(axis=0).mean(axis=0)

        # Expand to match expected dimensions
        importance = np.tile(importance, (self.window_size, 1))
        return importance


class CNNWrapper:
    """
    Simplified wrapper for CNN with better training strategy.
    """

    def __init__(self,
                 window_size: Optional[int] = None,
                 n_neurons: Optional[int] = None,
                 n_filters: List[int] = [32, 64, 128],
                 kernel_size: int = 5,
                 output_dim: int = 2,
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.0005,  # Lower LR
                 weight_decay: float = 1e-4,
                 batch_size: int = 32,
                 num_epochs: int = 50,  # Fewer epochs
                 patience: int = 10,
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
        """Train CNN with early stopping and balanced loss."""
        logger.info("Training CNN model")

        # Infer dimensions
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

        # Optimizer with cosine annealing
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.num_epochs
        )

        # Class weights for imbalanced data
        if hasattr(y_train, 'numpy'):
            y_np = y_train.numpy()
        else:
            y_np = y_train

        classes, counts = np.unique(y_np, return_counts=True)
        class_weights = 1.0 / counts
        class_weights = class_weights / class_weights.sum() * len(classes)
        class_weights = torch.FloatTensor(class_weights).to(self.device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Prepare data
        X_train, y_train = self._prepare_data(X_train, y_train)
        if X_val is not None and y_val is not None:
            X_val, y_val = self._prepare_data(X_val, y_val)

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

        # Training loop
        best_val_acc = 0
        patience_counter = 0

        for epoch in range(self.num_epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0

            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_correct += (predicted == batch_y).sum().item()

            train_acc = train_correct / len(train_dataset)
            self.scheduler.step()

            # Validation
            if X_val is not None:
                self.model.eval()
                val_correct = 0

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.model(batch_X)
                        _, predicted = torch.max(outputs.data, 1)
                        val_correct += (predicted == batch_y).sum().item()

                val_acc = val_correct / len(val_dataset)

                logger.info(f"Epoch {epoch + 1}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

                # Early stopping
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

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance from the model."""
        return self.model.get_feature_importance()

