"""
CNN model optimized for superior performance on deconvolved signals.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)


class CNNModel(nn.Module):
    """1D CNN for temporal pattern detection in neural activity."""

    def __init__(self, window_size: int, n_neurons: int, **kwargs):
        super().__init__()
        self.window_size = window_size
        self.n_neurons = n_neurons

        # Extract parameters
        n_filters = kwargs.get('n_filters', [64, 128, 256])
        kernel_size = kwargs.get('kernel_size', 3)
        dropout_rate = kwargs.get('dropout_rate', 0.2)

        # Convolutional layers
        self.conv1 = nn.Conv1d(n_neurons, n_filters[0], kernel_size, padding=1)
        self.bn1 = nn.BatchNorm1d(n_filters[0])

        self.conv2 = nn.Conv1d(n_filters[0], n_filters[1], kernel_size, padding=1)
        self.bn2 = nn.BatchNorm1d(n_filters[1])

        self.conv3 = nn.Conv1d(n_filters[1], n_filters[2], kernel_size, padding=1)
        self.bn3 = nn.BatchNorm1d(n_filters[2])

        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(n_filters[2], 2)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming initialization for better gradient flow."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass with residual connections."""
        # Reshape: (batch, window_size, n_neurons) -> (batch, n_neurons, window_size)
        x = x.permute(0, 2, 1)

        # Convolutional blocks
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Global pooling and classification
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def get_feature_importance(self) -> np.ndarray:
        """Extract feature importance from first conv layer."""
        weights = self.conv1.weight.data.abs().cpu().numpy()
        importance = weights.mean(axis=(0, 2))  # Average over filters and kernel
        return np.tile(importance, (self.window_size, 1))


class CNNWrapper:
    """Wrapper providing sklearn-like interface for CNN."""

    def __init__(self, window_size: int, n_neurons: int, device='cuda', **kwargs):
        self.window_size = window_size
        self.n_neurons = n_neurons
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model_params = kwargs
        self.model = None

        # Set random seed
        torch.manual_seed(kwargs.get('random_state', 42))

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train the CNN model."""
        # Initialize model
        self.model = CNNModel(self.window_size, self.n_neurons, **self.model_params).to(self.device)

        # Prepare data
        X_train = self._to_tensor(X_train)
        y_train = self._to_tensor(y_train, dtype=torch.long)

        # Training parameters
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.model_params.get('learning_rate', 0.0005),
            weight_decay=1e-4
        )

        # Class weights for imbalanced data
        class_counts = torch.bincount(y_train)
        class_weights = 1.0 / class_counts.float()
        class_weights = class_weights / class_weights.sum() * 2
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))

        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        # Training loop
        self.model.train()
        num_epochs = self.model_params.get('num_epochs', 100)

        for epoch in range(num_epochs):
            total_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}: Loss = {total_loss/len(loader):.4f}")

        return self

    def predict(self, X):
        """Make predictions."""
        self.model.eval()
        X = self._to_tensor(X)

        with torch.no_grad():
            outputs = self.model(X)
            _, predicted = torch.max(outputs, 1)

        return predicted.cpu().numpy()

    def predict_proba(self, X):
        """Get prediction probabilities."""
        self.model.eval()
        X = self._to_tensor(X)

        with torch.no_grad():
            outputs = self.model(X)
            probs = F.softmax(outputs, dim=1)

        return probs.cpu().numpy()

    def get_feature_importance(self, *args) -> np.ndarray:
        """Get feature importance."""
        return self.model.get_feature_importance()

    def _to_tensor(self, data, dtype=torch.float32):
        """Convert data to tensor and move to device."""
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=dtype)
        elif hasattr(data, 'numpy'):
            data = data.clone().detach()
        return data.to(self.device)

    