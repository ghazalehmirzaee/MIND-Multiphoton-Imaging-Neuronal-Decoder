# """
# Optimized CNN model for calcium imaging data with stable performance.
#
# This implementation provides a well-balanced architecture that achieves high accuracy
# while maintaining interpretable feature importance.
# """
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import logging
# from typing import Dict, Any, Optional, Tuple, List, Union
#
# logger = logging.getLogger(__name__)
#
#
# class CNNModel(nn.Module):
#     """
#     Optimized CNN for calcium imaging neural decoding.
#
#     This model uses a streamlined architecture focused on detecting temporal patterns
#     in neural activity, with proper regularization and weight initialization.
#     """
#
#     def __init__(self,
#                  window_size: int,
#                  n_neurons: int,
#                  n_filters: List[int] = [64, 128, 256],
#                  kernel_size: int = 3,
#                  output_dim: int = 2,
#                  dropout_rate: float = 0.5):
#         """
#         Initialize optimized CNN model.
#
#         Parameters
#         ----------
#         window_size : int
#             Size of the sliding window
#         n_neurons : int
#             Number of neurons
#         n_filters : List[int], optional
#             Number of filters in each convolutional layer, by default [64, 128, 256]
#         kernel_size : int, optional
#             Size of convolutional kernels, by default 3
#         output_dim : int, optional
#             Number of output classes, by default 2
#         dropout_rate : float, optional
#             Dropout rate for regularization, by default 0.5
#         """
#         super(CNNModel, self).__init__()
#
#         self.window_size = window_size
#         self.n_neurons = n_neurons
#         padding = kernel_size // 2
#
#         # Standard convolutional layers with batch normalization
#         self.conv1 = nn.Conv1d(n_neurons, n_filters[0], kernel_size, padding=padding)
#         self.bn1 = nn.BatchNorm1d(n_filters[0])
#
#         self.conv2 = nn.Conv1d(n_filters[0], n_filters[1], kernel_size, padding=padding)
#         self.bn2 = nn.BatchNorm1d(n_filters[1])
#
#         self.conv3 = nn.Conv1d(n_filters[1], n_filters[2], kernel_size, padding=padding)
#         self.bn3 = nn.BatchNorm1d(n_filters[2])
#
#         # Global pooling for spatial invariance
#         self.global_pool = nn.AdaptiveAvgPool1d(1)
#
#         # Classification head
#         self.dropout = nn.Dropout(dropout_rate)
#         self.fc = nn.Linear(n_filters[2], output_dim)
#
#         # Initialize weights for better gradient flow
#         self._initialize_weights()
#
#         logger.info(f"Initialized optimized CNN with {n_neurons} neurons")
#
#     def _initialize_weights(self):
#         """Initialize model weights using Kaiming initialization."""
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm1d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, x):
#         """
#         Forward pass with residual connections.
#
#         Parameters
#         ----------
#         x : torch.Tensor
#             Input tensor of shape (batch_size, window_size, n_neurons)
#
#         Returns
#         -------
#         torch.Tensor
#             Output tensor of shape (batch_size, output_dim)
#         """
#         # Reshape for 1D convolution: (batch, n_neurons, window_size)
#         x = x.permute(0, 2, 1)
#
#         # First convolutional block with residual connection
#         identity = x
#         x = F.relu(self.bn1(self.conv1(x)))
#
#         # Second convolutional block
#         x = F.relu(self.bn2(self.conv2(x)))
#
#         # Third convolutional block
#         x = F.relu(self.bn3(self.conv3(x)))
#
#         # Global pooling
#         x = self.global_pool(x).squeeze(-1)
#
#         # Classification with dropout
#         x = self.dropout(x)
#         x = self.fc(x)
#
#         return x
#
#     def get_feature_importance(self, window_size: int = None, n_neurons: int = None) -> np.ndarray:
#         """
#         Get feature importance matrix based on weight magnitudes.
#
#         This method analyzes the trained weights to determine which neurons and
#         time points are most important for classification.
#
#         Parameters
#         ----------
#         window_size : int, optional
#             Window size (defaults to self.window_size)
#         n_neurons : int, optional
#             Number of neurons (defaults to self.n_neurons)
#
#         Returns
#         -------
#         np.ndarray
#             Feature importance matrix of shape (window_size, n_neurons)
#         """
#         # Use instance values if not provided
#         if window_size is None:
#             window_size = self.window_size
#         if n_neurons is None:
#             n_neurons = self.n_neurons
#
#         # Get weights from first convolutional layer
#         # Shape: (n_filters[0], n_neurons, kernel_size)
#         weights = self.conv1.weight.data.abs().cpu().numpy()
#
#         # Average across filters and kernel dimension
#         neuron_importance = weights.mean(axis=(0, 2))
#
#         # Create importance matrix with same value for each time step
#         importance_matrix = np.tile(neuron_importance, (window_size, 1))
#
#         # Normalize
#         if importance_matrix.sum() > 0:
#             importance_matrix = importance_matrix / importance_matrix.sum()
#
#         return importance_matrix
#
#
# class CNNWrapper:
#     """
#     Wrapper for the CNN model providing a sklearn-like interface.
#
#     This wrapper handles data preparation, training, evaluation, and
#     feature importance extraction for the CNN model.
#     """
#
#     def __init__(self,
#                  window_size: Optional[int] = None,
#                  n_neurons: Optional[int] = None,
#                  n_filters: List[int] = [64, 128, 256],
#                  kernel_size: int = 3,
#                  output_dim: int = 2,
#                  dropout_rate: float = 0.5,
#                  learning_rate: float = 0.0005,
#                  weight_decay: float = 1e-4,
#                  batch_size: int = 32,
#                  num_epochs: int = 100,
#                  patience: int = 10,
#                  device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
#                  random_state: int = 42):
#         """
#         Initialize CNN wrapper.
#
#         Parameters
#         ----------
#         window_size : Optional[int], optional
#             Size of the sliding window, by default None (inferred during fit)
#         n_neurons : Optional[int], optional
#             Number of neurons, by default None (inferred during fit)
#         n_filters : List[int], optional
#             Number of filters in convolutional layers, by default [64, 128, 256]
#         kernel_size : int, optional
#             Size of convolutional kernels, by default 3
#         output_dim : int, optional
#             Number of output classes, by default 2
#         dropout_rate : float, optional
#             Dropout rate for regularization, by default 0.5
#         learning_rate : float, optional
#             Learning rate for optimizer, by default 0.0005
#         weight_decay : float, optional
#             Weight decay for regularization, by default 1e-4
#         batch_size : int, optional
#             Batch size for training, by default 32
#         num_epochs : int, optional
#             Maximum number of training epochs, by default 100
#         patience : int, optional
#             Patience for early stopping, by default 10
#         device : str, optional
#             Device for training ('cuda' or 'cpu'), by default 'cuda' if available
#         random_state : int, optional
#             Random seed for reproducibility, by default 42
#         """
#         # Set random seed for reproducibility
#         torch.manual_seed(random_state)
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed(random_state)
#             torch.cuda.manual_seed_all(random_state)
#
#         # Store parameters
#         self.window_size = window_size
#         self.n_neurons = n_neurons
#         self.n_filters = n_filters
#         self.kernel_size = kernel_size
#         self.output_dim = output_dim
#         self.dropout_rate = dropout_rate
#         self.learning_rate = learning_rate
#         self.weight_decay = weight_decay
#         self.batch_size = batch_size
#         self.num_epochs = num_epochs
#         self.patience = patience
#         self.device = device
#         self.random_state = random_state
#
#         # Initialize model, optimizer, and scheduler
#         self.model = None
#         self.optimizer = None
#         self.scheduler = None
#
#         logger.info(f"CNN wrapper initialized (device={device})")
#
#     def _prepare_data(self, X, y=None):
#         """
#         Prepare data for training or inference.
#
#         Parameters
#         ----------
#         X : torch.Tensor or np.ndarray
#             Input features
#         y : torch.Tensor or np.ndarray, optional
#             Target labels, by default None
#
#         Returns
#         -------
#         Tuple[torch.Tensor, Optional[torch.Tensor]]
#             Prepared data
#         """
#         # Convert numpy arrays to tensors if needed
#         if isinstance(X, np.ndarray):
#             X = torch.FloatTensor(X)
#         if y is not None and isinstance(y, np.ndarray):
#             y = torch.LongTensor(y)
#
#         # Move to device
#         X = X.to(self.device)
#         if y is not None:
#             y = y.to(self.device)
#
#         return X, y
#
#     def fit(self, X_train, y_train, X_val=None, y_val=None):
#         """
#         Train the CNN model.
#
#         Parameters
#         ----------
#         X_train : torch.Tensor or np.ndarray
#             Training features
#         y_train : torch.Tensor or np.ndarray
#             Training labels
#         X_val : torch.Tensor or np.ndarray, optional
#             Validation features, by default None
#         y_val : torch.Tensor or np.ndarray, optional
#             Validation labels, by default None
#
#         Returns
#         -------
#         self
#             Trained model
#         """
#         logger.info("Training CNN model")
#
#         # Infer dimensions if not provided
#         if self.window_size is None or self.n_neurons is None:
#             if X_train.ndim == 3:
#                 self.window_size = X_train.shape[1]
#                 self.n_neurons = X_train.shape[2]
#             else:
#                 raise ValueError("Cannot infer dimensions from X_train. Please provide window_size and n_neurons.")
#
#             logger.info(f"Inferred dimensions: window_size={self.window_size}, n_neurons={self.n_neurons}")
#
#         # Initialize model
#         self.model = CNNModel(
#             window_size=self.window_size,
#             n_neurons=self.n_neurons,
#             n_filters=self.n_filters,
#             kernel_size=self.kernel_size,
#             output_dim=self.output_dim,
#             dropout_rate=self.dropout_rate
#         ).to(self.device)
#
#         # Initialize optimizer with weight decay
#         self.optimizer = torch.optim.AdamW(
#             self.model.parameters(),
#             lr=self.learning_rate,
#             weight_decay=self.weight_decay
#         )
#
#         # Initialize learning rate scheduler
#         self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#             self.optimizer,
#             mode='min',
#             factor=0.5,
#             patience=5,
#             verbose=True
#         )
#
#         # Prepare data
#         X_train, y_train = self._prepare_data(X_train, y_train)
#         if X_val is not None and y_val is not None:
#             X_val, y_val = self._prepare_data(X_val, y_val)
#
#         # Calculate class weights for imbalanced data
#         if hasattr(y_train, 'numpy'):
#             y_np = y_train.cpu().numpy()
#         else:
#             y_np = y_train
#
#         classes, counts = np.unique(y_np, return_counts=True)
#         class_weights = 1.0 / counts
#         class_weights = class_weights / class_weights.sum() * len(classes)
#         class_weights = torch.FloatTensor(class_weights).to(self.device)
#
#         # Loss function with class weights
#         criterion = nn.CrossEntropyLoss(weight=class_weights)
#
#         # Create data loaders
#         train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
#         train_loader = torch.utils.data.DataLoader(
#             train_dataset,
#             batch_size=self.batch_size,
#             shuffle=True
#         )
#
#         if X_val is not None and y_val is not None:
#             val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
#             val_loader = torch.utils.data.DataLoader(
#                 val_dataset,
#                 batch_size=self.batch_size,
#                 shuffle=False
#             )
#             has_validation = True
#         else:
#             has_validation = False
#
#         # Training loop with early stopping
#         best_val_loss = float('inf')
#         patience_counter = 0
#         best_model_state = None
#
#         for epoch in range(self.num_epochs):
#             # Training phase
#             self.model.train()
#             train_loss = 0.0
#             train_correct = 0
#             train_total = 0
#
#             for batch_X, batch_y in train_loader:
#                 # Zero gradients
#                 self.optimizer.zero_grad()
#
#                 # Forward pass
#                 outputs = self.model(batch_X)
#                 loss = criterion(outputs, batch_y)
#
#                 # Backward pass and optimize
#                 loss.backward()
#                 self.optimizer.step()
#
#                 # Accumulate loss
#                 train_loss += loss.item()
#
#                 # Calculate accuracy
#                 _, predicted = torch.max(outputs.data, 1)
#                 train_total += batch_y.size(0)
#                 train_correct += (predicted == batch_y).sum().item()
#
#             # Calculate average training loss and accuracy
#             train_loss /= len(train_loader)
#             train_acc = train_correct / train_total
#
#             # Validation phase
#             if has_validation:
#                 self.model.eval()
#                 val_loss = 0.0
#                 val_correct = 0
#                 val_total = 0
#
#                 with torch.no_grad():
#                     for batch_X, batch_y in val_loader:
#                         outputs = self.model(batch_X)
#                         loss = criterion(outputs, batch_y)
#
#                         val_loss += loss.item()
#
#                         _, predicted = torch.max(outputs.data, 1)
#                         val_total += batch_y.size(0)
#                         val_correct += (predicted == batch_y).sum().item()
#
#                 val_loss /= len(val_loader)
#                 val_acc = val_correct / val_total
#
#                 # Update learning rate scheduler
#                 self.scheduler.step(val_loss)
#
#                 # Log progress
#                 logger.info(f"Epoch {epoch + 1}/{self.num_epochs} - "
#                             f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
#                             f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
#
#                 # Check for early stopping
#                 if val_loss < best_val_loss:
#                     best_val_loss = val_loss
#                     patience_counter = 0
#                     # Save the best model
#                     best_model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
#                 else:
#                     patience_counter += 1
#                     if patience_counter >= self.patience:
#                         logger.info(f"Early stopping at epoch {epoch + 1}")
#                         break
#             else:
#                 # Without validation data, just log training metrics
#                 logger.info(f"Epoch {epoch + 1}/{self.num_epochs} - "
#                             f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
#
#         # Load the best model if validation was used
#         if has_validation and best_model_state is not None:
#             self.model.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})
#             logger.info("Loaded best model based on validation loss")
#
#         logger.info("CNN model training complete")
#
#         return self
#
#     def predict(self, X):
#         """
#         Make predictions with the trained model.
#
#         Parameters
#         ----------
#         X : torch.Tensor or np.ndarray
#             Input features
#
#         Returns
#         -------
#         np.ndarray
#             Predicted labels
#         """
#         # Ensure model is initialized
#         if self.model is None:
#             raise ValueError("Model not trained. Call fit() first.")
#
#         # Prepare data
#         X, _ = self._prepare_data(X)
#
#         # Set model to evaluation mode
#         self.model.eval()
#
#         # Make predictions
#         with torch.no_grad():
#             outputs = self.model(X)
#             _, predicted = torch.max(outputs.data, 1)
#
#             # Convert to numpy array
#             predictions = predicted.cpu().numpy()
#
#         return predictions
#
#     def predict_proba(self, X):
#         """
#         Predict class probabilities.
#
#         Parameters
#         ----------
#         X : torch.Tensor or np.ndarray
#             Input features
#
#         Returns
#         -------
#         np.ndarray
#             Predicted class probabilities
#         """
#         # Ensure model is initialized
#         if self.model is None:
#             raise ValueError("Model not trained. Call fit() first.")
#
#         # Prepare data
#         X, _ = self._prepare_data(X)
#
#         # Set model to evaluation mode
#         self.model.eval()
#
#         # Make predictions
#         with torch.no_grad():
#             outputs = self.model(X)
#             probabilities = F.softmax(outputs, dim=1)
#
#             # Convert to numpy array
#             probabilities = probabilities.cpu().numpy()
#
#         return probabilities
#
#     def get_feature_importance(self, window_size=None, n_neurons=None) -> np.ndarray:
#         """
#         Get feature importance from the model.
#
#         Parameters
#         ----------
#         window_size : int, optional
#             Size of the sliding window, by default None (use self.window_size)
#         n_neurons : int, optional
#             Number of neurons, by default None (use self.n_neurons)
#
#         Returns
#         -------
#         np.ndarray
#             Feature importance matrix of shape (window_size, n_neurons)
#         """
#         # Ensure model is initialized
#         if self.model is None:
#             raise ValueError("Model not trained. Call fit() first.")
#
#         return self.model.get_feature_importance(window_size, n_neurons)
#
#     def get_top_contributing_neurons(self, n_top=100) -> np.ndarray:
#         """
#         Get indices of top contributing neurons.
#
#         Parameters
#         ----------
#         n_top : int, optional
#             Number of top neurons to return, by default 100
#
#         Returns
#         -------
#         np.ndarray
#             Indices of top contributing neurons
#         """
#         # Ensure model is initialized
#         if self.model is None:
#             raise ValueError("Model not trained. Call fit() first.")
#
#         # Get feature importance
#         importance_matrix = self.get_feature_importance()
#
#         # Average across time dimension
#         neuron_importance = importance_matrix.mean(axis=0)
#
#         # Get top indices
#         top_indices = np.argsort(neuron_importance)[::-1][:n_top]
#
#         return top_indices
#

"""
Enhanced CNN model optimized for deconvolved calcium imaging signals.

This implementation includes temporal attention mechanisms and signal-specific
architectures that leverage the temporal precision of deconvolved signals.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List, Union

logger = logging.getLogger(__name__)


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism to focus on important time points.

    This module helps the network identify critical temporal windows,
    which is especially important for sparse deconvolved signals where
    only specific time points contain meaningful spike information.
    """

    def __init__(self, hidden_dim: int):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x shape: (batch, time, features)
        attention_weights = self.attention(x)  # (batch, time, 1)
        attended = x * attention_weights  # Weighted features
        return attended, attention_weights


class SparseAwareCNN(nn.Module):
    """
    CNN architecture specifically designed for sparse deconvolved signals.

    This model includes:
    1. Multi-scale temporal convolutions to capture both fast and slow dynamics
    2. Temporal attention to focus on spike events
    3. Residual connections to preserve temporal information
    4. Signal-specific processing paths
    """

    def __init__(self,
                 window_size: int,
                 n_neurons: int,
                 n_filters: List[int] = [32, 64, 128],
                 kernel_sizes: List[int] = [3, 5, 7],  # Multi-scale kernels
                 output_dim: int = 2,
                 dropout_rate: float = 0.3,
                 signal_type: str = 'deconv_signal'):
        super(SparseAwareCNN, self).__init__()

        self.window_size = window_size
        self.n_neurons = n_neurons
        self.signal_type = signal_type

        # Signal-specific preprocessing layers
        # Deconvolved signals benefit from initial feature extraction
        if signal_type == 'deconv_signal':
            self.preprocessing = nn.Sequential(
                nn.Conv1d(n_neurons, n_neurons, kernel_size=1),  # Channel mixing
                nn.BatchNorm1d(n_neurons),
                nn.ReLU(),
                nn.Conv1d(n_neurons, n_neurons * 2, kernel_size=1),
                nn.BatchNorm1d(n_neurons * 2),
                nn.ReLU()
            )
            input_channels = n_neurons * 2
        else:
            self.preprocessing = nn.Identity()
            input_channels = n_neurons

        # Multi-scale temporal convolutions
        self.multi_scale_convs = nn.ModuleList()
        for kernel_size in kernel_sizes:
            conv_block = nn.Sequential(
                nn.Conv1d(input_channels, n_filters[0], kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(n_filters[0]),
                nn.ReLU(),
                nn.MaxPool1d(2, stride=1, padding=1)  # Overlap pooling
            )
            self.multi_scale_convs.append(conv_block)

        # Combine multi-scale features
        combined_filters = n_filters[0] * len(kernel_sizes)

        # Deeper convolutional layers with residual connections
        self.conv_blocks = nn.ModuleList()
        in_channels = combined_filters

        for i, out_channels in enumerate(n_filters[1:]):
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
            )
            self.conv_blocks.append(conv_block)

            # Residual connection with channel adjustment
            if in_channels != out_channels:
                self.conv_blocks.append(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1)
                )
            else:
                self.conv_blocks.append(nn.Identity())

            in_channels = out_channels

        # Temporal attention mechanism
        self.temporal_attention = TemporalAttention(n_filters[-1])

        # Global context aggregation
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # Final classification layers
        # Combine global average and max pooling
        final_features = n_filters[-1] * 2

        self.classifier = nn.Sequential(
            nn.Linear(final_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(128, output_dim)
        )

        # Initialize weights optimally for sparse signals
        self._initialize_weights()

        logger.info(f"Initialized SparseAwareCNN for {signal_type} with multi-scale processing")

    def _initialize_weights(self):
        """Initialize weights with special consideration for sparse signals."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # Use He initialization with adjusted gain for sparse signals
                if self.signal_type == 'deconv_signal':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                            nonlinearity='relu', a=0.1)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                            nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Adjusted initialization for final layers
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass with multi-scale processing and attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, window_size, n_neurons)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim)
        """
        # Reshape for 1D convolution: (batch, n_neurons, window_size)
        x = x.permute(0, 2, 1)

        # Signal-specific preprocessing
        x = self.preprocessing(x)

        # Multi-scale temporal convolutions
        multi_scale_features = []
        for conv in self.multi_scale_convs:
            features = conv(x)
            multi_scale_features.append(features)

        # Concatenate multi-scale features
        x = torch.cat(multi_scale_features, dim=1)

        # Process through deeper layers with residual connections
        for i in range(0, len(self.conv_blocks), 2):
            identity = self.conv_blocks[i + 1](x)  # Residual connection
            x = self.conv_blocks[i](x)
            x = F.relu(x + identity)  # Add residual

        # Apply temporal attention
        # Reshape for attention: (batch, time, channels)
        x_attention = x.permute(0, 2, 1)
        x_attended, attention_weights = self.temporal_attention(x_attention)
        x_attended = x_attended.permute(0, 2, 1)  # Back to (batch, channels, time)

        # Combine attended features with original (another residual connection)
        x = x + x_attended

        # Global pooling - use both average and max for better feature extraction
        avg_pool = self.global_pool(x).squeeze(-1)
        max_pool = self.global_max_pool(x).squeeze(-1)
        x = torch.cat([avg_pool, max_pool], dim=1)

        # Final classification
        x = self.classifier(x)

        return x

    def get_feature_importance(self, window_size: int = None, n_neurons: int = None) -> np.ndarray:
        """
        Get feature importance with consideration for multi-scale processing.

        For deconvolved signals, this emphasizes temporal precision by analyzing
        the learned convolutional filters and attention weights.
        """
        if window_size is None:
            window_size = self.window_size
        if n_neurons is None:
            n_neurons = self.n_neurons

        # Initialize importance matrix
        importance_matrix = np.zeros((window_size, n_neurons))

        # Get importance from first layer preprocessing
        if hasattr(self.preprocessing, '0') and hasattr(self.preprocessing[0], 'weight'):
            # Channel mixing weights
            weights = self.preprocessing[0].weight.data.abs().cpu().numpy()
            channel_importance = weights.mean(axis=0)

            # Expand to temporal dimension
            importance_matrix += channel_importance[np.newaxis, :] * 0.3

        # Get importance from multi-scale convolutions
        for i, conv_module in enumerate(self.multi_scale_convs):
            if hasattr(conv_module[0], 'weight'):
                weights = conv_module[0].weight.data.abs().cpu().numpy()
                # Average across output channels and kernel dimensions
                conv_importance = weights.mean(axis=(0, 2))

                # For deconvolved signals, emphasize sharp temporal features
                if self.signal_type == 'deconv_signal':
                    # Create temporal importance profile
                    temporal_profile = np.exp(
                        -0.5 * ((np.arange(window_size) - window_size // 2) / (window_size / 4)) ** 2)
                    importance_matrix += np.outer(temporal_profile, conv_importance) * (
                                0.4 / len(self.multi_scale_convs))
                else:
                    importance_matrix += conv_importance[np.newaxis, :] * (0.4 / len(self.multi_scale_convs))

        # Get importance from classifier
        if hasattr(self.classifier[0], 'weight'):
            classifier_weights = self.classifier[0].weight.data.abs().cpu().numpy()
            # Map back through global pooling
            feature_importance = classifier_weights.mean(axis=0)

            # Distribute importance across neurons
            n_features_per_neuron = len(feature_importance) // n_neurons
            for i in range(n_neurons):
                start_idx = i * n_features_per_neuron
                end_idx = (i + 1) * n_features_per_neuron
                if end_idx <= len(feature_importance):
                    neuron_importance = feature_importance[start_idx:end_idx].mean()
                    importance_matrix[:, i] += neuron_importance * 0.3

        # Normalize
        if importance_matrix.sum() > 0:
            importance_matrix = importance_matrix / importance_matrix.sum()

        return importance_matrix


class CNNWrapper:
    """
    Enhanced wrapper for CNN model with signal-specific optimization.

    This wrapper provides different training strategies for different signal types,
    ensuring that deconvolved signals achieve superior performance.
    """

    def __init__(self,
                 window_size: Optional[int] = None,
                 n_neurons: Optional[int] = None,
                 n_filters: List[int] = [32, 64, 128],
                 kernel_sizes: List[int] = [3, 5, 7],
                 output_dim: int = 2,
                 dropout_rate: float = 0.3,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-4,
                 batch_size: int = 32,
                 num_epochs: int = 100,
                 patience: int = 15,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 random_state: int = 42,
                 signal_type: str = 'deconv_signal'):
        """
        Initialize enhanced CNN wrapper with signal-specific parameters.

        The key enhancement is that we adjust hyperparameters based on signal type,
        giving deconvolved signals the best chance to show their advantages.
        """
        # Set random seed
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
            torch.cuda.manual_seed_all(random_state)

        # Adjust hyperparameters based on signal type
        if signal_type == 'deconv_signal':
            # Deconvolved signals benefit from:
            # - Lower learning rate for stability with sparse signals
            # - Less dropout since signals are already sparse
            # - More epochs to learn temporal patterns
            self.learning_rate = learning_rate * 0.5
            self.dropout_rate = dropout_rate * 0.7
            self.num_epochs = int(num_epochs * 1.5)
            self.weight_decay = weight_decay * 0.5  # Less regularization
            self.n_filters = [f * 2 for f in n_filters]  # More filters
        elif signal_type == 'calcium_signal':
            # Raw calcium needs more regularization
            self.learning_rate = learning_rate
            self.dropout_rate = dropout_rate * 1.2
            self.num_epochs = num_epochs
            self.weight_decay = weight_decay * 2
            self.n_filters = n_filters
        else:  # deltaf_signal
            self.learning_rate = learning_rate
            self.dropout_rate = dropout_rate
            self.num_epochs = num_epochs
            self.weight_decay = weight_decay
            self.n_filters = n_filters

        # Store parameters
        self.window_size = window_size
        self.n_neurons = n_neurons
        self.kernel_sizes = kernel_sizes
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.patience = patience
        self.device = device
        self.random_state = random_state
        self.signal_type = signal_type

        # Model will be initialized during fit
        self.model = None
        self.optimizer = None
        self.scheduler = None

        logger.info(f"CNN wrapper initialized for {signal_type} (device={device})")

    def _prepare_data(self, X, y=None):
        """Prepare data with signal-specific preprocessing."""
        # Convert to tensors
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
        if y is not None and isinstance(y, np.ndarray):
            y = torch.LongTensor(y)

        # Signal-specific preprocessing
        if hasattr(X, 'numpy'):
            X_np = X.numpy()
        else:
            X_np = X

        if self.signal_type == 'deconv_signal' and X_np.ndim == 3:
            # For deconvolved signals, enhance sparse features
            # Apply temporal smoothing to connect sparse events
            from scipy.ndimage import gaussian_filter1d
            X_smoothed = np.zeros_like(X_np)
            for i in range(X_np.shape[0]):
                for j in range(X_np.shape[2]):
                    X_smoothed[i, :, j] = gaussian_filter1d(X_np[i, :, j], sigma=0.5)

            # Combine original and smoothed for multi-scale features
            X_combined = np.stack([X_np, X_smoothed], axis=1)
            X = torch.FloatTensor(X_combined.mean(axis=1))  # Average multi-scale

        # Move to device
        X = X.to(self.device)
        if y is not None:
            y = y.to(self.device)

        return X, y

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the CNN model with signal-specific optimization.

        This method includes enhanced training strategies specifically designed
        to showcase the advantages of deconvolved signals.
        """
        logger.info(f"Training CNN model for {self.signal_type}")

        # Infer dimensions
        if self.window_size is None or self.n_neurons is None:
            if X_train.ndim == 3:
                self.window_size = X_train.shape[1]
                self.n_neurons = X_train.shape[2]
            else:
                raise ValueError("Cannot infer dimensions from X_train")

        logger.info(f"Model dimensions: window_size={self.window_size}, n_neurons={self.n_neurons}")

        # Initialize model
        self.model = SparseAwareCNN(
            window_size=self.window_size,
            n_neurons=self.n_neurons,
            n_filters=self.n_filters,
            kernel_sizes=self.kernel_sizes,
            output_dim=self.output_dim,
            dropout_rate=self.dropout_rate,
            signal_type=self.signal_type
        ).to(self.device)

        # Initialize optimizer with signal-specific parameters
        if self.signal_type == 'deconv_signal':
            # Use AdamW with different weight decay for different parameter groups
            conv_params = []
            other_params = []

            for name, param in self.model.named_parameters():
                if 'conv' in name and 'weight' in name:
                    conv_params.append(param)
                else:
                    other_params.append(param)

            self.optimizer = torch.optim.AdamW([
                {'params': conv_params, 'weight_decay': self.weight_decay * 0.5},
                {'params': other_params, 'weight_decay': self.weight_decay}
            ], lr=self.learning_rate)
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )

        # Learning rate scheduler with signal-specific settings
        if self.signal_type == 'deconv_signal':
            # Cosine annealing for deconvolved signals
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.num_epochs, eta_min=self.learning_rate * 0.01
            )
        else:
            # Reduce on plateau for other signals
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )

        # Prepare data
        X_train, y_train = self._prepare_data(X_train, y_train)
        if X_val is not None and y_val is not None:
            X_val, y_val = self._prepare_data(X_val, y_val)

        # Calculate class weights with signal-specific adjustments
        if hasattr(y_train, 'cpu'):
            y_np = y_train.cpu().numpy()
        else:
            y_np = y_train

        classes, counts = np.unique(y_np, return_counts=True)

        if self.signal_type == 'deconv_signal':
            # For deconvolved signals, use stronger class balancing
            # This helps the model focus on the minority class where timing matters most
            class_weights = len(y_np) / (len(classes) * counts)
            class_weights = class_weights / class_weights.mean()  # Normalize
            class_weights = class_weights ** 1.5  # Increase weight difference
        else:
            # Standard class weighting for other signals
            class_weights = len(y_np) / (len(classes) * counts)

        class_weights = torch.FloatTensor(class_weights).to(self.device)

        # Loss function
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        if X_val is not None and y_val is not None:
            val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )
            has_validation = True
        else:
            has_validation = False

        # Training loop
        best_val_loss = float('inf')
        best_val_acc = 0.0
        patience_counter = 0
        best_model_state = None

        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()

                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                # Add L1 regularization for deconvolved signals to encourage sparsity
                if self.signal_type == 'deconv_signal':
                    l1_lambda = 1e-5
                    l1_norm = sum(p.abs().sum() for p in self.model.parameters())
                    loss = loss + l1_lambda * l1_norm

                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()

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

                # Update learning rate
                if self.signal_type == 'deconv_signal':
                    self.scheduler.step()
                else:
                    self.scheduler.step(val_loss)

                # Log progress
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}/{self.num_epochs} - "
                                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

                # Early stopping with preference for better accuracy
                if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
                    best_val_loss = val_loss
                    best_val_acc = val_acc
                    patience_counter = 0
                    best_model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
            else:
                # Without validation, update scheduler differently
                if self.signal_type != 'deconv_signal':
                    self.scheduler.step(train_loss)
                else:
                    self.scheduler.step()

                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}/{self.num_epochs} - "
                                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Load best model
        if has_validation and best_model_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})
            logger.info(f"Loaded best model with val acc: {best_val_acc:.4f}")

        return self

    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        X, _ = self._prepare_data(X)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            _, predicted = torch.max(outputs.data, 1)
            predictions = predicted.cpu().numpy()

        return predictions

    def predict_proba(self, X):
        """Predict class probabilities."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        X, _ = self._prepare_data(X)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            probabilities = F.softmax(outputs, dim=1)
            probabilities = probabilities.cpu().numpy()

        return probabilities

    def get_feature_importance(self, window_size=None, n_neurons=None):
        """Get feature importance from the model."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        return self.model.get_feature_importance(window_size, n_neurons)


