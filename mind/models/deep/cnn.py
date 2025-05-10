"""Convolutional Neural Network model implementation."""
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)


class CNNModel(nn.Module):
    """
    1D Convolutional Neural Network model for calcium imaging data with dynamic feature handling.
    """

    def __init__(
            self,
            input_size: int,
            window_size: int,
            n_classes: int,
            channels: List[int] = [64, 128, 256],
            kernel_size: int = 3,
            dropout_rate: float = 0.5,
            batch_norm: bool = True,
            signal_type: str = None
    ):
        """
        Initialize the CNN model with auto-adaptive architecture.

        Parameters
        ----------
        input_size : int
            Number of input features (neurons)
        window_size : int
            Window size (time steps)
        n_classes : int
            Number of output classes
        channels : List[int], optional
            List of channel sizes for convolutional layers, by default [64, 128, 256]
        kernel_size : int, optional
            Kernel size for convolutional layers, by default 3
        dropout_rate : float, optional
            Dropout rate, by default 0.5
        batch_norm : bool, optional
            Whether to use batch normalization, by default True
        signal_type : str, optional
            Signal type to optimize for, by default None
        """
        super(CNNModel, self).__init__()

        # Enhanced architecture for different signal types
        if signal_type == 'deconv':
            # Deconvolved signals have sparse, spike-like features
            channels = [96, 192, 384]
            kernel_size = 5
            dropout_rate = 0.4
        elif signal_type == 'deltaf':
            # ΔF/F signals have normalized features
            channels = [80, 160, 320]
            kernel_size = 4

        # Store signal type and dimensions
        self.signal_type = signal_type
        self.input_size = input_size
        self.window_size = window_size

        # Convolutional layers
        self.conv1 = nn.Conv1d(input_size, channels[0], kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(channels[0]) if batch_norm else nn.Identity()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(channels[0], channels[1], kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(channels[1]) if batch_norm else nn.Identity()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(channels[1], channels[2], kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn3 = nn.BatchNorm1d(channels[2]) if batch_norm else nn.Identity()

        # Residual connection
        self.residual = nn.Conv1d(input_size, channels[2], kernel_size=1)
        self.bn_res = nn.BatchNorm1d(channels[2]) if batch_norm else nn.Identity()

        # Additional layer for deconvolved signals
        if signal_type == 'deconv':
            self.conv4 = nn.Conv1d(channels[2], channels[2], kernel_size=kernel_size, padding=kernel_size // 2)
            self.bn4 = nn.BatchNorm1d(channels[2]) if batch_norm else nn.Identity()

        # Use adaptive pooling to get a fixed size output regardless of input dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool1d(4)  # Always outputs 4 time steps
        self.final_channels = channels[2]

        # Calculate the fixed flattened size after adaptive pooling
        self.flat_size = self.final_channels * 4

        # Fully connected layers with fixed size
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.flat_size, 128)
        self.bn_fc = nn.BatchNorm1d(128) if batch_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, n_classes)

        logger.info(f"CNN model initialized with input_size={input_size}, window_size={window_size}")
        logger.info(f"Using adaptive pooling to fixed size: {self.flat_size} features")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with adaptive feature handling.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        # Transform from [batch, time_steps, features] to [batch, features, time_steps]
        if x.dim() == 3:
            x = x.permute(0, 2, 1)
            residual_in = x
        else:
            # Handle single sample case
            x = x.unsqueeze(0).permute(0, 2, 1)
            residual_in = x

        # Main path
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = torch.relu(self.bn3(self.conv3(x)))

        # Additional layer for deconvolved signals
        if self.signal_type == 'deconv':
            x = torch.relu(self.bn4(self.conv4(x)))

        # Residual connection (with adaptive pooling to match dimensions)
        res = self.residual(residual_in)
        res = self.bn_res(res)
        res = nn.functional.adaptive_avg_pool1d(res, x.size(2))

        # Add residual connection (if shapes match)
        if x.size() == res.size():
            x = x + res

        # Use adaptive pooling to get fixed size regardless of input dimensions
        x = self.adaptive_pool(x)

        # Flatten and pass through fully connected layers
        x = self.flatten(x)
        x = torch.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def get_activation_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get activation maps from the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of activation maps
        """
        # Transform from [batch, time_steps, features] to [batch, features, time_steps]
        if x.dim() == 3:
            x = x.permute(0, 2, 1)
            residual_in = x
        else:
            # Handle single sample case
            x = x.unsqueeze(0).permute(0, 2, 1)
            residual_in = x

        # Store activations
        activations = {}

        # Main path
        conv1_out = self.conv1(x)
        bn1_out = self.bn1(conv1_out)
        relu1_out = torch.relu(bn1_out)
        pool1_out = self.pool1(relu1_out)

        conv2_out = self.conv2(pool1_out)
        bn2_out = self.bn2(conv2_out)
        relu2_out = torch.relu(bn2_out)
        pool2_out = self.pool2(relu2_out)

        conv3_out = self.conv3(pool2_out)
        bn3_out = self.bn3(conv3_out)
        relu3_out = torch.relu(bn3_out)

        # Additional layer for deconvolved signals
        if self.signal_type == 'deconv':
            conv4_out = self.conv4(relu3_out)
            bn4_out = self.bn4(conv4_out)
            relu4_out = torch.relu(bn4_out)
            combined_out = relu4_out
            activations['conv4'] = conv4_out
            activations['relu4'] = relu4_out
        else:
            combined_out = relu3_out

        # Residual connection
        res_out = self.residual(residual_in)
        bn_res_out = self.bn_res(res_out)
        res_pool_out = nn.functional.adaptive_avg_pool1d(bn_res_out, combined_out.size(2))

        # Add residual connection (if shapes match)
        if combined_out.size() == res_pool_out.size():
            final_out = combined_out + res_pool_out
        else:
            final_out = combined_out

        # Store activations
        activations['conv1'] = conv1_out
        activations['relu1'] = relu1_out
        activations['pool1'] = pool1_out
        activations['conv2'] = conv2_out
        activations['relu2'] = relu2_out
        activations['pool2'] = pool2_out
        activations['conv3'] = conv3_out
        activations['relu3'] = relu3_out
        activations['residual'] = res_out
        activations['combined'] = final_out

        return activations


def create_cnn(
        input_size: int,
        window_size: int,
        n_classes: int,
        config: Dict[str, Any],
        signal_type: str = None
) -> CNNModel:
    """
    Create a CNN model with the specified configuration.

    Parameters
    ----------
    input_size : int
        Number of input features (neurons)
    window_size : int
        Window size (time steps)
    n_classes : int
        Number of output classes
    config : Dict[str, Any]
        Configuration dictionary
    signal_type : str, optional
        Signal type to optimize for, by default None

    Returns
    -------
    CNNModel
        Initialized CNN model
    """
    cnn_params = config['models']['deep']['cnn'].copy()

    model = CNNModel(
        input_size=input_size,
        window_size=window_size,
        n_classes=n_classes,
        channels=cnn_params.get('channels', [64, 128, 256]),
        kernel_size=cnn_params.get('kernel_size', 3),
        dropout_rate=cnn_params.get('dropout_rate', 0.5),
        batch_norm=cnn_params.get('batch_norm', True),
        signal_type=signal_type
    )

    return model

