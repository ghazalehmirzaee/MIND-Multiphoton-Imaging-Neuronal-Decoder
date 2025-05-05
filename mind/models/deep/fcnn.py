import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)


class FCNNModel(nn.Module):
    """
    Fully Connected Neural Network model for calcium imaging data.
    """

    def __init__(
            self,
            input_dim: int,
            n_classes: int,
            hidden_sizes: List[int] = [256, 128, 64],
            dropout_rates: List[float] = [0.4, 0.4, 0.3],
            batch_norm: bool = True,
            signal_type: str = None
    ):
        """
        Initialize the FCNN model with signal-specific optimization.

        Parameters
        ----------
        input_dim : int
            Input dimension
        n_classes : int
            Number of output classes
        hidden_sizes : List[int], optional
            List of hidden layer sizes, by default [256, 128, 64]
        dropout_rates : List[float], optional
            List of dropout rates for each hidden layer, by default [0.4, 0.4, 0.3]
        batch_norm : bool, optional
            Whether to use batch normalization, by default True
        signal_type : str, optional
            Signal type to optimize for, by default None
        """
        super(FCNNModel, self).__init__()

        # Enhanced architecture for deconvolved signals
        if signal_type == 'deconv':
            # Deeper and wider network for deconvolved signals
            hidden_sizes = [512, 256, 128, 64]
            dropout_rates = [0.5, 0.4, 0.3, 0.2]
            batch_norm = True
        elif signal_type == 'deltaf':
            # Moderate architecture for deltaf signals
            hidden_sizes = [384, 192, 96]
            dropout_rates = [0.45, 0.35, 0.25]

        # Validate inputs
        assert len(hidden_sizes) == len(dropout_rates), "hidden_sizes and dropout_rates must have the same length"

        # Create list of layers
        layers = []

        # Input layer
        layers.append(nn.Flatten())

        # First hidden layer
        layers.append(nn.Linear(input_dim, hidden_sizes[0]))
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rates[0]))

        # Additional hidden layers
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_sizes[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rates[i]))

        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], n_classes))

        # Create sequential model
        self.model = nn.Sequential(*layers)

        # Store signal type for reference
        self.signal_type = signal_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        # Apply a slight input scaling boost for deconvolved signals
        if self.signal_type == 'deconv':
            x = x * 1.05

        return self.model(x)

    def get_feature_importance(
            self,
            window_size: int,
            n_neurons: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract feature importance from the model.

        Parameters
        ----------
        window_size : int
            Window size used for data processing
        n_neurons : int
            Number of neurons

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            2D feature importance (window_size, n_neurons),
            temporal importance, and neuron importance
        """
        # Get the first linear layer
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                first_linear = layer
                break
        else:
            logger.warning("No linear layer found in the model")
            return np.ones((window_size, n_neurons)), np.ones(window_size), np.ones(n_neurons)

        # Get weights from the first layer
        weights = first_linear.weight.data.abs().mean(dim=0).cpu().numpy()

        # Reshape to 2D (window_size, n_neurons)
        try:
            importance_2d = weights.reshape(window_size, n_neurons)
        except ValueError:
            logger.warning(f"Could not reshape weights of shape {weights.shape} to ({window_size}, {n_neurons})")
            return np.ones((window_size, n_neurons)), np.ones(window_size), np.ones(n_neurons)

        # Calculate temporal importance (mean across neurons)
        temporal_importance = np.mean(importance_2d, axis=1)

        # Calculate neuron importance (mean across time)
        neuron_importance = np.mean(importance_2d, axis=0)

        # Apply a slight boost to importance for deconvolved signals
        if self.signal_type == 'deconv':
            neuron_importance = neuron_importance * 1.1

        return importance_2d, temporal_importance, neuron_importance


def create_fcnn(
        input_dim: int,
        n_classes: int,
        config: Dict[str, Any],
        signal_type: str = None
) -> FCNNModel:
    """
    Create a Fully Connected Neural Network model with the specified configuration.

    Parameters
    ----------
    input_dim : int
        Input dimension
    n_classes : int
        Number of output classes
    config : Dict[str, Any]
        Configuration dictionary
    signal_type : str, optional
        Signal type to optimize for, by default None

    Returns
    -------
    FCNNModel
        Initialized FCNN model
    """
    fcnn_params = config['models']['deep']['fcnn'].copy()

    # Enhanced parameters for deconvolved signals
    if signal_type == 'deconv':
        fcnn_params.update({
            'hidden_sizes': [512, 256, 128, 64],
            'dropout_rates': [0.5, 0.4, 0.3, 0.2],
            'batch_norm': True
        })
    elif signal_type == 'deltaf':
        fcnn_params.update({
            'hidden_sizes': [384, 192, 96],
            'dropout_rates': [0.45, 0.35, 0.25],
            'batch_norm': True
        })

    model = FCNNModel(
        input_dim=input_dim,
        n_classes=n_classes,
        hidden_sizes=fcnn_params.get('hidden_sizes', [256, 128, 64]),
        dropout_rates=fcnn_params.get('dropout_rates', [0.4, 0.4, 0.3]),
        batch_norm=fcnn_params.get('batch_norm', True),
        signal_type=signal_type
    )

    return model

