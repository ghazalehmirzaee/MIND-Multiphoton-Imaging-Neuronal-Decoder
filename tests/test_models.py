"""Tests for model functions."""
import pytest
import numpy as np
import torch
import os
from mind.models.classical.random_forest import (
    create_random_forest,
    extract_feature_importance as rf_extract_feature_importance
)
from mind.models.classical.svm import create_svm
from mind.models.classical.mlp import (
    create_mlp,
    extract_feature_importance as mlp_extract_feature_importance
)
from mind.models.deep.fcnn import create_fcnn
from mind.models.deep.cnn import create_cnn


def test_create_random_forest():
    """Test create_random_forest function."""
    # Create sample config
    config = {
        'models': {
            'classical': {
                'random_forest': {
                    'n_estimators': 100,
                    'max_depth': 20,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'class_weight': 'balanced'
                }
            }
        },
        'experiment': {
            'seed': 42
        }
    }

    # Create model
    model = create_random_forest(config)

    # Check model
    assert hasattr(model, 'fit')
    assert hasattr(model, 'predict')
    assert model.n_estimators == 100
    assert model.max_depth == 20
    assert model.min_samples_split == 5
    assert model.min_samples_leaf == 2
    assert model.class_weight == 'balanced'
    assert model.random_state == 42


def test_rf_extract_feature_importance():
    """Test RF extract_feature_importance function."""

    # Create sample model with feature importances
    class MockRF:
        def __init__(self):
            self.feature_importances_ = np.random.rand(100)

    model = MockRF()

    # Extract feature importance
    importance_2d, temporal_importance, neuron_importance = rf_extract_feature_importance(
        model, window_size=10, n_neurons=10
    )

    # Check output shapes
    assert importance_2d.shape == (10, 10)
    assert temporal_importance.shape == (10,)
    assert neuron_importance.shape == (10,)


def test_create_svm():
    """Test create_svm function."""
    # Create sample config
    config = {
        'models': {
            'classical': {
                'svm': {
                    'kernel': 'rbf',
                    'C': 1.0,
                    'gamma': 'scale',
                    'class_weight': 'balanced',
                    'probability': True
                }
            }
        },
        'experiment': {
            'seed': 42
        }
    }

    # Create model
    model = create_svm(config)

    # Check model
    assert hasattr(model, 'fit')
    assert hasattr(model, 'predict')
    assert model.kernel == 'rbf'
    assert model.C == 1.0
    assert model.gamma == 'scale'
    assert model.class_weight == 'balanced'
    assert model.probability == True
    assert model.random_state == 42


def test_create_mlp():
    """Test create_mlp function."""
    # Create sample config
    config = {
        'models': {
            'classical': {
                'mlp': {
                    'hidden_layer_sizes': (128, 64),
                    'activation': 'relu',
                    'solver': 'adam',
                    'alpha': 0.0001,
                    'learning_rate': 'adaptive',
                    'early_stopping': True
                }
            }
        },
        'experiment': {
            'seed': 42
        }
    }

    # Create model
    model = create_mlp(config)

    # Check model
    assert hasattr(model, 'fit')
    assert hasattr(model, 'predict')
    assert model.hidden_layer_sizes == (128, 64)
    assert model.activation == 'relu'
    assert model.solver == 'adam'
    assert model.alpha == 0.0001
    assert model.learning_rate == 'adaptive'
    assert model.early_stopping == True
    assert model.random_state == 42


def test_mlp_extract_feature_importance():
    """Test MLP extract_feature_importance function."""

    # Create sample model with coefficients
    class MockMLP:
        def __init__(self):
            self.coefs_ = [np.random.rand(100, 50)]

    model = MockMLP()

    # Extract feature importance
    importance_2d, temporal_importance, neuron_importance = mlp_extract_feature_importance(
        model, window_size=10, n_neurons=10
    )

    # Check output shapes
    assert importance_2d.shape == (10, 10)
    assert temporal_importance.shape == (10,)
    assert neuron_importance.shape == (10,)


def test_create_fcnn():
    """Test create_fcnn function."""
    # Create sample config
    config = {
        'models': {
            'deep': {
                'fcnn': {
                    'hidden_sizes': [256, 128, 64],
                    'dropout_rates': [0.4, 0.4, 0.3],
                    'batch_norm': True
                }
            }
        }
    }

    # Create model
    model = create_fcnn(
        input_dim=150,
        n_classes=3,
        config=config
    )

    # Check model
    assert isinstance(model, torch.nn.Module)
    assert hasattr(model, 'forward')

    # Check forward pass
    batch_size = 10
    x = torch.randn(batch_size, 150)
    output = model(x)

    assert output.shape == (batch_size, 3)


def test_create_cnn():
    """Test create_cnn function."""
    # Create sample config
    config = {
        'models': {
            'deep': {
                'cnn': {
                    'channels': [64, 128, 256],
                    'kernel_size': 3,
                    'dropout_rate': 0.5,
                    'batch_norm': True
                }
            }
        }
    }

    # Create model
    model = create_cnn(
        input_size=10,
        window_size=15,
        n_classes=3,
        config=config
    )

    # Check model
    assert isinstance(model, torch.nn.Module)
    assert hasattr(model, 'forward')

    # Check forward pass
    batch_size = 10
    x = torch.randn(batch_size, 15, 10)
    output = model(x)

    assert output.shape == (batch_size, 3)

