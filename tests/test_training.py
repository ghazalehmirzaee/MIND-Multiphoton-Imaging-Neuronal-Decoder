"""Tests for training functions."""
import pytest
import numpy as np
import torch
import torch.nn as nn
import os
import tempfile
from mind.training.train import (
    FocalLoss,
    create_dataloaders,
    create_model_name
)
from mind.training.trainer_classical import (
    train_random_forest,
    train_svm,
    train_mlp,
    save_classical_models,
    load_classical_models
)
from mind.training.trainer_deep import (
    create_optimizer,
    save_deep_models,
    load_deep_models
)


def test_focal_loss():
    """Test FocalLoss class."""
    # Create loss function
    loss_fn = FocalLoss(alpha=1.0, gamma=2.0)

    # Create sample data
    inputs = torch.randn(10, 3)
    targets = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

    # Calculate loss
    loss = loss_fn(inputs, targets)

    # Check output
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])  # Scalar
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_create_dataloaders():
    """Test create_dataloaders function."""
    # Create sample data
    data = {
        'X_train_calcium': np.random.rand(100, 150),
        'y_train_calcium': np.zeros(100),
        'X_val_calcium': np.random.rand(20, 150),
        'y_val_calcium': np.zeros(20),
        'X_test_calcium': np.random.rand(20, 150),
        'y_test_calcium': np.zeros(20),
        'window_size': 15,
        'n_calcium_neurons': 10
    }

    # Create dataloaders
    dataloaders = create_dataloaders(
        data=data,
        signal_type='calcium',
        batch_size=32,
        window_size=15,
        reshape=False
    )

    # Check output
    assert 'train_loader' in dataloaders
    assert 'val_loader' in dataloaders
    assert 'test_loader' in dataloaders
    assert 'input_dim' in dataloaders
    assert 'window_size' in dataloaders
    assert 'n_neurons' in dataloaders
    assert 'n_classes' in dataloaders


def test_create_model_name():
    """Test create_model_name function."""
    # Create model name
    model_name = create_model_name('calcium', 'random_forest')

    # Check output
    assert model_name == 'calcium_random_forest'


def test_create_optimizer():
    """Test create_optimizer function."""
    # Create sample model
    model = nn.Linear(10, 3)

    # Create sample config
    config = {
        'training': {
            'learning_rate': 0.001,
            'weight_decay': 1e-5
        }
    }

    # Create optimizer
    optimizer = create_optimizer(model, config)

    # Check output
    assert isinstance(optimizer, torch.optim.Optimizer)
    assert optimizer.param_groups[0]['lr'] == 0.001
    assert optimizer.param_groups[0]['weight_decay'] == 1e-5


def test_save_and_load_models():
    """Test save_classical_models and load_classical_models functions."""
    # Create sample models
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC

    models = {
        'calcium_random_forest': RandomForestClassifier(),
        'calcium_svm': SVC()
    }

    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save models
        save_classical_models(models, output_dir=temp_dir)

        # Check files
        assert os.path.exists(os.path.join(temp_dir, 'calcium_random_forest.pkl'))
        assert os.path.exists(os.path.join(temp_dir, 'calcium_svm.pkl'))

        # Load models
        loaded_models = load_classical_models(
            model_names=['calcium_random_forest', 'calcium_svm'],
            input_dir=temp_dir
        )

        # Check loaded models
        assert set(loaded_models.keys()) == set(models.keys())
        assert isinstance(loaded_models['calcium_random_forest'], RandomForestClassifier)
        assert isinstance(loaded_models['calcium_svm'], SVC)

