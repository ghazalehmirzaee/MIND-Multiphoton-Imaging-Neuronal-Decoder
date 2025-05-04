"""Tests for data loading and processing functions."""
import pytest
import numpy as np
import os
import tempfile
from mind.data.loader import (
    load_matlab_data,
    load_behavioral_data,
    align_neural_behavioral_data,
    save_processed_data,
    load_processed_data
)
from mind.data.processor import (
    smooth_signals,
    create_sliding_windows,
    normalize_data,
    calculate_class_weights,
    process_data,
    split_data
)


def test_smooth_signals():
    """Test smooth_signals function."""
    # Create sample data
    data = np.random.rand(100, 10)

    # Apply smoothing
    smoothed_data = smooth_signals(data, window_length=5, polyorder=2)

    # Check output shape
    assert smoothed_data.shape == data.shape


def test_create_sliding_windows():
    """Test create_sliding_windows function."""
    # Create sample data
    data = np.random.rand(100, 10)
    labels = np.zeros(100)
    labels[20:40] = 1
    labels[60:80] = 2

    # Create windows
    windows, window_labels = create_sliding_windows(
        data, labels, window_size=15, step_size=1
    )

    # Check output shapes
    assert windows.shape[0] == 100 - 15 + 1
    assert windows.shape[1] == 15 * 10
    assert window_labels.shape[0] == 100 - 15 + 1


def test_normalize_data():
    """Test normalize_data function."""
    # Create sample data
    data = np.random.rand(100, 150)

    # Apply normalization
    normalized_data, scaler = normalize_data(data, scaler_type='robust')

    # Check output shape
    assert normalized_data.shape == data.shape

    # Check scaler
    assert hasattr(scaler, 'transform')


def test_calculate_class_weights():
    """Test calculate_class_weights function."""
    # Create sample data
    y_train = np.zeros(100)
    y_train[20:40] = 1
    y_train[60:80] = 2

    # Calculate class weights
    class_weights = calculate_class_weights(y_train)

    # Check output
    assert isinstance(class_weights, dict)
    assert set(class_weights.keys()) == {0, 1, 2}
    assert all(w > 0 for w in class_weights.values())


def test_save_and_load_processed_data():
    """Test save_processed_data and load_processed_data functions."""
    # Create sample data
    data = {
        'X_calcium': np.random.rand(100, 150),
        'y_calcium': np.zeros(100),
        'window_size': 15,
        'n_calcium_neurons': 10,
        'scalers': {'calcium': None}
    }

    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix='.npz') as temp_file:
        # Save data
        save_processed_data(data, temp_file.name)

        # Load data
        loaded_data = load_processed_data(temp_file.name)

        # Check keys
        assert set(loaded_data.keys()) == set(data.keys())

        # Check values
        assert np.array_equal(loaded_data['X_calcium'], data['X_calcium'])
        assert np.array_equal(loaded_data['y_calcium'], data['y_calcium'])
        assert loaded_data['window_size'] == data['window_size']
        assert loaded_data['n_calcium_neurons'] == data['n_calcium_neurons']

