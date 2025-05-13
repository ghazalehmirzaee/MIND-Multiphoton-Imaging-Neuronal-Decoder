"""
Common training utilities.
"""
import torch
import numpy as np
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


def get_model_inputs_from_dataset(dataset: torch.utils.data.Dataset, indices=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract features and labels from a PyTorch dataset.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        PyTorch dataset
    indices : List[int], optional
        Indices to extract, by default None (all)

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Features and labels
    """
    if indices is None:
        indices = range(len(dataset))

    # For datasets where __getitem__ returns a tuple of (features, labels)
    if isinstance(dataset[0], tuple) and len(dataset[0]) == 2:
        features = []
        labels = []

        for idx in indices:
            X, y = dataset[idx]
            features.append(X)
            labels.append(y)

        # Stack features and labels
        features = torch.stack(features)
        labels = torch.stack(labels)

        return features, labels
    else:
        raise ValueError("Dataset does not return (features, labels) tuples")


def get_train_val_test_data(datasets: Dict[str, Dict[str, torch.utils.data.Dataset]],
                            signal_type: str) -> Tuple:
    """
    Extract train, validation, and test data for a specific signal type.

    Parameters
    ----------
    datasets : Dict[str, Dict[str, torch.utils.data.Dataset]]
        Dictionary of datasets
    signal_type : str
        Signal type to extract ('calcium_signal', 'deltaf_signal', or 'deconv_signal')

    Returns
    -------
    Tuple
        (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    logger.info(f"Extracting data for signal type: {signal_type}")

    # Check if signal type exists
    if signal_type not in datasets:
        raise ValueError(f"Signal type '{signal_type}' not found in datasets")

    # Extract datasets for this signal type
    signal_datasets = datasets[signal_type]

    # Extract data for each split
    X_train, y_train = get_model_inputs_from_dataset(signal_datasets['train'])
    X_val, y_val = get_model_inputs_from_dataset(signal_datasets['val'])
    X_test, y_test = get_model_inputs_from_dataset(signal_datasets['test'])

    logger.info(f"Data shapes - X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test

