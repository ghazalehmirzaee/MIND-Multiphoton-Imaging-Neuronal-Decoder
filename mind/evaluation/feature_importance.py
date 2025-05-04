"""Feature importance analysis functions."""
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


def reshape_importance(
        importance: np.ndarray,
        window_size: int,
        n_neurons: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reshape feature importance array and calculate temporal and neuron importance.

    Parameters
    ----------
    importance : np.ndarray
        Feature importance array
    window_size : int
        Window size
    n_neurons : int
        Number of neurons

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        2D importance, temporal importance, and neuron importance
    """
    # Reshape to 2D (window_size, n_neurons)
    try:
        importance_2d = importance.reshape(window_size, n_neurons)
    except ValueError:
        logger.warning(
            f"Could not reshape importance array of shape {importance.shape} to ({window_size}, {n_neurons})")
        return np.ones((window_size, n_neurons)), np.ones(window_size), np.ones(n_neurons)

    # Calculate temporal importance (mean across neurons)
    temporal_importance = np.mean(importance_2d, axis=1)

    # Calculate neuron importance (mean across time)
    neuron_importance = np.mean(importance_2d, axis=0)

    return importance_2d, temporal_importance, neuron_importance


def identify_important_features(
        feature_importance: Dict[str, Dict[str, np.ndarray]],
        top_n: int = 20
) -> Dict[str, Dict[str, Any]]:
    """
    Identify the most important features.

    Parameters
    ----------
    feature_importance : Dict[str, Dict[str, np.ndarray]]
        Dictionary containing feature importance data
    top_n : int, optional
        Number of top features to return, by default 20

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary containing important features
    """
    important_features = {}

    for key, importance in feature_importance.items():
        if 'neuron_importance' not in importance:
            continue

        neuron_importance = importance['neuron_importance']

        # Get top neurons
        top_neuron_indices = np.argsort(neuron_importance)[-top_n:][::-1]
        top_neuron_values = neuron_importance[top_neuron_indices]

        # Store results
        important_features[key] = {
            'top_neuron_indices': top_neuron_indices.tolist(),
            'top_neuron_values': top_neuron_values.tolist()
        }

        # Add temporal information if available
        if 'temporal_importance' in importance:
            temporal_importance = importance['temporal_importance']
            peak_time = np.argmax(temporal_importance)
            important_features[key]['peak_time'] = int(peak_time)
            important_features[key]['peak_value'] = float(temporal_importance[peak_time])

    return important_features


def compare_feature_importance(
        feature_importance: Dict[str, Dict[str, np.ndarray]],
        signal_types: List[str] = ['calcium', 'deltaf', 'deconv'],
        model_types: List[str] = ['rf', 'mlp']
) -> Dict[str, Any]:
    """
    Compare feature importance across signal types and models.

    Parameters
    ----------
    feature_importance : Dict[str, Dict[str, np.ndarray]]
        Dictionary containing feature importance data
    signal_types : List[str], optional
        List of signal types, by default ['calcium', 'deltaf', 'deconv']
    model_types : List[str], optional
        List of model types, by default ['rf', 'mlp']

    Returns
    -------
    Dict[str, Any]
        Dictionary containing comparison results
    """
    comparison = {
        'overlap': {},
        'agreement': {}
    }

    # Get top neurons for each signal type and model
    top_neurons = {}
    top_n = 20

    for signal_type in signal_types:
        for model_type in model_types:
            key = f"{signal_type}_{model_type}"

            if key not in feature_importance:
                continue

            if 'neuron_importance' not in feature_importance[key]:
                continue

            # Get neuron importance
            neuron_importance = feature_importance[key]['neuron_importance']

            # Get top neurons
            top_neuron_indices = np.argsort(neuron_importance)[-top_n:]

            # Store top neurons
            top_neurons[key] = set(top_neuron_indices)

    # Calculate overlap between signal types (same model)
    for model_type in model_types:
        for i, signal_i in enumerate(signal_types):
            key_i = f"{signal_i}_{model_type}"

            if key_i not in top_neurons:
                continue

            for j, signal_j in enumerate(signal_types[i + 1:], i + 1):
                key_j = f"{signal_j}_{model_type}"

                if key_j not in top_neurons:
                    continue

                # Calculate overlap
                overlap = top_neurons[key_i].intersection(top_neurons[key_j])
                overlap_size = len(overlap)

                # Store overlap
                comparison['overlap'][f"{key_i}_vs_{key_j}"] = {
                    'neurons': sorted(list(overlap)),
                    'count': overlap_size,
                    'percentage': overlap_size / top_n * 100
                }

    # Calculate agreement between models (same signal type)
    for signal_type in signal_types:
        for i, model_i in enumerate(model_types):
            key_i = f"{signal_type}_{model_i}"

            if key_i not in top_neurons:
                continue

            for j, model_j in enumerate(model_types[i + 1:], i + 1):
                key_j = f"{signal_type}_{model_j}"

                if key_j not in top_neurons:
                    continue

                # Calculate overlap
                overlap = top_neurons[key_i].intersection(top_neurons[key_j])
                overlap_size = len(overlap)

                # Store overlap
                comparison['agreement'][f"{key_i}_vs_{key_j}"] = {
                    'neurons': sorted(list(overlap)),
                    'count': overlap_size,
                    'percentage': overlap_size / top_n * 100
                }

    return comparison
