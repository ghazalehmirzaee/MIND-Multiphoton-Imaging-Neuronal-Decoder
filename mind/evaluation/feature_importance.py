"""
Feature importance analysis for neuronal decoding models.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging

logger = logging.getLogger(__name__)

def extract_feature_importance(model, window_size: int, n_neurons: int) -> np.ndarray:
    """
    Extract feature importance from a trained model.
    """
    logger.info("Extracting feature importance from model")

    try:
        # Check for the get_feature_importance method specific to our models
        if hasattr(model, 'get_feature_importance'):
            # Some models need dimensions as input
            try:
                importance = model.get_feature_importance(window_size, n_neurons)
            except TypeError:
                # Try without arguments
                importance = model.get_feature_importance()

        # SVM doesn't have feature importance
        elif hasattr(model, 'model') and 'svm' in str(model.model).lower():
            logger.warning("SVM models do not provide feature importance. Returning zeros.")
            importance = np.zeros((window_size, n_neurons))

        # Fallback for sklearn-like models
        elif hasattr(model, 'feature_importances_'):
            # Make sure feature_importances_ is a NumPy array
            if hasattr(model.feature_importances_, 'cpu'):
                importance = model.feature_importances_.cpu().numpy().reshape(window_size, n_neurons)
            else:
                importance = model.feature_importances_.reshape(window_size, n_neurons)

        # Handle models with coef_ (like some linear models)
        elif hasattr(model, 'coef_'):
            # Make sure coef_ is a NumPy array
            if hasattr(model.coef_, 'cpu'):
                importance = np.abs(model.coef_.cpu().numpy()).reshape(window_size, n_neurons)
            else:
                importance = np.abs(model.coef_).reshape(window_size, n_neurons)
            importance = importance / importance.sum() if importance.sum() > 0 else importance

        else:
            logger.warning("Model does not provide feature importance. Returning zeros.")
            importance = np.zeros((window_size, n_neurons))

        # Normalize 
        if importance.sum() > 0:
            importance = importance / importance.sum()

        logger.info(f"Extracted feature importance with shape {importance.shape}")

        return importance

    except Exception as e:
        logger.error(f"Error extracting feature importance: {e}")
        # Return zeros as a fallback
        return np.zeros((window_size, n_neurons))


def analyze_temporal_importance(importance_matrix: np.ndarray) -> np.ndarray:
    """
    Analyze temporal importance by averaging across neurons.
    """
    # Calculate mean importance across neurons for each time step
    temporal_importance = importance_matrix.mean(axis=1)

    # Normalize
    if temporal_importance.sum() > 0:
        temporal_importance = temporal_importance / temporal_importance.sum()

    return temporal_importance


def analyze_neuron_importance(importance_matrix: np.ndarray, top_n: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analyze neuron-specific importance by averaging across time steps.
    """
    # Calculate mean importance across time steps for each neuron
    neuron_importance = importance_matrix.mean(axis=0)

    # Get indices of top neurons
    top_indices = np.argsort(neuron_importance)[::-1][:top_n]

    # Normalize
    if neuron_importance.sum() > 0:
        neuron_importance = neuron_importance / neuron_importance.sum()

    return neuron_importance, top_indices

def find_important_time_windows(importance_matrix: np.ndarray, percentile: float = 90) -> List[Tuple[int, int]]:
    """
    Find time windows with high feature importance.
    """
    # Calculate temporal importance
    temporal_importance = analyze_temporal_importance(importance_matrix)

    # Calculate threshold
    threshold = np.percentile(temporal_importance, percentile)

    # Find time points above threshold
    above_threshold = temporal_importance > threshold

    # Find contiguous segments
    segments = []
    start = None

    for i, above in enumerate(above_threshold):
        if above and start is None:
            start = i
        elif not above and start is not None:
            segments.append((start, i - 1))
            start = None

    # Handle case where the last segment extends to the end
    if start is not None:
        segments.append((start, len(above_threshold) - 1))

    return segments


def create_importance_summary(importance_matrix: np.ndarray, window_size: int, n_neurons: int) -> Dict[str, Any]:
    """
    Create a summary of feature importance.
    """
    # Analyze temporal importance
    temporal_importance = analyze_temporal_importance(importance_matrix)

    # Analyze neuron importance and get top neurons
    neuron_importance, top_neuron_indices = analyze_neuron_importance(importance_matrix, top_n=20)

    # Find important time windows
    important_windows = find_important_time_windows(importance_matrix)

    # Create importance summary
    summary = {
        'temporal_importance': temporal_importance,
        'neuron_importance': neuron_importance,
        'top_neuron_indices': top_neuron_indices,
        'important_windows': important_windows,
        'importance_matrix': importance_matrix
    }

    return summary

