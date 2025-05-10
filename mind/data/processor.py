"""Data processing functions."""
import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from tqdm import tqdm

from mind.data.loader import (
    load_matlab_data,
    load_behavioral_data,
    align_neural_behavioral_data,
    save_processed_data
)
from mind.utils.experiment_tracking import log_metrics, log_figures

logger = logging.getLogger(__name__)


def create_sliding_windows(
        data: np.ndarray,
        labels: np.ndarray,
        window_size: int = 15,
        step_size: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows from the data and corresponding labels.

    Parameters
    ----------
    data : np.ndarray
        Input data with shape (frames, neurons)
    labels : np.ndarray
        Input labels with shape (frames,)
    window_size : int, optional
        Size of the sliding window, by default 15
    step_size : int, optional
        Step size for sliding, by default 1

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Windowed data with shape (n_windows, window_size * neurons)
        and corresponding labels
    """
    n_samples = data.shape[0]
    windows = []
    window_labels = []

    for i in tqdm(range(0, n_samples - window_size + 1, step_size),
                  desc="Creating windows"):
        window = data[i:i + window_size, :]
        # Use majority vote for the window label
        window_label_counts = np.bincount(labels[i:i + window_size].astype(int))
        label = np.argmax(window_label_counts)

        windows.append(window.flatten())
        window_labels.append(label)

    return np.array(windows), np.array(window_labels)


def calculate_class_weights(y_train: np.ndarray) -> Dict[int, float]:
    """
    Calculate class weights inversely proportional to class frequencies.

    Parameters
    ----------
    y_train : np.ndarray
        Training labels

    Returns
    -------
    Dict[int, float]
        Dictionary mapping class indices to weights
    """
    classes = np.unique(y_train)
    class_counts = np.bincount(y_train.astype(int))
    total_samples = len(y_train)

    # Convert max class to int explicitly
    max_class = int(np.max(classes))
    full_class_counts = np.zeros(max_class + 1)

    for cls in classes:
        cls_int = int(cls)
        if cls_int < len(class_counts):
            full_class_counts[cls_int] = class_counts[cls_int]

    # Calculate weights with smoothing to prevent extreme values
    class_weights = {}
    for cls in classes:
        cls_int = int(cls)
        if full_class_counts[cls_int] > 0:
            # Apply smoothing to prevent extreme weights
            weight = np.log1p(total_samples / (len(classes) * full_class_counts[cls_int]))
            class_weights[cls_int] = min(10.0, max(1.0, weight))  # Clip between 1 and 10
        else:
            class_weights[cls_int] = 1.0

    logger.info(f"Calculated class weights: {class_weights}")
    return class_weights


def process_data(
        neural_data: Dict[str, np.ndarray],
        config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Process neural data with sliding windows for binary classification.
    Does not apply normalization or other preprocessing as per requirements.

    Parameters
    ----------
    neural_data : Dict[str, np.ndarray]
        Dictionary containing neural data
    config : Dict[str, Any]
        Configuration dictionary

    Returns
    -------
    Dict[str, Any]
        Dictionary containing processed data
    """
    logger.info("Processing data with sliding windows for binary classification")

    # Extract configuration
    window_size = config['data']['window_size']
    step_size = config['data']['step_size']

    # Validate input data
    required_keys = ['calcium_signal', 'deltaf_cells_not_excluded', 'deconv_mat_wanted', 'labels']
    for key in required_keys:
        if key not in neural_data:
            raise ValueError(f"Missing required key in neural_data: {key}")

    # Extract data
    calcium_signal = neural_data['calcium_signal']
    deltaf_cells_not_excluded = neural_data['deltaf_cells_not_excluded']
    deconv_mat_wanted = neural_data['deconv_mat_wanted']
    labels = neural_data['labels']

    # Get dimensions
    n_frames, n_calcium_neurons = calcium_signal.shape
    _, n_deltaf_neurons = deltaf_cells_not_excluded.shape
    _, n_deconv_neurons = deconv_mat_wanted.shape

    logger.info(f"Number of frames: {n_frames}")
    logger.info(f"Calcium signal neurons: {n_calcium_neurons}")
    logger.info(f"ΔF/F neurons: {n_deltaf_neurons}")
    logger.info(f"Deconvolved neurons: {n_deconv_neurons}")

    # Handle NaN values (basic preprocessing)
    if np.isnan(calcium_signal).any():
        logger.warning("calcium_signal contains NaN values")
        calcium_signal = np.nan_to_num(calcium_signal, nan=0.0)
    if np.isnan(deltaf_cells_not_excluded).any():
        logger.warning("deltaf_cells_not_excluded contains NaN values")
        deltaf_cells_not_excluded = np.nan_to_num(deltaf_cells_not_excluded, nan=0.0)
    if np.isnan(deconv_mat_wanted).any():
        logger.warning("deconv_mat_wanted contains NaN values")
        deconv_mat_wanted = np.nan_to_num(deconv_mat_wanted, nan=0.0)

    # Ensure binary classification (0 = no footstep, 1 = contralateral footstep)
    binary_labels = labels.copy()
    binary_labels[binary_labels > 1] = 0  # Convert ipsilateral (2) to no footstep (0)

    # Count label distribution
    unique_labels, counts = np.unique(binary_labels, return_counts=True)
    logger.info("Binary label distribution:")
    for lbl, cnt in zip(unique_labels, counts):
        logger.info(f"  Label {int(lbl)}: {cnt} frames")

    # Create sliding windows without normalization
    logger.info(f"Creating sliding windows (size={window_size}, step={step_size})")
    try:
        X_calcium, y_calcium = create_sliding_windows(calcium_signal, binary_labels, window_size, step_size)
        X_deltaf, y_deltaf = create_sliding_windows(deltaf_cells_not_excluded, binary_labels, window_size, step_size)
        X_deconv, y_deconv = create_sliding_windows(deconv_mat_wanted, binary_labels, window_size, step_size)
    except Exception as e:
        logger.error(f"Error creating sliding windows: {e}")
        raise ValueError(f"Failed to create sliding windows: {e}")

    logger.info(f"Created {X_calcium.shape[0]} windows for each signal type")

    # Prepare processed data dictionary
    processed_data = {
        'X_calcium': X_calcium,
        'y_calcium': y_calcium,
        'X_deltaf': X_deltaf,
        'y_deltaf': y_deltaf,
        'X_deconv': X_deconv,
        'y_deconv': y_deconv,
        'window_size': window_size,
        'n_calcium_neurons': n_calcium_neurons,
        'n_deltaf_neurons': n_deltaf_neurons,
        'n_deconv_neurons': n_deconv_neurons,
        'raw_calcium': calcium_signal,
        'raw_deltaf': deltaf_cells_not_excluded,
        'raw_deconv': deconv_mat_wanted,
        'binary_task': True  # Indicate binary classification
    }

    logger.info("Data processing completed successfully")
    return processed_data


def split_data(
        processed_data: Dict[str, Any],
        config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Split data into train, validation, and test sets.

    Parameters
    ----------
    processed_data : Dict[str, Any]
        Dictionary containing processed data
    config : Dict[str, Any]
        Configuration dictionary

    Returns
    -------
    Dict[str, Any]
        Dictionary containing split data
    """
    from sklearn.model_selection import train_test_split

    logger.info("Splitting data into train, validation, and test sets")

    # Extract configuration
    test_size = config['data'].get('test_size', 0.2)
    val_size = config['data'].get('val_size', 0.15)
    random_state = config['experiment'].get('seed', 42)

    result = {}

    # Process each signal type
    for signal_type in ['calcium', 'deltaf', 'deconv']:
        logger.info(f"Splitting {signal_type} data")
        X = processed_data[f'X_{signal_type}']
        y = processed_data[f'y_{signal_type}']

        # First split: training + validation vs test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y  # Ensure balanced classes
        )

        # Second split: training vs validation
        # Adjust validation size to get the right proportion from the remaining data
        adjusted_val_size = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=adjusted_val_size,
            random_state=random_state,
            stratify=y_train_val  # Ensure balanced classes
        )

        # Store splits
        result[f'X_train_{signal_type}'] = X_train
        result[f'X_val_{signal_type}'] = X_val
        result[f'X_test_{signal_type}'] = X_test
        result[f'y_train_{signal_type}'] = y_train
        result[f'y_val_{signal_type}'] = y_val
        result[f'y_test_{signal_type}'] = y_test

        # Print class distribution
        for subset_name, y_subset in [('train', y_train), ('validation', y_val), ('test', y_test)]:
            class_counts = np.bincount(y_subset.astype(int))
            logger.info(f"{subset_name} set class distribution ({signal_type}):")
            logger.info(f"  No footstep (0): {class_counts[0]} samples")
            if len(class_counts) > 1:
                logger.info(f"  Right foot (1): {class_counts[1]} samples")

    # Calculate class weights for each signal type
    class_weights = {}
    for signal_type in ['calcium', 'deltaf', 'deconv']:
        y_train = result[f'y_train_{signal_type}']
        class_weights[signal_type] = calculate_class_weights(y_train)

    # Add class weights to result
    result['class_weights'] = class_weights

    # Store raw signals for visualization
    for signal_type in ['calcium', 'deltaf', 'deconv']:
        if f'raw_{signal_type}' in processed_data:
            result[f'raw_{signal_type}'] = processed_data[f'raw_{signal_type}']

    # Add metadata
    for key in ['window_size', 'n_calcium_neurons', 'n_deltaf_neurons', 'n_deconv_neurons', 'binary_task']:
        if key in processed_data:
            result[key] = processed_data[key]

    logger.info("Data splitting completed successfully")
    return result


def main(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main processing function.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary

    Returns
    -------
    Dict[str, Any]
        Dictionary containing processed and split data
    """
    # Setup logging
    logger.info(
        f"Starting data processing with window_size={config['data']['window_size']}, step_size={config['data']['step_size']}")

    # Load data
    matlab_file = config['data']['file']
    behavior_file = config['data'].get('behavior_file')

    # Load neural data from MATLAB file
    neural_data = load_matlab_data(matlab_file)

    # If behavior file is provided, load and align behavioral data
    if behavior_file:
        behavior_df = load_behavioral_data(behavior_file)
        neural_data = align_neural_behavioral_data(neural_data, behavior_df,
                                                   binary_task=config['data'].get('binary_task', True))
    elif 'labels' not in neural_data:
        logger.warning("No behavioral data provided and no labels found in neural data")
        logger.warning("Creating dummy labels (all zeros)")
        neural_data['labels'] = np.zeros(neural_data['calcium_signal'].shape[0])
        neural_data['binary_task'] = True

    # Process data
    processed_data = process_data(neural_data, config)

    # Split data
    split_data_dict = split_data(processed_data, config)

    # Combine processed and split data
    final_data = {**processed_data, **split_data_dict}

    # Add the matlab file path to the data dictionary
    final_data['matlab_file'] = matlab_file

    # Get the project root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # Save processed data to default location
    processed_dir = os.path.join(project_root, 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    filename = f"{os.path.basename(matlab_file).split('.')[0]}_processed.npz"
    output_path = os.path.join(processed_dir, filename)

    save_processed_data(final_data, output_path)

    logger.info(f"Data processing completed. Results saved to {output_path}")

    return final_data

