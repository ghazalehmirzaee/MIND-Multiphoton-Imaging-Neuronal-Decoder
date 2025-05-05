import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from typing import Dict, Optional, Tuple, Union, Any, List
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


def smooth_signals(
        data: np.ndarray,
        window_length: int = 5,
        polyorder: int = 2
) -> np.ndarray:
    """
    Apply Savitzky-Golay filter to smooth signals.

    Parameters
    ----------
    data : np.ndarray
        Input data with shape (frames, neurons)
    window_length : int, optional
        Length of the filter window (must be odd), by default 5
    polyorder : int, optional
        Order of the polynomial, by default 2

    Returns
    -------
    np.ndarray
        Smoothed data
    """
    smoothed_data = np.zeros_like(data)
    n_frames, n_neurons = data.shape

    for i in range(n_neurons):
        smoothed_data[:, i] = savgol_filter(
            data[:, i],
            window_length=window_length,
            polyorder=polyorder,
            mode='nearest'
        )

    return smoothed_data


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


def normalize_data(
        data: np.ndarray,
        scaler_type: str = 'robust'
) -> Tuple[np.ndarray, Union[StandardScaler, RobustScaler, MinMaxScaler]]:
    """
    Normalize the data using the specified scaler.

    Parameters
    ----------
    data : np.ndarray
        Input data to normalize
    scaler_type : str, optional
        Type of scaler to use ('standard', 'robust', 'minmax'), by default 'robust'

    Returns
    -------
    Tuple[np.ndarray, Union[StandardScaler, RobustScaler, MinMaxScaler]]
        Normalized data and the fitted scaler
    """
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Invalid scaler type: {scaler_type}")

    normalized_data = scaler.fit_transform(data)
    return normalized_data, scaler


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
    Process neural data with sliding windows and prepare for ML with minimal preprocessing.
    This version removes aggressive optimizations while ensuring deconvolved signals can show
    their natural advantages.

    Parameters
    ----------
    neural_data : Dict[str, np.ndarray]
        Dictionary containing neural data and labels
    config : Dict[str, Any]
        Configuration dictionary

    Returns
    -------
    Dict[str, Any]
        Dictionary containing processed data
    """
    logger.info("Processing data with sliding windows")

    # Extract configuration
    window_size = config['data']['window_size']
    step_size = config['data']['step_size']

    # Simplified preprocessing configuration - removed unnecessary options
    normalize = config['processing'].get('normalize', True)
    scaler_type = config['processing'].get('scaler', 'robust')

    # Validate input data
    required_keys = ['calcium_signal', 'deltaf_cells_not_excluded', 'deconv_mat_wanted', 'labels']
    for key in required_keys:
        if key not in neural_data:
            raise ValueError(f"Missing required key in neural_data: {key}")
        if not isinstance(neural_data[key], np.ndarray):
            raise ValueError(f"Key {key} in neural_data is not a numpy array: {type(neural_data[key])}")

    # Extract data
    calcium_signal = neural_data['calcium_signal']
    deltaf_cells_not_excluded = neural_data['deltaf_cells_not_excluded']
    deconv_mat_wanted = neural_data['deconv_mat_wanted']
    labels = neural_data['labels']

    # Validate data shapes
    if calcium_signal.ndim != 2:
        raise ValueError(f"calcium_signal must be 2D, got shape {calcium_signal.shape}")
    if deltaf_cells_not_excluded.ndim != 2:
        raise ValueError(f"deltaf_cells_not_excluded must be 2D, got shape {deltaf_cells_not_excluded.shape}")
    if deconv_mat_wanted.ndim != 2:
        raise ValueError(f"deconv_mat_wanted must be 2D, got shape {deconv_mat_wanted.shape}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {labels.shape}")

    # Get dimensions
    n_frames, n_calcium_neurons = calcium_signal.shape
    _, n_deltaf_neurons = deltaf_cells_not_excluded.shape
    _, n_deconv_neurons = deconv_mat_wanted.shape

    logger.info(f"Number of frames: {n_frames}")
    logger.info(f"Calcium signal neurons: {n_calcium_neurons}")
    logger.info(f"ΔF/F neurons: {n_deltaf_neurons}")
    logger.info(f"Deconvolved neurons: {n_deconv_neurons}")

    # Basic NaN handling - this is essential preprocessing, not optimization
    if np.isnan(calcium_signal).any():
        logger.warning("calcium_signal contains NaN values")
        calcium_signal = np.nan_to_num(calcium_signal, nan=0.0)
    if np.isnan(deltaf_cells_not_excluded).any():
        logger.warning("deltaf_cells_not_excluded contains NaN values")
        deltaf_cells_not_excluded = np.nan_to_num(deltaf_cells_not_excluded, nan=0.0)
    if np.isnan(deconv_mat_wanted).any():
        logger.warning("deconv_mat_wanted contains NaN values")
        deconv_mat_wanted = np.nan_to_num(deconv_mat_wanted, nan=0.0)

    # Handle infinite values - again, essential preprocessing
    if not np.isfinite(calcium_signal).all():
        logger.warning("calcium_signal contains infinite values")
        calcium_signal = np.nan_to_num(calcium_signal, nan=0.0, posinf=1e10, neginf=-1e10)
    if not np.isfinite(deltaf_cells_not_excluded).all():
        logger.warning("deltaf_cells_not_excluded contains infinite values")
        deltaf_cells_not_excluded = np.nan_to_num(deltaf_cells_not_excluded, nan=0.0, posinf=1e10, neginf=-1e10)
    if not np.isfinite(deconv_mat_wanted).all():
        logger.warning("deconv_mat_wanted contains infinite values")
        deconv_mat_wanted = np.nan_to_num(deconv_mat_wanted, nan=0.0, posinf=1e10, neginf=-1e10)

    # Convert to float - basic data preparation
    calcium_signal = calcium_signal.astype(np.float64)
    deltaf_cells_not_excluded = deltaf_cells_not_excluded.astype(np.float64)
    deconv_mat_wanted = deconv_mat_wanted.astype(np.float64)
    labels = labels.astype(np.int32)

    # REMOVED: Smoothing signals - removed this preprocessing step
    # We'll use raw signals directly

    # Create sliding windows
    logger.info(f"Creating sliding windows (size={window_size}, step={step_size})")
    try:
        X_calcium, y_calcium = create_sliding_windows(calcium_signal, labels, window_size, step_size)
        X_deltaf, y_deltaf = create_sliding_windows(deltaf_cells_not_excluded, labels, window_size, step_size)
        X_deconv, y_deconv = create_sliding_windows(deconv_mat_wanted, labels, window_size, step_size)
    except Exception as e:
        logger.error(f"Error creating sliding windows: {e}")
        raise ValueError(f"Failed to create sliding windows: {e}")

    logger.info(f"Created {X_calcium.shape[0]} windows for each signal type")

    # Normalize data if requested - basic but necessary normalization
    scalers = {}
    if normalize:
        logger.info(f"Normalizing data using {scaler_type} scaler")
        try:
            X_calcium_norm, scaler_calcium = normalize_data(X_calcium, scaler_type)
            X_deltaf_norm, scaler_deltaf = normalize_data(X_deltaf, scaler_type)
            X_deconv_norm, scaler_deconv = normalize_data(X_deconv, scaler_type)

            scalers = {
                'calcium': scaler_calcium,
                'deltaf': scaler_deltaf,
                'deconv': scaler_deconv
            }
        except Exception as e:
            logger.warning(f"Error during normalization: {e}. Falling back to original data.")
            X_calcium_norm = X_calcium
            X_deltaf_norm = X_deltaf
            X_deconv_norm = X_deconv
    else:
        X_calcium_norm = X_calcium
        X_deltaf_norm = X_deltaf
        X_deconv_norm = X_deconv

    # REMOVED: Signal-specific optimizations for deconvolved signals

    # Prepare processed data dictionary
    processed_data = {
        'X_calcium': X_calcium_norm,
        'y_calcium': y_calcium,
        'X_deltaf': X_deltaf_norm,
        'y_deltaf': y_deltaf,
        'X_deconv': X_deconv_norm,
        'y_deconv': y_deconv,
        'window_size': window_size,
        'n_calcium_neurons': n_calcium_neurons,
        'n_deltaf_neurons': n_deltaf_neurons,
        'n_deconv_neurons': n_deconv_neurons,
        'scalers': scalers,
        'raw_calcium': calcium_signal,
        'raw_deltaf': deltaf_cells_not_excluded,
        'raw_deconv': deconv_mat_wanted,
        'binary_task': True  # Set this to True for binary classification
    }

    # Final validation of all arrays to ensure they are numeric
    for key, value in processed_data.items():
        if isinstance(value, np.ndarray):
            # Check dtype - object dtype might contain dictionaries or other non-numeric values
            if value.dtype == np.dtype('O'):
                logger.warning(f"Array {key} still has object dtype after processing.")
                # Try to convert to appropriate dtype
                try:
                    if key.startswith('X_'):
                        processed_data[key] = value.astype(np.float64)
                    elif key.startswith('y_'):
                        processed_data[key] = value.astype(np.int32)
                except Exception as e:
                    logger.error(f"Cannot convert {key} to numerical format: {e}")
                    raise ValueError(f"Cannot convert {key} to numerical format: {e}")

            # Check for NaN or inf values
            if key.startswith('X_') or key.startswith('raw_'):
                if np.isnan(value).any() or not np.isfinite(value).all():
                    logger.warning(f"Array {key} contains NaN or infinite values after processing.")
                    processed_data[key] = np.nan_to_num(value, nan=0.0, posinf=1e10, neginf=-1e10)

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
            stratify=y
        )

        # Second split: training vs validation
        # Adjust validation size to get the right proportion from the remaining data
        adjusted_val_size = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=adjusted_val_size,
            random_state=random_state,
            stratify=y_train_val
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
            logger.info(f"{subset_name} set class distribution:")
            logger.info(f"  No footstep (0): {class_counts[0]} samples")
            if len(class_counts) > 1:
                logger.info(f"  Right foot (1): {class_counts[1]} samples")
            if len(class_counts) > 2:
                logger.info(f"  Left foot (2): {class_counts[2]} samples")

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
    for key in ['window_size', 'n_calcium_neurons', 'n_deltaf_neurons', 'n_deconv_neurons', 'scalers']:
        if key in processed_data:
            result[key] = processed_data[key]

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
    matlab_file = config['data']['matlab_file']
    behavior_file = config['data']['behavior_file']

    neural_data = load_matlab_data(matlab_file)
    behavior_df = load_behavioral_data(behavior_file)

    # Align neural and behavioral data
    neural_data = align_neural_behavioral_data(neural_data, behavior_df)

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


if __name__ == "__main__":
    import hydra
    from omegaconf import OmegaConf, DictConfig
    import logging
    from mind.utils.logging import setup_logging


    @hydra.main(config_path="../config", config_name="default")
    def process_data_main(cfg: DictConfig) -> None:
        # Setup logging
        setup_logging()

        # Convert to dictionary
        config = OmegaConf.to_container(cfg, resolve=True)

        # Process data
        main(config)


    process_data_main()
