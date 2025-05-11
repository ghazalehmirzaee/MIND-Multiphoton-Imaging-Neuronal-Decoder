# mind/data/processor.py
import numpy as np
from typing import Dict, Tuple, List, Any
from sklearn.model_selection import train_test_split


def create_sliding_windows(data: Dict[str, np.ndarray],
                           window_size: int = 15,
                           step_size: int = 1) -> Dict[str, np.ndarray]:
    """
    Create sliding windows from the neural signals and corresponding labels.

    Parameters
    ----------
    data : Dict[str, np.ndarray]
        Dictionary containing neural signals and frame labels
    window_size : int, optional
        Size of the sliding window in frames
    step_size : int, optional
        Step size for sliding the window

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing windowed neural signals and corresponding labels
    """
    print(f"Creating sliding windows (size={window_size}, step={step_size})...")
    windowed_data = {}

    num_frames = data['labels'].shape[0]

    # Calculate number of windows
    num_windows = (num_frames - window_size) // step_size + 1
    print(f"  Number of windows: {num_windows}")

    # For each signal type, create windowed data
    for key in ['calcium_signal', 'deltaf', 'deconv']:
        if key not in data or data[key] is None:
            continue

        signal = data[key]
        num_neurons = signal.shape[1]

        # Initialize windowed array
        windowed_signal = np.zeros((num_windows, window_size, num_neurons))

        # Create sliding windows
        for i in range(num_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size
            windowed_signal[i] = signal[start_idx:end_idx, :]

        # Reshape to (num_windows, window_size * num_neurons)
        windowed_data[key] = windowed_signal.reshape(num_windows, -1)

    # Create labels for each window (use the label of the last frame in each window)
    windowed_labels = np.zeros(num_windows, dtype=int)
    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size - 1  # Last frame in the window
        window_labels = data['labels'][start_idx:end_idx + 1]

        # Use majority label in the window
        unique_labels, counts = np.unique(window_labels, return_counts=True)
        windowed_labels[i] = unique_labels[np.argmax(counts)]

    windowed_data['labels'] = windowed_labels

    # Print windowed data shapes
    for key, val in windowed_data.items():
        print(f"  {key}: {val.shape}")

    # Print label distribution
    unique, counts = np.unique(windowed_labels, return_counts=True)
    label_dist = dict(zip(unique, counts))
    print(f"  Label distribution: {label_dist}")

    return windowed_data


def split_data(data: Dict[str, np.ndarray],
               test_size: float = 0.15,
               val_size: float = 0.15,
               random_state: int = 42) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Split data into training, validation, and test sets.

    Parameters
    ----------
    data : Dict[str, np.ndarray]
        Dictionary containing windowed neural signals and labels
    test_size : float, optional
        Proportion of data to use for testing
    val_size : float, optional
        Proportion of data to use for validation
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    Dict[str, Dict[str, np.ndarray]]
        Dictionary containing 'train', 'val', and 'test' splits
    """
    print(f"Splitting data (test={test_size}, val={val_size})...")
    splits = {'train': {}, 'val': {}, 'test': {}}

    # Calculate effective validation size from remaining data after test split
    effective_val_size = val_size / (1 - test_size)

    # Get labels
    labels = data['labels']

    # First split off test set
    X_temp_indices, X_test_indices, y_temp, y_test = train_test_split(
        np.arange(len(labels)), labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    # Then split the remaining data into train and validation
    X_train_indices, X_val_indices, y_train, y_val = train_test_split(
        X_temp_indices, y_temp,
        test_size=effective_val_size,
        random_state=random_state,
        stratify=y_temp
    )

    # Split each signal type
    for key in ['calcium_signal', 'deltaf', 'deconv', 'labels']:
        if key not in data or data[key] is None:
            continue

        splits['train'][key] = data[key][X_train_indices]
        splits['val'][key] = data[key][X_val_indices]
        splits['test'][key] = data[key][X_test_indices]

    # Print split shapes
    for split_name, split_data in splits.items():
        print(f"  {split_name} split:")
        for key, val in split_data.items():
            print(f"    {key}: {val.shape}")

    return splits

