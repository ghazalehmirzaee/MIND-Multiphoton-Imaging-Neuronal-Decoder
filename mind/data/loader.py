import os
import numpy as np
import scipy.io as sio
import pandas as pd
from typing import Dict, Optional, Tuple, Union, Any
import logging

logger = logging.getLogger(__name__)


def load_matlab_data(file_path: str) -> Dict[str, np.ndarray]:
    """
    Load calcium imaging data from MATLAB files.

    Parameters
    ----------
    file_path : str
        Path to the MATLAB file

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing the following keys:
        - 'calcium_signal': Raw calcium signal (frames × neurons)
        - 'deltaf_cells_not_excluded': ΔF/F for valid neurons (frames × valid_neurons)
        - 'deconv_mat_wanted': Deconvolved signals for valid neurons (frames × valid_neurons)
        - 'valid_neurons': Indices of valid neurons
    """
    logger.info(f"Loading MATLAB data from {file_path}")

    try:
        mat_data = sio.loadmat(file_path)

        # Extract the three specific signal types
        calcium_signal = mat_data['calciumsignal']  # Raw signal
        deltaf_cells_not_excluded = mat_data['deltaf_cells_not_excluded']  # ΔF/F
        deconv_mat_wanted = mat_data['DeconvMat_wanted']  # Deconvolved

        # Extract valid neuron indices for reference
        valid_neurons = mat_data['cells_not_excluded'].flatten() - 1  # 0-based indexing

        logger.info(f"Loaded data dimensions:")
        logger.info(f"Calcium signal: {calcium_signal.shape}")
        logger.info(f"ΔF/F (valid neurons): {deltaf_cells_not_excluded.shape}")
        logger.info(f"Deconvolved (valid neurons): {deconv_mat_wanted.shape}")
        logger.info(f"Number of valid neurons: {len(valid_neurons)}")

        return {
            'calcium_signal': calcium_signal,
            'deltaf_cells_not_excluded': deltaf_cells_not_excluded,
            'deconv_mat_wanted': deconv_mat_wanted,
            'valid_neurons': valid_neurons
        }

    except Exception as e:
        logger.error(f"Error loading MATLAB file: {e}")
        raise

def align_neural_behavioral_data(neural_data, behavior_df, binary_task=True):
    """Align neural recording frames with behavioral events with binary option."""
    # Extract frame information
    frame_starts = behavior_df['Frame Start'].values
    frame_ends = behavior_df['Frame End'].values
    foot_sides = behavior_df['Foot (L/R)'].values

    # Create label array
    num_frames = neural_data['calcium_signal'].shape[0]
    labels = np.zeros(num_frames)

    if binary_task:
        # Binary task: 0 for No footstep, 1 for RIGHT foot (contralateral) only
        for i in range(len(frame_starts)):
            start = int(frame_starts[i])
            end = int(frame_ends[i])

            if start < num_frames and end < num_frames:
                # Only label RIGHT foot (contralateral) as 1
                if foot_sides[i] == 'R':
                    labels[start:end + 1] = 1
    else:
        # Original multi-class labeling
        for i in range(len(frame_starts)):
            start = int(frame_starts[i])
            end = int(frame_ends[i])

            if start < num_frames and end < num_frames:
                label_value = 1 if foot_sides[i] == 'R' else 2
                labels[start:end + 1] = label_value

    # Count instances of each class
    class_counts = np.bincount(labels.astype(int))
    logger.info(f"Label distribution:")
    logger.info(f"No footstep (0): {class_counts[0]} frames")
    logger.info(f"Right foot (1): {class_counts[1]} frames")

    neural_data['labels'] = labels
    neural_data['binary_task'] = binary_task

    return neural_data

def load_processed_data(file_path: str = None) -> Dict[str, Any]:
    """
    Load preprocessed data from NPZ file.

    Parameters
    ----------
    file_path : str, optional
        Path to the NPZ file. If None, will look in default location.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing preprocessed data
    """
    logger.info(f"Loading processed data")

    # If file_path is not provided, look in the default location
    if file_path is None:
        # Get the project root directory
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        # Look in the default processed data directory
        processed_dir = os.path.join(project_root, 'data', 'processed')

        # Find all NPZ files in the directory
        processed_files = [f for f in os.listdir(processed_dir) if f.endswith('.npz')]

        if not processed_files:
            logger.error(f"No processed data files found in {processed_dir}")
            raise FileNotFoundError(f"No processed data files found in {processed_dir}")

        # Use the first file or ask the user to specify if multiple files exist
        if len(processed_files) > 1:
            logger.warning(f"Multiple processed data files found: {processed_files}")
            logger.warning(f"Using the first file: {processed_files[0]}")

        file_path = os.path.join(processed_dir, processed_files[0])

    try:
        logger.info(f"Loading processed data from {file_path}")
        data = np.load(file_path, allow_pickle=True)
        data_dict = {key: data[key] for key in data.files}

        # Convert object arrays to appropriate types
        for key in data_dict:
            if isinstance(data_dict[key], np.ndarray) and data_dict[key].dtype == np.dtype('O'):
                if key == 'scalers' or key == 'class_weights':
                    data_dict[key] = data_dict[key].item()

        logger.info(f"Successfully loaded processed data")
        return data_dict

    except Exception as e:
        logger.error(f"Error loading processed data: {e}")
        raise


def save_processed_data(data_dict: Dict[str, Any], file_path: str = None) -> None:
    """
    Save preprocessed data to NPZ file.

    Parameters
    ----------
    data_dict : Dict[str, Any]
        Dictionary containing data to save
    file_path : str, optional
        Path to save the NPZ file. If None, will save in default location.
    """
    # If file_path is not provided, save to the default location
    if file_path is None:
        # Get the project root directory
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        # Default processed data directory
        processed_dir = os.path.join(project_root, 'data', 'processed')

        # Create default filename from the first MATLAB file if available
        if 'matlab_file' in data_dict:
            matlab_file = data_dict['matlab_file']
            filename = f"{os.path.basename(matlab_file).split('.')[0]}_processed.npz"
        else:
            # Use a timestamp if no MATLAB file is available
            import time
            filename = f"processed_data_{int(time.time())}.npz"

        file_path = os.path.join(processed_dir, filename)

    logger.info(f"Saving processed data to {file_path}")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    try:
        np.savez(file_path, **data_dict)
        logger.info(f"Data successfully saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving processed data: {e}")
        raise

