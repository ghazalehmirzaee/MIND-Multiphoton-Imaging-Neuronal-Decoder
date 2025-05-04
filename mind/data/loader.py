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


def load_behavioral_data(file_path: str) -> pd.DataFrame:
    """
    Load behavioral data from Excel file.

    Parameters
    ----------
    file_path : str
        Path to the Excel file

    Returns
    -------
    pd.DataFrame
        DataFrame containing behavioral data with standardized column names
    """
    logger.info(f"Loading behavioral data from {file_path}")

    try:
        behavior_df = pd.read_excel(file_path)

        # Check and standardize column names
        required_columns = ['Frame Start', 'Frame End', 'Foot (L/R)']
        for col in required_columns:
            if col not in behavior_df.columns:
                logger.warning(f"Required column '{col}' not found in the Excel file.")

                # Try to detect similar columns and rename them
                if col == 'Frame Start' and 'Start' in behavior_df.columns:
                    behavior_df['Frame Start'] = behavior_df['Start']
                    logger.info(f"Renamed 'Start' to 'Frame Start'")

                elif col == 'Frame End' and 'End' in behavior_df.columns:
                    behavior_df['Frame End'] = behavior_df['End']
                    logger.info(f"Renamed 'End' to 'Frame End'")

                elif col == 'Foot (L/R)' and 'Foot' in behavior_df.columns:
                    behavior_df['Foot (L/R)'] = behavior_df['Foot']
                    logger.info(f"Renamed 'Foot' to 'Foot (L/R)'")

                else:
                    logger.error(f"Could not find or infer column '{col}'")
                    raise ValueError(f"Required column '{col}' not found in the Excel file")

        logger.info(f"Successfully loaded behavioral data with {len(behavior_df)} entries")
        logger.info(f"Columns in behavioral data: {behavior_df.columns.tolist()}")

        return behavior_df

    except Exception as e:
        logger.error(f"Error loading behavioral data: {e}")
        raise


def align_neural_behavioral_data(
        neural_data: Dict[str, np.ndarray],
        behavior_df: pd.DataFrame
) -> Dict[str, np.ndarray]:
    """
    Align neural recording frames with behavioral events.

    Parameters
    ----------
    neural_data : Dict[str, np.ndarray]
        Dictionary containing neural data
    behavior_df : pd.DataFrame
        DataFrame containing behavioral data

    Returns
    -------
    Dict[str, np.ndarray]
        Updated neural data dictionary with labels, where:
        - 0: No footstep
        - 1: RIGHT foot (contralateral)
        - 2: LEFT foot (ipsilateral)
    """
    logger.info("Aligning neural recording frames with behavioral events")

    # Extract frame information
    frame_starts = behavior_df['Frame Start'].values
    frame_ends = behavior_df['Frame End'].values
    foot_sides = behavior_df['Foot (L/R)'].values

    # Create label array
    num_frames = neural_data['calcium_signal'].shape[0]
    labels = np.zeros(num_frames)

    # Create different labels: RIGHT=1, LEFT=2
    for i in range(len(frame_starts)):
        start = int(frame_starts[i])
        end = int(frame_ends[i])

        # Check if frames are within data range
        if start < num_frames and end < num_frames:
            # Right foot (contralateral) = 1, Left foot (ipsilateral) = 2
            label_value = 1 if foot_sides[i] == 'R' else 2
            labels[start:end + 1] = label_value

    # Count instances of each class
    class_counts = np.bincount(labels.astype(int))
    logger.info(f"Label distribution:")
    logger.info(f"No footstep (0): {class_counts[0]} frames")
    if len(class_counts) > 1:
        logger.info(f"Right foot (1): {class_counts[1]} frames")
    if len(class_counts) > 2:
        logger.info(f"Left foot (2): {class_counts[2]} frames")

    # Add labels to the data dictionary
    neural_data['labels'] = labels

    return neural_data


def load_processed_data(file_path: str) -> Dict[str, Any]:
    """
    Load preprocessed data from NPZ file.

    Parameters
    ----------
    file_path : str
        Path to the NPZ file

    Returns
    -------
    Dict[str, Any]
        Dictionary containing preprocessed data
    """
    logger.info(f"Loading processed data from {file_path}")

    try:
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


def save_processed_data(data_dict: Dict[str, Any], file_path: str) -> None:
    """
    Save preprocessed data to NPZ file.

    Parameters
    ----------
    data_dict : Dict[str, Any]
        Dictionary containing data to save
    file_path : str
        Path to save the NPZ file
    """
    logger.info(f"Saving processed data to {file_path}")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    try:
        np.savez(file_path, **data_dict)
        logger.info(f"Data successfully saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving processed data: {e}")
        raise

