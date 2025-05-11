# mind/data/loader.py
import os
import numpy as np
import pandas as pd
import h5py
import hdf5storage
from typing import Dict, Tuple, List, Optional


def load_matlab_data(file_path: str) -> Dict[str, np.ndarray]:
    """
    Load MATLAB .mat file containing calcium imaging data.

    Parameters
    ----------
    file_path : str
        Path to MATLAB file

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing the following keys:
        - 'calcium_signal': Raw calcium signal (2999 x 764)
        - 'deltaf': ∆F/F signal (2999 x 581)
        - 'deconv': Deconvolved signal (2999 x 581)
    """
    print(f"Loading MATLAB data from {file_path}")
    data = {}

    # Load MATLAB file
    mat_data = hdf5storage.loadmat(file_path)

    # Extract signals
    data['calcium_signal'] = mat_data.get('calciumsignal', None)
    data['deltaf'] = mat_data.get('deltaf_cells_not_excluded', None)
    data['deconv'] = mat_data.get('DeconvMat_wanted', None)

    # Ensure all required data is loaded
    missing_keys = [key for key, val in data.items() if val is None]
    if missing_keys:
        print(f"Warning: Missing data keys: {missing_keys}")

    # Print shapes to verify data
    for key, val in data.items():
        if val is not None:
            print(f"  {key}: {val.shape}")

    return data


def load_behavior_data(file_path: str) -> pd.DataFrame:
    """
    Load behavioral data from Excel file.

    Parameters
    ----------
    file_path : str
        Path to Excel file containing behavioral data

    Returns
    -------
    pd.DataFrame
        DataFrame containing behavioral data
    """
    print(f"Loading behavioral data from {file_path}")
    try:
        behavior_data = pd.read_excel(file_path)
        print(f"  Loaded behavioral data with {len(behavior_data)} entries")
        return behavior_data
    except Exception as e:
        print(f"Error loading behavioral data: {e}")
        return pd.DataFrame()


def align_neural_behavior(neural_data: Dict[str, np.ndarray],
                          behavior_data: pd.DataFrame,
                          binary_task: bool = True) -> Dict[str, np.ndarray]:
    """
    Align neural data with behavioral events and create frame labels.

    Parameters
    ----------
    neural_data : Dict[str, np.ndarray]
        Dictionary containing neural signals
    behavior_data : pd.DataFrame
        DataFrame containing behavioral data
    binary_task : bool, optional
        If True, create binary labels (0: no footstep, 1: contralateral footstep)
        If False, create multiclass labels (0: no footstep, 1: contralateral, 2: ipsilateral)

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with neural data and additional 'labels' key containing frame labels
    """
    print("Aligning neural data with behavior...")
    # Initialize labels with zeros (no footstep)
    num_frames = neural_data['calcium_signal'].shape[0]
    labels = np.zeros(num_frames, dtype=int)

    # Process each behavioral event
    for _, row in behavior_data.iterrows():
        start_frame = int(row['Frame Start']) if 'Frame Start' in row else int(row['Frame_Start'])
        end_frame = int(row['Frame End']) if 'Frame End' in row else int(row['Frame_End'])
        foot = row['Foot (L/R)'] if 'Foot (L/R)' in row else row['Foot']

        # Make sure frames are within bounds
        start_frame = max(0, min(start_frame, num_frames - 1))
        end_frame = max(0, min(end_frame, num_frames - 1))

        # Assign labels based on foot
        if binary_task:
            # Only label contralateral (right) footsteps as 1, ignore ipsilateral
            if foot == 'R':
                labels[start_frame:end_frame + 1] = 1
        else:
            # Label contralateral (right) as 1, ipsilateral (left) as 2
            if foot == 'R':
                labels[start_frame:end_frame + 1] = 1
            elif foot == 'L':
                labels[start_frame:end_frame + 1] = 2

    # Add labels to the data dictionary
    neural_data['labels'] = labels

    # Print label distribution
    unique, counts = np.unique(labels, return_counts=True)
    label_dist = dict(zip(unique, counts))
    print(f"  Label distribution: {label_dist}")

    return neural_data


def load_dataset(neural_path: str, behavior_path: str, binary_task: bool = True) -> Dict[str, np.ndarray]:
    """
    Load and align neural and behavioral data.

    Parameters
    ----------
    neural_path : str
        Path to MATLAB file containing neural data
    behavior_path : str
        Path to Excel file containing behavioral data
    binary_task : bool, optional
        If True, create binary classification task (no footstep vs contralateral)

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing neural signals and frame labels
    """
    # Load neural data
    neural_data = load_matlab_data(neural_path)

    # Load behavioral data
    behavior_data = load_behavior_data(behavior_path)

    # Align neural and behavioral data
    aligned_data = align_neural_behavior(neural_data, behavior_data, binary_task)

    return aligned_data

