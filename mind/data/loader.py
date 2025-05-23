"""
Efficient data loader for calcium imaging and behavioral data.
"""
import numpy as np
import pandas as pd
import scipy.io
import hdf5storage
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def load_calcium_signals(mat_file_path: str) -> Dict[str, np.ndarray]:
    """Load calcium imaging signals from MATLAB file."""
    try:
        # Try scipy.io first, then hdf5storage
        try:
            data = scipy.io.loadmat(mat_file_path)
        except NotImplementedError:
            data = hdf5storage.loadmat(mat_file_path)

        signals = {
            'calcium_signal': data.get('calciumsignal_wanted'),
            'deltaf_signal': data.get('deltaf_cells_not_excluded'),
            'deconv_signal': data.get('DeconvMat_wanted')
        }

        # Log signal shapes
        for name, signal in signals.items():
            if signal is not None:
                logger.info(f"{name}: {signal.shape}")

        return signals

    except Exception as e:
        logger.error(f"Error loading calcium signals: {e}")
        raise


def create_frame_labels(behavior_df: pd.DataFrame, num_frames: int) -> np.ndarray:
    """Create binary frame labels (0: no footstep, 1: contralateral footstep)."""
    frame_labels = np.zeros(num_frames, dtype=np.int32)

    for _, row in behavior_df.iterrows():
        foot = str(row['Foot (L/R)']).lower()
        start = max(0, int(row['Frame Start']))
        end = min(num_frames - 1, int(row['Frame End']))

        # Only label contralateral (right) footsteps as 1
        if 'right' in foot or 'r' == foot:
            frame_labels[start:end + 1] = 1

    # Log class distribution
    unique, counts = np.unique(frame_labels, return_counts=True)
    for label, count in zip(unique, counts):
        logger.info(f"Label {label}: {count} frames ({count/num_frames*100:.1f}%)")

    return frame_labels


def load_and_align_data(mat_file_path: str, xlsx_file_path: str) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Load and align calcium imaging data with behavioral labels."""
    # Load calcium signals
    calcium_signals = load_calcium_signals(mat_file_path)

    # Get number of frames from first available signal
    num_frames = next(s.shape[0] for s in calcium_signals.values() if s is not None)

    # Load behavioral data and create labels
    behavior_df = pd.read_excel(xlsx_file_path)
    frame_labels = create_frame_labels(behavior_df, num_frames)

    return calcium_signals, frame_labels

