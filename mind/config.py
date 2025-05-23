"""
Configuration for MIND: Multiphoton Imaging Neural Decoder.
Optimized for publication with focus on key findings.
"""
import torch

DEFAULT_CONFIG = {
    # Data parameters
    "data": {
        "window_size": 15,  # 15 frames = ~975ms at 15.4Hz
        "step_size": 1,
        "test_size": 0.15,
        "val_size": 0.15,
        "mat_file": "data/raw/SFL13_5_8112021_002_new_modified.mat",
        "xlsx_file": "data/raw/SFL13_5_8112021_002_new.xlsx"
    },

    # Model parameters optimized for calcium imaging
    "models": {
        "random_forest": {
            "n_estimators": 300,
            "max_depth": 20,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "class_weight": "balanced_subsample",
            "random_state": 42
        },
        "svm": {
            "C": 1.0,
            "kernel": "rbf",
            "gamma": "scale",
            "class_weight": "balanced",
            "use_pca": True,
            "n_components": 0.95,
            "random_state": 42
        },
        "mlp": {
            "hidden_layer_sizes": (128, 64, 32),
            "activation": "relu",
            "learning_rate_init": 0.001,
            "max_iter": 300,
            "early_stopping": True,
            "random_state": 42
        },
        "fcnn": {
            "hidden_dims": [256, 128, 64],
            "dropout_rate": 0.3,
            "learning_rate": 0.001,
            "num_epochs": 100,
            "random_state": 42
        },
        "cnn": {
            "n_filters": [64, 128, 256],
            "kernel_size": 3,
            "dropout_rate": 0.2,
            "learning_rate": 0.0005,
            "num_epochs": 100,
            "random_state": 42
        }
    },

    # Training parameters
    "training": {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "output_dir": "outputs/results",
        "seed": 42
    },

    # Visualization colors (scientific color scheme)
    "visualization": {
        "signal_colors": {
            "calcium_signal": "#356d9e",
            "deltaf_signal": "#4c8b64",
            "deconv_signal": "#a85858"
        }
    }
}

