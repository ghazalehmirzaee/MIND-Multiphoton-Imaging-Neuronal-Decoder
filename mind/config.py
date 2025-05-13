"""
Fixed configuration with proper Random Forest class balancing.
"""
import torch

DEFAULT_CONFIG = {
    # Data parameters
    "data": {
        "window_size": 15,
        "step_size": 1,
        "test_size": 0.15,
        "val_size": 0.15,
        "batch_size": 32,
        "num_workers": 4,
        "binary_classification": True,  # Always use binary classification
        "mat_file": "data/raw/SFL13_5_8112021_002_new.mat",
        "xlsx_file": "data/raw/SFL13_5_8112021_002_new.xlsx"
    },

    # Model parameters
    "models": {
        "random_forest": {
            "n_estimators": 300,
            "max_depth": 20,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "class_weight": "balanced",  # CRITICAL: This MUST be set for class balancing
            "n_jobs": -1,
            "random_state": 42,
            "criterion": "gini",  # Added for completeness
            "bootstrap": True    # Added for completeness
        },
        "svm": {
            "C": 1.0,
            "kernel": "rbf",
            "gamma": "scale",
            "class_weight": "balanced",  # Also important for SVM
            "probability": True,
            "random_state": 42,
            "n_components": 0.95,
            "use_pca": True
        },
        "mlp": {
            "hidden_layer_sizes": (64, 128, 32),
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.0001,
            "batch_size": "auto",
            "learning_rate": "adaptive",
            "learning_rate_init": 0.001,
            "max_iter": 300,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 15,
            "random_state": 42
        },
        "fcnn": {
            "hidden_dims": [256, 128, 64],
            "output_dim": 2,  # Binary classification
            "dropout_rate": 0.4,
            "learning_rate": 0.001,
            "weight_decay": 1e-5,
            "batch_size": 32,
            "num_epochs": 100,
            "patience": 15,
            "random_state": 42
        },
        "cnn": {
            "n_filters": [64, 128, 256],
            "kernel_size": 3,
            "output_dim": 2,  # Binary classification
            "dropout_rate": 0.3,
            "learning_rate": 0.001,
            "weight_decay": 1e-5,
            "batch_size": 32,
            "num_epochs": 100,
            "patience": 15,
            "random_state": 42
            # Removed use_focal_loss as it was causing issues
        }
    },

    # Training parameters
    "training": {
        "optimize_hyperparams": False,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "output_dir": "outputs/results"
    },

    # W&B parameters
    "wandb": {
        "use_wandb": True,
        "project_name": "mind-calcium-imaging",
        "entity": None  # Set to your W&B username or team
    },

    # Visualization parameters
    "visualization": {
        "output_dir": "outputs/figures",
        "dpi": 300,
        "format": "png"
    },

    # Binary classification parameters
    "classification": {
        "task": "binary",  # Always binary: no footstep (0) vs contralateral (1)
        "labels": ["No footstep", "Contralateral"],
        "n_classes": 2
    }
}


def get_config():
    """
    Get default configuration with device check.

    Returns
    -------
    dict
        Default configuration
    """
    # Update device based on CUDA availability
    if not torch.cuda.is_available() and DEFAULT_CONFIG["training"]["device"] == "cuda":
        DEFAULT_CONFIG["training"]["device"] = "cpu"

    return DEFAULT_CONFIG

