"""
Updated trainer module with NumPy 2.0 compatibility and enhanced reproducibility.
"""
import time
import numpy as np
import json
from pathlib import Path
import logging
import wandb
import torch
import random

from mind.models.classical.random_forest import RandomForestModel
from mind.models.classical.svm import SVMModel
from mind.models.classical.mlp import MLPModel
from mind.models.deep.fcnn import FCNNWrapper
from mind.models.deep.cnn import CNNWrapper
from mind.evaluation.metrics import evaluate_model
from mind.evaluation.feature_importance import extract_feature_importance, create_importance_summary
from mind.training.train import get_train_val_test_data

logger = logging.getLogger(__name__)


def set_model_seed(seed: int = 42):
    """
    Set random seeds for model initialization and training reproducibility.

    This function ensures that model weights are initialized consistently
    and that any randomness during training is reproducible.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def numpy_to_json_serializable(obj):
    """
    Convert numpy arrays and other non-serializable objects to JSON-serializable format.

    This function is compatible with both NumPy 1.x and 2.x, using a more general
    approach to detect numpy types rather than checking for specific type names
    that might change between versions.

    Parameters
    ----------
    obj : any
        Object to convert

    Returns
    -------
    any
        JSON-serializable object
    """
    # Handle numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # Handle numpy scalars using the generic base class
    # This works with both NumPy 1.x and 2.x
    elif isinstance(obj, np.generic):
        return obj.item()

    # Handle regular Python types
    elif isinstance(obj, bool):
        return bool(obj)
    elif isinstance(obj, (int, float)):
        return obj

    # Handle collections
    elif isinstance(obj, dict):
        return {key: numpy_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return [numpy_to_json_serializable(item) for item in obj]

    # Handle None
    elif obj is None:
        return None

    # Handle custom objects with __dict__
    elif hasattr(obj, '__dict__'):
        return numpy_to_json_serializable(obj.__dict__)

    # Last resort: convert to string
    else:
        try:
            return str(obj)
        except:
            logger.warning(f"Could not convert object of type {type(obj)} to JSON-serializable format")
            return None


def train_model(model_type, model_params, datasets, signal_type, window_size, n_neurons,
                output_dir, device="cuda", optimize_hyperparams=False, use_wandb=True):
    """
    Unified training function for all model types with reproducibility guarantees.

    This function ensures reproducible training by:
    1. Setting random seeds before model initialization
    2. Using deterministic operations where possible
    3. Properly converting all results to JSON-serializable format

    Parameters
    ----------
    model_type : str
        Type of model ('random_forest', 'svm', 'mlp', 'fcnn', 'cnn')
    model_params : dict
        Model parameters including random_state
    datasets : dict
        Dictionary of datasets
    signal_type : str
        Type of signal
    window_size : int
        Window size
    n_neurons : int
        Number of neurons
    output_dir : str
        Output directory
    device : str, optional
        Device for training ('cuda' or 'cpu'), by default "cuda"
    optimize_hyperparams : bool, optional
        Whether to optimize hyperparameters, by default False
    use_wandb : bool, optional
        Whether to use W&B for tracking, by default True

    Returns
    -------
    dict
        Dictionary of results with all numpy arrays converted to lists
    """
    logger.info(f"Training {model_type} on {signal_type}")

    # Set seed before model initialization for reproducibility
    seed = model_params.get('random_state', 42)
    set_model_seed(seed)

    # Extract data
    X_train, y_train, X_val, y_val, X_test, y_test = get_train_val_test_data(datasets, signal_type)

    # Initialize model with reproducibility settings
    try:
        if model_type == 'random_forest':
            model_params = model_params.copy()
            model_params["optimize_hyperparams"] = optimize_hyperparams
            model = RandomForestModel(**model_params)

        elif model_type == 'svm':
            model_params = model_params.copy()
            model_params["optimize_hyperparams"] = optimize_hyperparams
            model = SVMModel(**model_params)

        elif model_type == 'mlp':
            model_params = model_params.copy()
            model_params["optimize_hyperparams"] = optimize_hyperparams
            model = MLPModel(**model_params)

        elif model_type == 'fcnn':
            model_params = model_params.copy()
            model_params["input_dim"] = window_size * n_neurons
            model_params["device"] = device
            model = FCNNWrapper(**model_params)

        elif model_type == 'cnn':
            model_params = model_params.copy()
            model_params["window_size"] = window_size
            model_params["n_neurons"] = n_neurons
            model_params["device"] = device
            model = CNNWrapper(**model_params)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        logger.info(f"Initialized {model_type} model with seed {seed}")

    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        raise

    # Train model
    try:
        start_time = time.time()
        model.fit(X_train, y_train, X_val, y_val)
        train_time = time.time() - start_time
        logger.info(f"Model training completed in {train_time:.2f} seconds")

    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise

    # Evaluate model
    try:
        results = evaluate_model(model, X_test, y_test)
        results['train_time'] = train_time

        # Log metrics to W&B if requested
        if use_wandb:
            try:
                # Check if wandb is initialized
                if wandb.run is None:
                    # Initialize wandb with a simple run
                    wandb.init(project="mind-calcium-imaging",
                               name=f"{model_type}_{signal_type}",
                               config={"model_type": model_type, "signal_type": signal_type})

                # Now log the metrics
                wandb.log(results.get('metrics', {}))
            except Exception as wandb_err:
                logger.warning(f"Error logging to W&B: {wandb_err}. Continuing without W&B logging.")
                use_wandb = False  # Disable for the rest of this run

        logger.info(f"Model evaluation complete: {results.get('metrics', {})}")

    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise

    # Extract feature importance
    try:
        importance_matrix = extract_feature_importance(model, window_size, n_neurons)
        importance_summary = create_importance_summary(importance_matrix, window_size, n_neurons)

        # Add importance summary to results
        results['importance_summary'] = importance_summary

        # Log to W&B if requested
        if use_wandb and isinstance(importance_summary, dict):
            try:
                # Log histograms if available
                if 'temporal_importance' in importance_summary:
                    wandb.log({"temporal_importance": wandb.Histogram(importance_summary['temporal_importance'])})
                if 'neuron_importance' in importance_summary:
                    wandb.log({"neuron_importance": wandb.Histogram(importance_summary['neuron_importance'])})

                # Save importance heatmap as a figure
                import matplotlib.pyplot as plt
                import seaborn as sns

                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(importance_matrix, cmap='viridis', ax=ax)
                ax.set_xlabel("Neuron")
                ax.set_ylabel("Time Step")
                ax.set_title(f"Feature Importance - {model_type} - {signal_type}")

                # Log to W&B
                wandb.log({"importance_heatmap": wandb.Image(fig)})
                plt.close(fig)
            except Exception as e:
                logger.warning(f"Could not log feature importance to W&B: {e}")

        logger.info("Feature importance analysis complete")

    except Exception as e:
        logger.warning(f"Could not extract feature importance: {e}")

    # Save results with robust JSON serialization
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{signal_type}_{model_type}_{timestamp}.json"

        # Convert all numpy arrays to JSON-serializable format
        results_json = numpy_to_json_serializable(results)

        # Add metadata
        results_json['metadata'] = {
            'model_name': model_type,
            'signal_type': signal_type,
            'timestamp': timestamp,
            'random_seed': seed,
            'window_size': window_size,
            'n_neurons': n_neurons
        }

        with open(output_dir / filename, 'w') as f:
            json.dump(results_json, f, indent=2)

        logger.info(f"Results saved to {output_dir / filename}")

    except Exception as e:
        logger.error(f"Error saving results: {e}")
        # Debug which specific data is causing issues
        logger.error("Attempting to identify non-serializable data...")
        for key, value in results.items():
            try:
                json.dumps(numpy_to_json_serializable(value))
            except Exception as inner_e:
                logger.error(f"Non-serializable data in {key}: {type(value)}")
                logger.error(f"Error: {inner_e}")
                # Try to provide more specific information
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        try:
                            json.dumps(numpy_to_json_serializable(subvalue))
                        except:
                            logger.error(f"  Issue in {key}.{subkey}: {type(subvalue)}")
        raise

    return results

