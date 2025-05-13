"""
Unified training module for all model types.
"""
import time
import numpy as np
import json
from pathlib import Path
import logging
import wandb
import torch

from mind.models.classical.random_forest import RandomForestModel
from mind.models.classical.svm import SVMModel
from mind.models.classical.mlp import MLPModel
from mind.models.deep.fcnn import FCNNWrapper
from mind.models.deep.cnn import CNNWrapper
from mind.evaluation.metrics import evaluate_model
from mind.evaluation.feature_importance import extract_feature_importance, create_importance_summary
from mind.training.train import get_train_val_test_data

logger = logging.getLogger(__name__)


def train_model(model_type, model_params, datasets, signal_type, window_size, n_neurons,
                output_dir, device="cuda", optimize_hyperparams=False, use_wandb=True):
    """
    Unified training function for all model types.

    Parameters
    ----------
    model_type : str
        Type of model ('random_forest', 'svm', 'mlp', 'fcnn', 'cnn')
    model_params : dict
        Model parameters
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
        Dictionary of results
    """
    logger.info(f"Training {model_type} on {signal_type}")

    # Extract data
    X_train, y_train, X_val, y_val, X_test, y_test = get_train_val_test_data(datasets, signal_type)

    # Initialize model
    try:
        if model_type == 'random_forest':
            model_params = model_params.copy()  # Create a copy to avoid modifying the original
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

        logger.info(f"Initialized {model_type} model")

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
            wandb.log(results.get('metrics', {}))

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
        if use_wandb:
            wandb.log({
                "temporal_importance": wandb.Histogram(importance_summary['temporal_importance']),
                "neuron_importance": wandb.Histogram(importance_summary['neuron_importance'])
            })

            # Save importance heatmap as a figure
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 8))
            import seaborn as sns
            sns.heatmap(importance_matrix, cmap='viridis', ax=ax)
            ax.set_xlabel("Neuron")
            ax.set_ylabel("Time Step")
            ax.set_title(f"Feature Importance - {model_type} - {signal_type}")

            # Log to W&B
            wandb.log({"importance_heatmap": wandb})
            plt.close(fig)

        logger.info("Feature importance analysis complete")

    except Exception as e:
        logger.warning(f"Could not extract feature importance: {e}")

    # Save results
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{signal_type}_{model_type}_{timestamp}.json"

        # Convert numpy arrays to lists for JSON serialization
        results_json = {}
        for key, value in results.items():
            if isinstance(value, dict):
                results_json[key] = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, np.ndarray):
                        results_json[key][subkey] = subvalue.tolist()
                    else:
                        results_json[key][subkey] = subvalue
            elif isinstance(value, np.ndarray):
                results_json[key] = value.tolist()
            else:
                results_json[key] = value

        # Add metadata
        results_json['metadata'] = {
            'model_name': model_type,
            'signal_type': signal_type,
            'timestamp': timestamp
        }

        with open(output_dir / filename, 'w') as f:
            json.dump(results_json, f, indent=2)

        logger.info(f"Results saved to {output_dir / filename}")

    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise

    return results

