"""Classical ML models trainer module."""
import os
import numpy as np
import pandas as pd
import pickle
from typing import Dict, Any, List, Optional, Tuple
import logging
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from mind.models.classical.random_forest import (
    create_random_forest,
    train_random_forest,
    extract_feature_importance as rf_extract_feature_importance
)
from mind.models.classical.svm import (
    create_svm,
    train_svm
)
from mind.models.classical.mlp import (
    create_mlp,
    train_mlp,
    extract_feature_importance as mlp_extract_feature_importance
)
from mind.utils.experiment_tracking import log_metrics, log_artifact
from mind.utils.logging import get_logger

# Set up logger
logger = get_logger(__name__)


def train_all_classical_models(
        data: Dict[str, Any],
        config: DictConfig,
        wandb_run: Any = None
) -> Dict[str, Any]:
    """
    Train all classical machine learning models on all signal types with enhanced
    data validation and performance tracking.

    Parameters
    ----------
    data : Dict[str, Any]
        Dictionary containing the processed data
    config : DictConfig
        Configuration dictionary (Hydra format)
    wandb_run : Any, optional
        Weights & Biases run, by default None

    Returns
    -------
    Dict[str, Any]
        Dictionary containing trained models and results
    """
    logger.info("Training all classical machine learning models")

    # Initialize results dictionary
    results = {
        'models': {},
        'metrics': {},
        'feature_importance': {}
    }

    # Define signal types and model types
    signal_types = config.data.signal_types
    model_types = config.models.classical_types

    # Extract experiment seed
    seed = config.experiment.seed

    # Train models for each signal type
    for signal_type in signal_types:
        logger.info(f"Training models for {signal_type} signal")

        # Extract data for this signal type
        X_train_key = f'X_train_{signal_type}'
        y_train_key = f'y_train_{signal_type}'
        X_val_key = f'X_val_{signal_type}'
        y_val_key = f'y_val_{signal_type}'

        # Check if data exists
        if not all(k in data for k in [X_train_key, y_train_key, X_val_key, y_val_key]):
            logger.error(f"Missing required data keys for {signal_type}")
            continue

        X_train = data[X_train_key]
        y_train = data[y_train_key]
        X_val = data[X_val_key]
        y_val = data[y_val_key]

        # Validate data types and handle potential issues
        try:
            X_train, y_train, X_val, y_val = _validate_and_process_data(
                X_train, y_train, X_val, y_val, signal_type
            )
        except ValueError as e:
            logger.error(f"Error validating data for {signal_type}: {e}")
            continue

        # Get class weights if available
        class_weights = data.get('class_weights', {}).get(signal_type, None)

        # Get window size and number of neurons
        window_size = data['window_size']
        n_neurons_key = f'n_{signal_type}_neurons'
        n_neurons = data.get(n_neurons_key, X_train.shape[1] // window_size)

        # Train each model type with proper error handling
        for model_type in model_types:
            try:
                logger.info(f"Training {model_type} for {signal_type} data")

                # Use appropriate training function for each model type
                if model_type == 'random_forest':
                    model, metrics = train_random_forest(
                        X_train, y_train, X_val, y_val, config,
                        optimize=config.training.optimize_hyperparams,
                        class_weights=class_weights,
                        signal_type=signal_type
                    )

                    # Extract feature importance
                    importance_2d, temporal_importance, neuron_importance = rf_extract_feature_importance(
                        model, window_size, n_neurons
                    )

                    # Store feature importance
                    results['feature_importance'][f"{signal_type}_rf"] = {
                        'importance_2d': importance_2d,
                        'temporal_importance': temporal_importance,
                        'neuron_importance': neuron_importance
                    }

                elif model_type == 'svm':
                    model, metrics, pca_transformer = train_svm(
                        X_train, y_train, X_val, y_val, config,
                        optimize=config.training.optimize_hyperparams,
                        class_weights=class_weights,
                        signal_type=signal_type
                    )

                    # Store PCA transformer if used
                    if pca_transformer is not None:
                        results['models'][f"{signal_type}_svm_pca"] = pca_transformer

                elif model_type == 'mlp':
                    model, metrics = train_mlp(
                        X_train, y_train, X_val, y_val, config,
                        optimize=config.training.optimize_hyperparams,
                        signal_type=signal_type
                    )

                    # Extract feature importance
                    importance_2d, temporal_importance, neuron_importance = mlp_extract_feature_importance(
                        model, window_size, n_neurons
                    )

                    # Store feature importance
                    results['feature_importance'][f"{signal_type}_mlp"] = {
                        'importance_2d': importance_2d,
                        'temporal_importance': temporal_importance,
                        'neuron_importance': neuron_importance
                    }

                # Store model and metrics
                results['models'][f"{signal_type}_{model_type}"] = model
                results['metrics'][f"{signal_type}_{model_type}"] = metrics

                # Log metrics to WandB if available
                if wandb_run is not None:
                    log_metrics(wandb_run, {
                        f"{signal_type}_{model_type}_val_accuracy": metrics['accuracy'],
                        f"{signal_type}_{model_type}_val_precision": metrics['precision_macro'],
                        f"{signal_type}_{model_type}_val_recall": metrics['recall_macro'],
                        f"{signal_type}_{model_type}_val_f1": metrics['f1_macro']
                    })

            except Exception as e:
                logger.error(f"Error training {model_type} for {signal_type}: {e}", exc_info=True)

    return results


def _validate_and_process_data(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        signal_type: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Validate and process data for training.

    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    X_val : np.ndarray
        Validation features
    y_val : np.ndarray
        Validation labels
    signal_type : str
        Signal type

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Processed X_train, y_train, X_val, y_val
    """
    # Validate data types
    for name, data in [("X_train", X_train), ("X_val", X_val),
                       ("y_train", y_train), ("y_val", y_val)]:
        if not isinstance(data, np.ndarray):
            raise ValueError(f"{name} for {signal_type} is not a numpy array: {type(data)}")

    # Check for NaN values
    if np.isnan(X_train).any():
        logger.warning(f"X_train for {signal_type} contains NaN values, replacing with zeros")
        X_train = np.nan_to_num(X_train, nan=0.0)

    if np.isnan(X_val).any():
        logger.warning(f"X_val for {signal_type} contains NaN values, replacing with zeros")
        X_val = np.nan_to_num(X_val, nan=0.0)

    # Ensure labels are integers for classification
    y_train = y_train.astype(int)
    y_val = y_val.astype(int)

    # Check class distribution
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_val, counts_val = np.unique(y_val, return_counts=True)

    logger.info(f"{signal_type} training set class distribution: {dict(zip(unique_train, counts_train))}")
    logger.info(f"{signal_type} validation set class distribution: {dict(zip(unique_val, counts_val))}")

    # Ensure binary classification (0=no footstep, 1=contralateral footstep)
    if len(unique_train) > 2 or len(unique_val) > 2:
        logger.warning(f"Converting multi-class to binary classification for {signal_type}")
        # Convert any non-zero class to 1 (contralateral footstep)
        y_train = (y_train > 0).astype(int)
        y_val = (y_val > 0).astype(int)

    return X_train, y_train, X_val, y_val


def test_classical_models(
        models: Dict[str, Any],
        data: Dict[str, Any],
        wandb_run: Any = None
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Test trained classical machine learning models on test data.

    Parameters
    ----------
    models : Dict[str, Any]
        Dictionary containing trained models
    data : Dict[str, Any]
        Dictionary containing the processed data
    wandb_run : Any, optional
        Weights & Biases run, by default None

    Returns
    -------
    Dict[str, Dict[str, Dict[str, Any]]]
        Dictionary containing test results organized by signal type and model type
    """
    logger.info("Testing classical models on test data")

    # Initialize results dictionary
    test_results = {}

    # Define signal types and model types
    signal_types = ['calcium', 'deltaf', 'deconv']
    model_types = ['random_forest', 'svm', 'mlp']

    # Test models for each signal type
    for signal_type in signal_types:
        logger.info(f"Testing models for {signal_type} signal")

        signal_results = {}

        # Extract test data for this signal type
        test_data_key = f'X_test_{signal_type}'
        test_labels_key = f'y_test_{signal_type}'

        if test_data_key not in data or test_labels_key not in data:
            logger.warning(f"Test data not found for {signal_type}")
            continue

        X_test = data[test_data_key]
        y_test = data[test_labels_key]

        # Handle NaN values
        if np.isnan(X_test).any():
            logger.warning(f"X_test for {signal_type} contains NaN values, replacing with zeros")
            X_test = np.nan_to_num(X_test, nan=0.0)

        # Ensure binary classification
        if len(np.unique(y_test)) > 2:
            logger.warning(f"Converting multi-class test labels to binary for {signal_type}")
            y_test = (y_test > 0).astype(int)

        # Test each model type
        for model_type in model_types:
            model_key = f"{signal_type}_{model_type}"

            if model_key not in models:
                logger.warning(f"Model {model_key} not found in models dictionary")
                continue

            model = models[model_key]

            try:
                # Apply PCA transformation for SVM if available
                if model_type == 'svm':
                    pca_key = f"{signal_type}_svm_pca"
                    if pca_key in models:
                        pca = models[pca_key]
                        X_test_transformed = pca.transform(X_test)
                        y_pred = model.predict(X_test_transformed)

                        # Get probabilities if available
                        try:
                            y_prob = model.predict_proba(X_test_transformed)
                        except:
                            y_prob = None
                    else:
                        y_pred = model.predict(X_test)

                        # Get probabilities if available
                        try:
                            y_prob = model.predict_proba(X_test)
                        except:
                            y_prob = None
                else:
                    y_pred = model.predict(X_test)

                    # Get probabilities if available
                    try:
                        y_prob = model.predict_proba(X_test)
                    except:
                        y_prob = None

                # Calculate classification metrics
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
                    'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
                    'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
                    'predictions': y_pred,
                    'targets': y_test
                }

                # Calculate ROC AUC if probabilities are available
                if y_prob is not None:
                    metrics['probabilities'] = y_prob

                    # Calculate ROC AUC for binary classification
                    if y_prob.shape[1] == 2:
                        metrics['roc_auc'] = roc_auc_score(y_test, y_prob[:, 1])

                # Calculate confusion matrix
                metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)

                # Store metrics
                signal_results[model_type] = metrics

                # Log metrics
                logger.info(f"{model_key} test metrics:")
                logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
                logger.info(f"  Precision (macro): {metrics['precision_macro']:.4f}")
                logger.info(f"  Recall (macro): {metrics['recall_macro']:.4f}")
                logger.info(f"  F1 (macro): {metrics['f1_macro']:.4f}")

                # Log metrics to Weights & Biases
                if wandb_run is not None:
                    log_metrics(wandb_run, {
                        f"test_{model_key}_accuracy": metrics['accuracy'],
                        f"test_{model_key}_precision_macro": metrics['precision_macro'],
                        f"test_{model_key}_recall_macro": metrics['recall_macro'],
                        f"test_{model_key}_f1_macro": metrics['f1_macro']
                    })

                    if 'roc_auc' in metrics:
                        log_metrics(wandb_run, {f"test_{model_key}_roc_auc": metrics['roc_auc']})

            except Exception as e:
                logger.error(f"Error testing {model_key}: {e}", exc_info=True)

        test_results[signal_type] = signal_results

    return test_results


def save_classical_models(
        models: Dict[str, Any],
        output_dir: str = 'models/classical',
        timestamp: Optional[str] = None
) -> None:
    """
    Save trained classical models with timestamp.

    Parameters
    ----------
    models : Dict[str, Any]
        Dictionary containing trained models
    output_dir : str, optional
        Output directory, by default 'models/classical'
    timestamp : Optional[str], optional
        Timestamp to include in the output directory, by default None
    """
    # Create timestamped output directory if timestamp is provided
    if timestamp:
        output_dir = os.path.join(output_dir, timestamp)

    logger.info(f"Saving classical models to {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save each model
    for model_name, model in models.items():
        # Skip non-model entries
        if not hasattr(model, 'predict'):
            continue

        # Save model
        model_path = os.path.join(output_dir, f"{model_name}.pkl")

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        logger.info(f"Saved model {model_name} to {model_path}")


def load_classical_models(
        model_names: List[str],
        input_dir: str = 'models/classical'
) -> Dict[str, Any]:
    """
    Load trained classical models.

    Parameters
    ----------
    model_names : List[str]
        List of model names to load
    input_dir : str, optional
        Input directory, by default 'models/classical'

    Returns
    -------
    Dict[str, Any]
        Dictionary containing loaded models
    """
    logger.info(f"Loading classical models from {input_dir}")

    # Initialize models dictionary
    models = {}

    # Load each model
    for model_name in model_names:
        model_path = os.path.join(input_dir, f"{model_name}.pkl")

        if not os.path.exists(model_path):
            logger.warning(f"Model {model_name} not found at {model_path}")
            continue

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        models[model_name] = model
        logger.info(f"Loaded model {model_name} from {model_path}")

    return models


def save_results(
        results: Dict[str, Any],
        output_file: str = 'results/metrics/classical_ml_results.json',
        timestamp: Optional[str] = None
) -> None:
    """
    Save results to JSON file with timestamp.

    Parameters
    ----------
    results : Dict[str, Any]
        Results dictionary
    output_file : str, optional
        Output file path, by default 'results/metrics/classical_ml_results.json'
    timestamp : Optional[str], optional
        Timestamp to include in the output file name, by default None
    """
    # Add timestamp to output file name if provided
    if timestamp:
        output_file = os.path.join(
            os.path.dirname(output_file),
            f"{os.path.splitext(os.path.basename(output_file))[0]}_{timestamp}.json"
        )

    logger.info(f"Saving results to {output_file}")

    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Define a function to convert NumPy types to Python native types
    def convert_numpy_to_python(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy_to_python(i) for i in obj]
        else:
            return obj

    # Convert numpy arrays and types to Python native types
    results_json = convert_numpy_to_python(results)

    # Remove models from results
    if 'models' in results_json:
        del results_json['models']

    # Save results to JSON file
    import json
    with open(output_file, 'w') as f:
        json.dump(results_json, f, indent=4)

    logger.info(f"Results saved to {output_file}")

