"""Classical ML models trainer module."""
import os
import numpy as np
import pandas as pd
import pickle
from typing import Dict, Any, List, Optional, Tuple
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from mind.models.classical.random_forest import (
    create_random_forest,
    optimize_random_forest,
    extract_feature_importance as rf_extract_feature_importance
)
from mind.models.classical.svm import create_svm, optimize_svm
from mind.models.classical.mlp import (
    create_mlp,
    optimize_mlp,
    extract_feature_importance as mlp_extract_feature_importance
)
from mind.utils.experiment_tracking import log_metrics

logger = logging.getLogger(__name__)

def train_random_forest(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: Dict[str, Any],
        optimize: bool = True,
        class_weights: Optional[Dict[int, float]] = None
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train a Random Forest model efficiently.
    """
    logger.info("Training Random Forest")

    if optimize:
        # Optimize hyperparameters
        model, best_params = optimize_random_forest(X_train, y_train, config, class_weights)
        logger.info(f"Optimized Random Forest parameters: {best_params}")
    else:
        # Create model with default parameters
        model = create_random_forest(config)

        # Apply class weights if provided
        if class_weights is not None:
            model.class_weight = class_weights

        # Train model
        model.fit(X_train, y_train.astype(int))

    # Evaluate model
    y_pred = model.predict(X_val)

    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision_macro = precision_score(y_val, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_val, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_val, y_pred, average='macro', zero_division=0)

    # Log metrics
    logger.info(f"Random Forest validation metrics:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision (macro): {precision_macro:.4f}")
    logger.info(f"  Recall (macro): {recall_macro:.4f}")
    logger.info(f"  F1 (macro): {f1_macro:.4f}")

    # Create metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro
    }

    return model, metrics


def train_svm(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: Dict[str, Any],
        optimize: bool = True,
        class_weights: Optional[Dict[int, float]] = None
) -> Tuple[Any, Dict[str, Any], Optional[Any]]:
    """
    Train a Support Vector Machine model efficiently.
    """
    logger.info("Training SVM")

    pca_transformer = None

    if optimize:
        # Optimize hyperparameters
        model, best_params, pca_transformer = optimize_svm(X_train, y_train, config, class_weights)
        logger.info(f"Optimized SVM parameters: {best_params}")
    else:
        # Create model with default parameters
        model = create_svm(config)

        # Apply class weights if provided
        if class_weights is not None:
            model.class_weight = class_weights

        # Check if PCA should be applied
        svm_params = config['models']['classical']['svm']
        use_pca = svm_params.get('pca', True)

        if use_pca:
            from sklearn.decomposition import PCA

            # Determine number of components - efficiently
            pca_components = svm_params.get('pca_components', 0.95)
            if isinstance(pca_components, float) and pca_components <= 1.0:
                n_components = pca_components
            else:
                n_components = min(100, int(pca_components))

            # Create and fit PCA transformer
            pca_transformer = PCA(n_components=n_components, random_state=config['experiment'].get('seed', 42))
            X_train_pca = pca_transformer.fit_transform(X_train)

            # Train model on transformed data
            model.fit(X_train_pca, y_train.astype(int))
        else:
            # Train model on original data
            model.fit(X_train, y_train.astype(int))

    # Evaluate model
    if pca_transformer is not None:
        X_val_pca = pca_transformer.transform(X_val)
        y_pred = model.predict(X_val_pca)
    else:
        y_pred = model.predict(X_val)

    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision_macro = precision_score(y_val, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_val, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_val, y_pred, average='macro', zero_division=0)

    # Log metrics
    logger.info(f"SVM validation metrics:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision (macro): {precision_macro:.4f}")
    logger.info(f"  Recall (macro): {recall_macro:.4f}")
    logger.info(f"  F1 (macro): {f1_macro:.4f}")

    # Create metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro
    }

    return model, metrics, pca_transformer


def train_mlp(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: Dict[str, Any],
        optimize: bool = True
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train a Multilayer Perceptron model efficiently.
    """
    logger.info("Training MLP")

    if optimize:
        # Optimize hyperparameters
        model, best_params = optimize_mlp(X_train, y_train, config)
        logger.info(f"Optimized MLP parameters: {best_params}")
    else:
        # Create model with default parameters
        model = create_mlp(config)

        # Train model
        model.fit(X_train, y_train.astype(int))

    # Evaluate model
    y_pred = model.predict(X_val)

    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision_macro = precision_score(y_val, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_val, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_val, y_pred, average='macro', zero_division=0)

    # Log metrics
    logger.info(f"MLP validation metrics:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision (macro): {precision_macro:.4f}")
    logger.info(f"  Recall (macro): {recall_macro:.4f}")
    logger.info(f"  F1 (macro): {f1_macro:.4f}")

    # Create metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro
    }

    return model, metrics

def train_all_classical_models(
        data: Dict[str, Any],
        config: Dict[str, Any],
        wandb_run: Any = None
) -> Dict[str, Any]:
    """
    Train all classical machine learning models on all signal types.

    Parameters
    ----------
    data : Dict[str, Any]
        Dictionary containing the processed data
    config : Dict[str, Any]
        Configuration dictionary
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
    signal_types = ['calcium', 'deltaf', 'deconv']
    model_types = ['random_forest', 'svm', 'mlp']

    # Train models for each signal type
    for signal_type in signal_types:
        logger.info(f"Training models for {signal_type} signal")

        # Extract data for this signal type
        X_train = data[f'X_train_{signal_type}']
        y_train = data[f'y_train_{signal_type}']
        X_val = data[f'X_val_{signal_type}']
        y_val = data[f'y_val_{signal_type}']

        # Get class weights if available
        class_weights = data.get('class_weights', {}).get(signal_type, None)

        # Get window size and number of neurons
        window_size = data['window_size']
        n_neurons_key = f'n_{signal_type}_neurons'
        n_neurons = data.get(n_neurons_key, X_train.shape[1] // window_size)

        # Train Random Forest
        rf_model, rf_metrics = train_random_forest(
            X_train, y_train, X_val, y_val, config,
            optimize=True, class_weights=class_weights
        )
        results['models'][f"{signal_type}_random_forest"] = rf_model
        results['metrics'][f"{signal_type}_random_forest"] = rf_metrics

        # Extract and store Random Forest feature importance
        rf_importance_2d, rf_temporal, rf_neuron = rf_extract_feature_importance(
            rf_model, window_size, n_neurons
        )
        results['feature_importance'][f"{signal_type}_rf"] = {
            'importance_2d': rf_importance_2d,
            'temporal_importance': rf_temporal,
            'neuron_importance': rf_neuron
        }

        # Train SVM
        svm_model, svm_metrics, pca_transformer = train_svm(
            X_train, y_train, X_val, y_val, config,
            optimize=True, class_weights=class_weights
        )
        results['models'][f"{signal_type}_svm"] = svm_model
        results['metrics'][f"{signal_type}_svm"] = svm_metrics
        if pca_transformer is not None:
            results['models'][f"{signal_type}_svm_pca"] = pca_transformer

        # Train MLP
        mlp_model, mlp_metrics = train_mlp(
            X_train, y_train, X_val, y_val, config, optimize=True
        )
        results['models'][f"{signal_type}_mlp"] = mlp_model
        results['metrics'][f"{signal_type}_mlp"] = mlp_metrics

        # Extract and store MLP feature importance
        mlp_importance_2d, mlp_temporal, mlp_neuron = mlp_extract_feature_importance(
            mlp_model, window_size, n_neurons
        )
        results['feature_importance'][f"{signal_type}_mlp"] = {
            'importance_2d': mlp_importance_2d,
            'temporal_importance': mlp_temporal,
            'neuron_importance': mlp_neuron
        }

        # Log metrics to Weights & Biases
        if wandb_run is not None:
            log_metrics(wandb_run, {
                f"{signal_type}_random_forest_accuracy": rf_metrics['accuracy'],
                f"{signal_type}_random_forest_f1_macro": rf_metrics['f1_macro'],
                f"{signal_type}_svm_accuracy": svm_metrics['accuracy'],
                f"{signal_type}_svm_f1_macro": svm_metrics['f1_macro'],
                f"{signal_type}_mlp_accuracy": mlp_metrics['accuracy'],
                f"{signal_type}_mlp_f1_macro": mlp_metrics['f1_macro']
            })

    return results


def test_classical_models(
        models: Dict[str, Any],
        data: Dict[str, Any],
        wandb_run: Any = None
) -> Dict[str, Any]:
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
    Dict[str, Any]
        Dictionary containing test results
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
        X_test = data[f'X_test_{signal_type}']
        y_test = data[f'y_test_{signal_type}']

        # Test each model type
        for model_type in model_types:
            model_key = f"{signal_type}_{model_type}"

            if model_key not in models:
                logger.warning(f"Model {model_key} not found in models dictionary")
                continue

            model = models[model_key]

            # Apply PCA transformation for SVM if available
            if model_type == 'svm':
                pca_key = f"{signal_type}_svm_pca"
                if pca_key in models:
                    pca = models[pca_key]
                    X_test_transformed = pca.transform(X_test)
                    y_pred = model.predict(X_test_transformed)
                else:
                    y_pred = model.predict(X_test)
            else:
                y_pred = model.predict(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
            recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

            # Store metrics
            metrics = {
                'accuracy': accuracy,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro,
                'f1_macro': f1_macro,
                'predictions': y_pred,
                'targets': y_test
            }

            signal_results[model_type] = metrics

            # Log metrics
            logger.info(f"{model_key} test metrics:")
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  Precision (macro): {precision_macro:.4f}")
            logger.info(f"  Recall (macro): {recall_macro:.4f}")
            logger.info(f"  F1 (macro): {f1_macro:.4f}")

            # Log metrics to Weights & Biases
            if wandb_run is not None:
                log_metrics(wandb_run, {
                    f"test_{model_key}_accuracy": accuracy,
                    f"test_{model_key}_precision_macro": precision_macro,
                    f"test_{model_key}_recall_macro": recall_macro,
                    f"test_{model_key}_f1_macro": f1_macro
                })

        test_results[signal_type] = signal_results

    return test_results


def save_classical_models(
        models: Dict[str, Any],
        output_dir: str = 'models/classical'
) -> None:
    """
    Save trained classical models.

    Parameters
    ----------
    models : Dict[str, Any]
        Dictionary containing trained models
    output_dir : str, optional
        Output directory, by default 'models/classical'
    """
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
        output_file: str = 'results/metrics/classical_ml_results.json'
) -> None:
    """
    Save results to JSON file.

    Parameters
    ----------
    results : Dict[str, Any]
        Results dictionary
    output_file : str, optional
        Output file path, by default 'results/metrics/classical_ml_results.json'
    """
    logger.info(f"Saving results to {output_file}")

    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Convert numpy arrays to lists
    results_json = {}

    for key, value in results.items():
        if isinstance(value, dict):
            results_json[key] = {}
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    results_json[key][k] = v.tolist()
                elif isinstance(v, dict):
                    results_json[key][k] = {}
                    for kk, vv in v.items():
                        if isinstance(vv, np.ndarray):
                            results_json[key][k][kk] = vv.tolist()
                        else:
                            results_json[key][k][kk] = vv
                else:
                    results_json[key][k] = v
        elif isinstance(value, np.ndarray):
            results_json[key] = value.tolist()
        else:
            results_json[key] = value

    # Remove models from results
    if 'models' in results_json:
        del results_json['models']

    # Save results to JSON file
    import json
    with open(output_file, 'w') as f:
        json.dump(results_json, f, indent=4)

    logger.info(f"Results saved to {output_file}")

