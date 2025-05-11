# mind/training/train.py
import numpy as np
from typing import Dict, Any, Optional, Type, List
import time
import os

# Import model classes
from mind.models.classical.random_forest import RandomForestModel
from mind.models.classical.svm import SVMModel
from mind.models.classical.mlp import MLPModel
from mind.models.deep.fcnn import FCNNModel
from mind.models.deep.cnn import CNNModel

# Define model classes dictionary
MODEL_CLASSES = {
    'random_forest': RandomForestModel,
    'svm': SVMModel,
    'mlp': MLPModel,
    'fcnn': FCNNModel,
    'cnn': CNNModel
}


def train_model(model_name: str,
                model_params: Dict[str, Any],
                X_train: np.ndarray,
                y_train: np.ndarray,
                X_val: Optional[np.ndarray] = None,
                y_val: Optional[np.ndarray] = None,
                output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Train a specified model with given parameters.

    Parameters
    ----------
    model_name : str
        Name of the model to train
    model_params : Dict[str, Any]
        Parameters for model initialization
    X_train : np.ndarray
        Training data
    y_train : np.ndarray
        Training labels
    X_val : np.ndarray, optional
        Validation data
    y_val : np.ndarray, optional
        Validation labels
    output_dir : str, optional
        Directory to save model and results

    Returns
    -------
    Dict[str, Any]
        Dictionary containing trained model, metrics, and other results
    """
    # Check if model_name is valid
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Invalid model name: {model_name}. "
                         f"Valid model names are: {list(MODEL_CLASSES.keys())}")

    # Initialize model
    ModelClass = MODEL_CLASSES[model_name]
    model = ModelClass(**model_params)

    # Print model information
    print(f"Training {model_name.upper()} model...")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")
    if X_val is not None and y_val is not None:
        print(f"  X_val shape: {X_val.shape}")
        print(f"  y_val shape: {y_val.shape}")

    # Start timing
    start_time = time.time()

    # Train model
    metrics = model.train(X_train, y_train, X_val, y_val)

    # End timing
    end_time = time.time()
    training_time = end_time - start_time
    print(f"  Training time: {training_time:.2f} seconds")

    # Save model if output_dir is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, f"{model_name}.model")
        model.save(model_path)
        print(f"  Model saved to {model_path}")

    # Print metric summary
    print("  Training metrics:")
    for key, val in metrics.items():
        if isinstance(val, (int, float)):
            print(f"    {key}: {val:.4f}")
        elif isinstance(val, (list, np.ndarray)) and len(val) > 0:
            print(f"    {key}: {val[-1]:.4f} (final)")

    # Return results
    results = {
        'model': model,
        'metrics': metrics,
        'training_time': training_time
    }

    return results


def train_multiple_models(model_configs: List[Dict[str, Any]],
                          X_train: np.ndarray,
                          y_train: np.ndarray,
                          X_val: Optional[np.ndarray] = None,
                          y_val: Optional[np.ndarray] = None,
                          output_dir: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Train multiple models specified in model_configs.

    Parameters
    ----------
    model_configs : List[Dict[str, Any]]
        List of dictionaries, each containing 'name' and 'params' keys
    X_train : np.ndarray
        Training data
    y_train : np.ndarray
        Training labels
    X_val : np.ndarray, optional
        Validation data
    y_val : np.ndarray, optional
        Validation labels
    output_dir : str, optional
        Directory to save models and results

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary mapping model names to their results
    """
    results = {}

    for config in model_configs:
        model_name = config['name']
        model_params = config['params']

        print(f"\nTraining model: {model_name}")

        # Create model-specific output directory if needed
        model_output_dir = None
        if output_dir is not None:
            model_output_dir = os.path.join(output_dir, model_name)
            os.makedirs(model_output_dir, exist_ok=True)

        # Train model
        model_results = train_model(
            model_name, model_params,
            X_train, y_train, X_val, y_val,
            model_output_dir
        )

        # Store results
        results[model_name] = model_results

    return results


def evaluate_model(model: Any,
                   X: np.ndarray,
                   y: np.ndarray,
                   dataset_name: str = "Test") -> Dict[str, float]:
    """
    Evaluate a trained model on the given dataset.

    Parameters
    ----------
    model : Any
        Trained model object
    X : np.ndarray
        Input data
    y : np.ndarray
        True labels
    dataset_name : str, optional
        Name of the dataset (for printing)

    Returns
    -------
    Dict[str, float]
        Dictionary containing evaluation metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    # Make predictions
    y_pred = model.predict(X)

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, average='weighted'),
        'recall': recall_score(y, y_pred, average='weighted'),
        'f1_score': f1_score(y, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y, y_pred)
    }

    # Print metrics
    print(f"{dataset_name} set evaluation:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    print(f"  Confusion Matrix:")
    print(metrics['confusion_matrix'])

    return metrics


def evaluate_multiple_models(models: Dict[str, Any],
                             X: np.ndarray,
                             y: np.ndarray,
                             dataset_name: str = "Test") -> Dict[str, Dict[str, float]]:
    """
    Evaluate multiple trained models on the given dataset.

    Parameters
    ----------
    models : Dict[str, Any]
        Dictionary mapping model names to trained model objects
    X : np.ndarray
        Input data
    y : np.ndarray
        True labels
    dataset_name : str, optional
        Name of the dataset (for printing)

    Returns
    -------
    Dict[str, Dict[str, float]]
        Dictionary mapping model names to their evaluation metrics
    """
    results = {}

    for model_name, model_data in models.items():
        model = model_data['model']

        print(f"\nEvaluating model: {model_name}")
        metrics = evaluate_model(model, X, y, dataset_name)

        # Store results
        results[model_name] = metrics

    return results

