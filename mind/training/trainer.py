"""
training pipeline ensuring reproducible results.
"""
import time
import numpy as np
import torch
import json
from pathlib import Path
import logging

from mind.evaluation.metrics import evaluate_model
from mind.evaluation.feature_importance import extract_feature_importance

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_model(model_type, model_params, datasets, signal_type, window_size, n_neurons, output_dir, device="cuda"):
    """Train a model and return results."""
    logger.info(f"Training {model_type} on {signal_type}")
    set_seed(model_params.get('random_state', 42))

    # Extract data
    train_dataset = datasets[signal_type]['train']
    val_dataset = datasets[signal_type]['val']
    test_dataset = datasets[signal_type]['test']

    # Convert datasets to arrays
    X_train = torch.stack([x for x, _ in train_dataset])
    y_train = torch.tensor([y.item() for _, y in train_dataset])
    X_val = torch.stack([x for x, _ in val_dataset])
    y_val = torch.tensor([y.item() for _, y in val_dataset])
    X_test = torch.stack([x for x, _ in test_dataset])
    y_test = torch.tensor([y.item() for _, y in test_dataset])

    # Initialize model
    if model_type == 'random_forest':
        from mind.models.classical.random_forest import RandomForestModel
        model = RandomForestModel(**model_params)
    elif model_type == 'svm':
        from mind.models.classical.svm import SVMModel
        model = SVMModel(**model_params)
    elif model_type == 'mlp':
        from mind.models.classical.mlp import MLPModel
        model = MLPModel(**model_params)
    elif model_type == 'fcnn':
        from mind.models.deep.fcnn import FCNNWrapper
        model = FCNNWrapper(window_size, n_neurons, device=device, **model_params)
    elif model_type == 'cnn':
        from mind.models.deep.cnn import CNNWrapper
        model = CNNWrapper(window_size, n_neurons, device=device, **model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Train model
    start_time = time.time()
    model.fit(X_train, y_train, X_val, y_val)
    train_time = time.time() - start_time

    # Evaluate model
    results = evaluate_model(model, X_test, y_test)
    results['train_time'] = train_time

    # Extract feature importance
    importance_matrix = extract_feature_importance(model, window_size, n_neurons)
    results['importance_summary'] = {
        'importance_matrix': importance_matrix.tolist(),
        'temporal_importance': importance_matrix.mean(axis=1).tolist(),
        'neuron_importance': importance_matrix.mean(axis=0).tolist(),
        'top_neuron_indices': np.argsort(importance_matrix.mean(axis=0))[::-1][:20].tolist()
    }

    # Add metadata
    results['metadata'] = {
        'model_name': model_type,
        'signal_type': signal_type,
        'window_size': window_size,
        'n_neurons': n_neurons
    }

    # Save results
    output_path = Path(output_dir) / f"{signal_type}_{model_type}_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_path}")
    return results

