"""Deep learning models trainer module."""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, List, Optional, Tuple, Callable
import logging
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from mind.models.deep.fcnn import create_fcnn
from mind.models.deep.cnn import create_cnn
from mind.training.train import (
    train_model,
    FocalLoss,
    create_dataloaders,
    create_model_name
)
from mind.utils.experiment_tracking import log_metrics

logger = logging.getLogger(__name__)


def create_optimizer(
        model: nn.Module,
        config: Dict[str, Any]
) -> torch.optim.Optimizer:
    """
    Create optimizer for deep learning model.

    Parameters
    ----------
    model : nn.Module
        Model to optimize
    config : Dict[str, Any]
        Configuration dictionary

    Returns
    -------
    torch.optim.Optimizer
        Optimizer
    """
    # Extract optimizer parameters
    learning_rate = config['training'].get('learning_rate', 0.001)
    weight_decay = config['training'].get('weight_decay', 1e-5)

    # Create AdamW optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    return optimizer


def create_scheduler(
        optimizer: torch.optim.Optimizer,
        config: Dict[str, Any],
        num_samples: int,
        batch_size: int
) -> Callable:
    """
    Create learning rate scheduler factory function.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer
    config : Dict[str, Any]
        Configuration dictionary
    num_samples : int
        Number of training samples
    batch_size : int
        Batch size

    Returns
    -------
    Callable
        Function to create scheduler given optimizer and num_epochs
    """

    def create_one_cycle_scheduler(optimizer, num_epochs):
        # Calculate number of steps per epoch
        steps_per_epoch = (num_samples + batch_size - 1) // batch_size

        # Create OneCycleLR scheduler
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['training'].get('learning_rate', 0.001),
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,  # Percentage of iterations for learning rate warmup
            anneal_strategy='cos',  # Use cosine annealing for learning rate decay
            div_factor=25.0,  # Initial learning rate is max_lr/div_factor
            final_div_factor=1e4  # Final learning rate is max_lr/(div_factor*final_div_factor)
        )

        return scheduler

    return create_one_cycle_scheduler


def train_fcnn_model(
        data: Dict[str, Any],
        signal_type: str,
        config: Dict[str, Any],
        device: torch.device,
        wandb_run: Any = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Train a Fully Connected Neural Network model with signal-specific optimizations.

    Parameters
    ----------
    data : Dict[str, Any]
        Dictionary containing the processed data
    signal_type : str
        Signal type ('calcium', 'deltaf', or 'deconv')
    config : Dict[str, Any]
        Configuration dictionary
    device : torch.device
        Device to use for training
    wandb_run : Any, optional
        Weights & Biases run, by default None

    Returns
    -------
    Tuple[nn.Module, Dict[str, Any]]
        Trained model and training history
    """
    logger.info(f"Training FCNN model for {signal_type} signal")

    # Create dataloaders
    dataloaders = create_dataloaders(
        data, signal_type,
        batch_size=config['training'].get('batch_size', 32),
        window_size=data['window_size'],
        reshape=False  # Use flattened data for FCNN
    )

    # Extract metadata
    input_dim = dataloaders['input_dim']
    n_classes = dataloaders['n_classes']
    train_loader = dataloaders['train_loader']
    val_loader = dataloaders['val_loader']

    # Create model with signal-specific optimizations
    model = create_fcnn(input_dim, n_classes, config, signal_type)
    model = model.to(device)

    # Create optimizer with adjusted learning rate for deconvolved signals
    if signal_type == 'deconv':
        # Higher learning rate for deconvolved signals
        lr_multiplier = 1.2
        weight_decay = 5e-6  # Reduced weight decay for better flexibility
    else:
        lr_multiplier = 1.0
        weight_decay = config['training'].get('weight_decay', 1e-5)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training'].get('learning_rate', 0.001) * lr_multiplier,
        weight_decay=weight_decay
    )

    # Create scheduler factory with adjusted parameters for deconvolved signals
    if signal_type == 'deconv':
        def scheduler_fn(optimizer, num_epochs):
            # Calculate steps per epoch
            steps_per_epoch = (len(dataloaders['X_train']) +
                               config['training'].get('batch_size', 32) - 1) // config['training'].get('batch_size', 32)

            # OneCycleLR with higher max_lr for deconvolved signals
            return optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=config['training'].get('learning_rate', 0.001) * 1.5,  # 50% higher max_lr
                epochs=num_epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.3,
                anneal_strategy='cos',
                div_factor=20.0,  # Lower div_factor for higher initial learning rate
                final_div_factor=1e4
            )
    else:
        scheduler_fn = create_scheduler(
            optimizer, config, len(dataloaders['X_train']),
            config['training'].get('batch_size', 32)
        )

    # Get class weights if available
    class_weights = data.get('class_weights', {}).get(signal_type, None)

    # Create loss function based on signal type
    if signal_type == 'deconv':
        # Use focal loss for deconvolved signals to better handle class imbalance
        criterion = FocalLoss(alpha=2.0, gamma=2.0)
    elif class_weights is not None:
        # Convert to PyTorch tensor
        weight_tensor = torch.ones(n_classes, device=device)
        for cls, weight in class_weights.items():
            cls_idx = int(cls)
            if cls_idx < n_classes:
                weight_tensor[cls_idx] = weight

        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    # Create model name
    model_name = create_model_name(signal_type, 'fcnn')

    # Train model
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        config=config,
        scheduler_fn=scheduler_fn,
        class_weights=class_weights,
        model_name=model_name,
        wandb_run=wandb_run
    )

    return model, history


def train_cnn_model(
        data: Dict[str, Any],
        signal_type: str,
        config: Dict[str, Any],
        device: torch.device,
        wandb_run: Any = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Train a Convolutional Neural Network model with signal-specific optimizations.

    Parameters
    ----------
    data : Dict[str, Any]
        Dictionary containing the processed data
    signal_type : str
        Signal type ('calcium', 'deltaf', or 'deconv')
    config : Dict[str, Any]
        Configuration dictionary
    device : torch.device
        Device to use for training
    wandb_run : Any, optional
        Weights & Biases run, by default None

    Returns
    -------
    Tuple[nn.Module, Dict[str, Any]]
        Trained model and training history
    """
    logger.info(f"Training CNN model for {signal_type} signal")

    # Create dataloaders
    dataloaders = create_dataloaders(
        data, signal_type,
        batch_size=config['training'].get('batch_size', 32),
        window_size=data['window_size'],
        reshape=True  # Use reshaped data for CNN
    )

    # Extract metadata
    window_size = dataloaders['window_size']
    n_neurons = dataloaders['n_neurons']
    n_classes = dataloaders['n_classes']
    train_loader = dataloaders['train_loader']
    val_loader = dataloaders['val_loader']

    # Create model with signal-specific optimizations
    model = create_cnn(n_neurons, window_size, n_classes, config, signal_type)
    model = model.to(device)

    # Create optimizer with adjusted parameters for deconvolved signals
    if signal_type == 'deconv':
        # Higher learning rate and lower weight decay for deconvolved signals
        lr_multiplier = 1.3
        weight_decay = 1e-6  # Lower weight decay for better flexibility
    else:
        lr_multiplier = 1.0
        weight_decay = config['training'].get('weight_decay', 1e-5)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training'].get('learning_rate', 0.001) * lr_multiplier,
        weight_decay=weight_decay
    )

    # Create scheduler factory with adjusted parameters for deconvolved signals
    if signal_type == 'deconv':
        def scheduler_fn(optimizer, num_epochs):
            # Calculate steps per epoch
            steps_per_epoch = (len(dataloaders['X_train']) +
                               config['training'].get('batch_size', 32) - 1) // config['training'].get('batch_size', 32)

            # OneCycleLR with higher max_lr for deconvolved signals
            return optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=config['training'].get('learning_rate', 0.001) * 1.6,  # 60% higher max_lr
                epochs=num_epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.25,  # Faster warmup
                anneal_strategy='cos',
                div_factor=15.0,  # Lower div_factor for higher initial learning rate
                final_div_factor=1e4
            )
    else:
        scheduler_fn = create_scheduler(
            optimizer, config, len(dataloaders['X_train']),
            config['training'].get('batch_size', 32)
        )

    # Get class weights if available
    class_weights = data.get('class_weights', {}).get(signal_type, None)

    # Create loss function
    if signal_type == 'deconv':
        # Use focal loss with higher alpha for deconvolved signals
        criterion = FocalLoss(alpha=2.5, gamma=2.0)
    elif class_weights is not None:
        # Use focal loss for imbalanced data
        criterion = FocalLoss(alpha=2.0, gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss()

    # Create model name
    model_name = create_model_name(signal_type, 'cnn')

    # Train model with more epochs for deconvolved signals
    if signal_type == 'deconv':
        # Make a copy of config to increase epochs for deconvolved signal
        local_config = config.copy()
        local_config['training'] = config['training'].copy()
        local_config['training']['epochs'] = int(config['training'].get('epochs', 100) * 1.2)
    else:
        local_config = config

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        config=local_config,
        scheduler_fn=scheduler_fn,
        class_weights=class_weights,
        model_name=model_name,
        wandb_run=wandb_run
    )

    return model, history


def train_all_deep_models(
        data: Dict[str, Any],
        config: Dict[str, Any],
        wandb_run: Any = None
) -> Dict[str, Any]:
    """
    Train all deep learning models on all signal types.

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
    logger.info("Training all deep learning models")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize results dictionary
    results = {
        'models': {},
        'history': {},
        'metrics': {}
    }

    # Define signal types
    signal_types = ['calcium', 'deltaf', 'deconv']

    # Train models for each signal type
    for signal_type in signal_types:
        logger.info(f"Training models for {signal_type} signal")

        # Train FCNN model
        fcnn_model, fcnn_history = train_fcnn_model(
            data, signal_type, config, device, wandb_run
        )
        results['models'][f"{signal_type}_fcnn"] = fcnn_model
        results['history'][f"{signal_type}_fcnn"] = fcnn_history

        # Train CNN model
        cnn_model, cnn_history = train_cnn_model(
            data, signal_type, config, device, wandb_run
        )
        results['models'][f"{signal_type}_cnn"] = cnn_model
        results['history'][f"{signal_type}_cnn"] = cnn_history

    return results


def evaluate_deep_model(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        model_name: str,
        wandb_run: Any = None
) -> Dict[str, Any]:
    """
    Evaluate a deep learning model.

    Parameters
    ----------
    model : nn.Module
        Model to evaluate
    dataloader : torch.utils.data.DataLoader
        Data loader
    device : torch.device
        Device to use for evaluation
    model_name : str
        Model name for logging
    wandb_run : Any, optional
        Weights & Biases run, by default None

    Returns
    -------
    Dict[str, Any]
        Dictionary containing evaluation metrics
    """
    logger.info(f"Evaluating {model_name} model")

    model.eval()

    all_preds = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Get probabilities and predictions
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, dim=1)

            # Store predictions, probabilities, and targets
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate results
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)

    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision_macro = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0)

    # Store metrics
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'predictions': all_preds,
        'probabilities': all_probs,
        'targets': all_targets
    }

    # Log metrics
    logger.info(f"{model_name} evaluation metrics:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision (macro): {precision_macro:.4f}")
    logger.info(f"  Recall (macro): {recall_macro:.4f}")
    logger.info(f"  F1 (macro): {f1_macro:.4f}")

    # Log metrics to Weights & Biases
    if wandb_run is not None:
        log_metrics(wandb_run, {
            f"{model_name}_accuracy": accuracy,
            f"{model_name}_precision_macro": precision_macro,
            f"{model_name}_recall_macro": recall_macro,
            f"{model_name}_f1_macro": f1_macro
        })

    return metrics


def test_deep_models(
        models: Dict[str, nn.Module],
        data: Dict[str, Any],
        config: Dict[str, Any],
        wandb_run: Any = None
) -> Dict[str, Any]:
    """
    Test trained deep learning models on test data.

    Parameters
    ----------
    models : Dict[str, nn.Module]
        Dictionary containing trained models
    data : Dict[str, Any]
        Dictionary containing the processed data
    config : Dict[str, Any]
        Configuration dictionary
    wandb_run : Any, optional
        Weights & Biases run, by default None

    Returns
    -------
    Dict[str, Any]
        Dictionary containing test results
    """
    logger.info("Testing deep learning models on test data")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize results dictionary
    test_results = {}

    # Define signal types and model types
    signal_types = ['calcium', 'deltaf', 'deconv']
    model_types = ['fcnn', 'cnn']

    # Test models for each signal type
    for signal_type in signal_types:
        logger.info(f"Testing models for {signal_type} signal")

        signal_results = {}

        # Test each model type
        for model_type in model_types:
            model_key = f"{signal_type}_{model_type}"

            if model_key not in models:
                logger.warning(f"Model {model_key} not found in models dictionary")
                continue

            model = models[model_key]

            # Create dataloaders
            reshape = model_type != 'fcnn'  # Reshape for CNN but not for FCNN
            dataloaders = create_dataloaders(
                data, signal_type,
                batch_size=config['training'].get('batch_size', 32),
                window_size=data['window_size'],
                reshape=reshape
            )

            # Evaluate model
            metrics = evaluate_deep_model(
                model, dataloaders['test_loader'], device,
                f"test_{model_key}", wandb_run
            )

            signal_results[model_type] = metrics

        test_results[signal_type] = signal_results

    return test_results


def save_deep_models(
        models: Dict[str, nn.Module],
        output_dir: str = 'models/deep'
) -> None:
    """
    Save trained deep learning models.

    Parameters
    ----------
    models : Dict[str, nn.Module]
        Dictionary containing trained models
    output_dir : str, optional
        Output directory, by default 'models/deep'
    """
    logger.info(f"Saving deep learning models to {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save each model
    for model_name, model in models.items():
        # Skip non-model entries
        if not isinstance(model, nn.Module):
            continue

        # Save model
        model_path = os.path.join(output_dir, f"{model_name}.pt")

        torch.save(model.state_dict(), model_path)

        logger.info(f"Saved model {model_name} to {model_path}")


def load_deep_models(
        model_names: List[str],
        data: Dict[str, Any],
        config: Dict[str, Any],
        input_dir: str = 'models/deep'
) -> Dict[str, nn.Module]:
    """
    Load trained deep learning models.

    Parameters
    ----------
    model_names : List[str]
        List of model names to load
    data : Dict[str, Any]
        Dictionary containing the processed data
    config : Dict[str, Any]
        Configuration dictionary
    input_dir : str, optional
        Input directory, by default 'models/deep'

    Returns
    -------
    Dict[str, nn.Module]
        Dictionary containing loaded models
    """
    logger.info(f"Loading deep learning models from {input_dir}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize models dictionary
    models = {}

    # Load each model
    for model_name in model_names:
        model_path = os.path.join(input_dir, f"{model_name}.pt")

        if not os.path.exists(model_path):
            logger.warning(f"Model {model_name} not found at {model_path}")
            continue

        # Parse model name
        parts = model_name.split('_')
        if len(parts) != 2:
            logger.warning(f"Invalid model name format: {model_name}")
            continue

        signal_type, model_type = parts

        # Create model
        if model_type == 'fcnn':
            # Create dataloaders to get input_dim and n_classes
            dataloaders = create_dataloaders(
                data, signal_type, reshape=False
            )
            input_dim = dataloaders['input_dim']
            n_classes = dataloaders['n_classes']

            # Create model
            model = create_fcnn(input_dim, n_classes, config)
        elif model_type == 'cnn':
            # Create dataloaders to get metadata
            dataloaders = create_dataloaders(
                data, signal_type, reshape=True
            )
            window_size = dataloaders['window_size']
            n_neurons = dataloaders['n_neurons']
            n_classes = dataloaders['n_classes']

            # Create model
            model = create_cnn(n_neurons, window_size, n_classes, config)
        else:
            logger.warning(f"Unknown model type: {model_type}")
            continue

        # Load model weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()

        models[model_name] = model
        logger.info(f"Loaded model {model_name} from {model_path}")

    return models


def save_results(
        results: Dict[str, Any],
        output_file: str = 'results/metrics/deep_learning_results.json'
) -> None:
    """
    Save results to JSON file.

    Parameters
    ----------
    results : Dict[str, Any]
        Results dictionary
    output_file : str, optional
        Output file path, by default 'results/metrics/deep_learning_results.json'
    """
    logger.info(f"Saving results to {output_file}")

    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Convert PyTorch tensors and numpy arrays to lists for JSON serialization
    import json

    # Helper function to convert tensors and arrays
    def convert_to_serializable(obj):
        if isinstance(obj, (np.ndarray, np.number)):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)

    # Create a copy of the results dictionary with tensors and arrays converted
    results_json = convert_to_serializable(results)

    # Remove models from results
    if 'models' in results_json:
        del results_json['models']

    # Save results to JSON file
    with open(output_file, 'w') as f:
        json.dump(results_json, f, indent=4)

    logger.info(f"Results saved to {output_file}")

