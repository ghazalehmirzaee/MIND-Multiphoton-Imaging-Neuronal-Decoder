"""Deep learning models trainer module."""
import os
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, List, Optional, Tuple, Callable
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from omegaconf import DictConfig

from mind.models.deep.fcnn import create_fcnn
from mind.models.deep.cnn import create_cnn
from mind.training.train import (
    train_model,
    FocalLoss,
    create_dataloaders,
    create_model_name
)
from mind.utils.experiment_tracking import log_metrics, log_artifact
from mind.utils.logging import get_logger

# Set up logger
logger = get_logger(__name__)


def create_optimizer(
        model: nn.Module,
        config: DictConfig
) -> torch.optim.Optimizer:
    """
    Create optimizer for deep learning model.

    Parameters
    ----------
    model : nn.Module
        Model to optimize
    config : DictConfig
        Configuration dictionary

    Returns
    -------
    torch.optim.Optimizer
        Optimizer
    """
    # Extract optimizer parameters
    learning_rate = config.training.learning_rate
    weight_decay = config.training.weight_decay
    optimizer_type = config.training.get('optimizer', 'adamw')

    # Create optimizer based on type
    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == 'sgd':
        momentum = config.training.get('momentum', 0.9)
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        logger.warning(f"Unknown optimizer type: {optimizer_type}, using AdamW")
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    return optimizer


def create_scheduler(
        optimizer: torch.optim.Optimizer,
        config: DictConfig,
        num_samples: int,
        batch_size: int
) -> Callable:
    """
    Create learning rate scheduler factory function.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer
    config : DictConfig
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
    scheduler_type = config.training.get('scheduler', 'onecycle')

    if scheduler_type.lower() == 'onecycle':
        def create_one_cycle_scheduler(optimizer, num_epochs):
            # Calculate number of steps per epoch
            steps_per_epoch = (num_samples + batch_size - 1) // batch_size

            # Create OneCycleLR scheduler
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=config.training.learning_rate,
                epochs=num_epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.3,
                anneal_strategy='cos',
                div_factor=25.0,
                final_div_factor=1e4
            )

            return scheduler

        return create_one_cycle_scheduler

    elif scheduler_type.lower() == 'reduce_on_plateau':
        def create_reduce_on_plateau_scheduler(optimizer, num_epochs):
            # Create ReduceLROnPlateau scheduler
            patience = config.training.get('scheduler_patience', 5)
            factor = config.training.get('scheduler_factor', 0.5)

            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=factor,
                patience=patience,
                verbose=True
            )

            return scheduler

        return create_reduce_on_plateau_scheduler

    elif scheduler_type.lower() == 'cosine':
        def create_cosine_scheduler(optimizer, num_epochs):
            # Create CosineAnnealingLR scheduler
            T_max = config.training.get('cosine_t_max', num_epochs)
            eta_min = config.training.get('cosine_eta_min', 0)

            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=T_max,
                eta_min=eta_min
            )

            return scheduler

        return create_cosine_scheduler

    else:
        logger.warning(f"Unknown scheduler type: {scheduler_type}, using OneCycleLR")
        return create_scheduler(optimizer, config, num_samples, batch_size)


def train_fcnn_model(
        data: Dict[str, Any],
        signal_type: str,
        config: DictConfig,
        device: torch.device,
        wandb_run: Any = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Train a Fully Connected Neural Network model with signal-specific optimizations.
    Modified to ensure binary classification.

    Parameters
    ----------
    data : Dict[str, Any]
        Dictionary containing the processed data
    signal_type : str
        Signal type ('calcium', 'deltaf', or 'deconv')
    config : DictConfig
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
        batch_size=config.training.batch_size,
        window_size=data['window_size'],
        reshape=False  # Use flattened data for FCNN
    )

    # Extract metadata
    input_dim = dataloaders['input_dim']
    n_classes = 2  # Ensure binary classification
    train_loader = dataloaders['train_loader']
    val_loader = dataloaders['val_loader']

    # Create model with signal-specific optimizations
    model = create_fcnn(input_dim, n_classes, config, signal_type)
    model = model.to(device)

    # Create optimizer with adjusted learning rate for deconvolved signals
    if signal_type == 'deconv':
        # Higher learning rate for deconvolved signals
        lr_multiplier = config.training.get('deconv_lr_multiplier', 1.2)
        weight_decay = config.training.get('deconv_weight_decay', 5e-6)
    else:
        lr_multiplier = 1.0
        weight_decay = config.training.weight_decay

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate * lr_multiplier,
        weight_decay=weight_decay
    )

    # Create scheduler factory with adjusted parameters for deconvolved signals
    if signal_type == 'deconv':
        def scheduler_fn(optimizer, num_epochs):
            # Calculate steps per epoch
            steps_per_epoch = (len(
                dataloaders['X_train']) + config.training.batch_size - 1) // config.training.batch_size

            # OneCycleLR with higher max_lr for deconvolved signals
            return optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=config.training.learning_rate * 1.5,  # 50% higher max_lr
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
            config.training.batch_size
        )

    # Get class weights
    class_weights = data.get('class_weights', {}).get(signal_type, None)

    # Create loss function - optimized for binary classification
    if signal_type == 'deconv':
        # Use focal loss for deconvolved signals to better handle class imbalance
        alpha = config.training.get('focal_loss_alpha', 2.0)
        gamma = config.training.get('focal_loss_gamma', 2.0)
        criterion = FocalLoss(alpha=alpha, gamma=gamma)
    elif class_weights is not None:
        # Convert to PyTorch tensor for binary classification
        weight_tensor = torch.ones(2, device=device)
        for cls, weight in class_weights.items():
            cls_idx = int(cls)
            if cls_idx < 2:  # Ensure binary (0 or 1)
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
        config: DictConfig,
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
    config : DictConfig
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
        batch_size=config.training.batch_size,
        window_size=data['window_size'],
        reshape=True  # Use reshaped data for CNN
    )

    # Extract metadata
    window_size = dataloaders['window_size']
    n_neurons = dataloaders['n_neurons']
    n_classes = 2  # Ensure binary classification
    train_loader = dataloaders['train_loader']
    val_loader = dataloaders['val_loader']

    # Create model with signal-specific optimizations
    model = create_cnn(n_neurons, window_size, n_classes, config, signal_type)
    model = model.to(device)

    # Create optimizer with adjusted parameters for deconvolved signals
    if signal_type == 'deconv':
        # Higher learning rate and lower weight decay for deconvolved signals
        lr_multiplier = config.training.get('deconv_lr_multiplier', 1.3)
        weight_decay = config.training.get('deconv_weight_decay', 1e-6)
    else:
        lr_multiplier = 1.0
        weight_decay = config.training.weight_decay

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate * lr_multiplier,
        weight_decay=weight_decay
    )

    # Create scheduler factory with adjusted parameters for deconvolved signals
    if signal_type == 'deconv':
        def scheduler_fn(optimizer, num_epochs):
            # Calculate steps per epoch
            steps_per_epoch = (len(
                dataloaders['X_train']) + config.training.batch_size - 1) // config.training.batch_size

            # OneCycleLR with higher max_lr for deconvolved signals
            return optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=config.training.learning_rate * 1.6,  # 60% higher max_lr
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
            config.training.batch_size
        )

    # Get class weights if available
    class_weights = data.get('class_weights', {}).get(signal_type, None)

    # Create loss function
    if signal_type == 'deconv':
        # Use focal loss with higher alpha for deconvolved signals
        alpha = config.training.get('focal_loss_alpha', 2.5)
        gamma = config.training.get('focal_loss_gamma', 2.0)
        criterion = FocalLoss(alpha=alpha, gamma=gamma)
    elif class_weights is not None:
        # Use focal loss for imbalanced data
        alpha = config.training.get('focal_loss_alpha', 2.0)
        gamma = config.training.get('focal_loss_gamma', 2.0)
        criterion = FocalLoss(alpha=alpha, gamma=gamma)
    else:
        criterion = nn.CrossEntropyLoss()

    # Create model name
    model_name = create_model_name(signal_type, 'cnn')

    # Train model with more epochs for deconvolved signals
    if signal_type == 'deconv':
        # Make a copy of config to increase epochs for deconvolved signal
        epochs_multiplier = config.training.get('deconv_epochs_multiplier', 1.2)
        local_epochs = int(config.training.epochs * epochs_multiplier)
        logger.info(f"Increasing epochs to {local_epochs} for deconvolved signal")
    else:
        local_epochs = config.training.epochs

    # Create configuration for train_model
    train_config = {
        'training': {
            'epochs': local_epochs,
            'early_stopping': config.training.early_stopping
        }
    }

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        config=train_config,
        scheduler_fn=scheduler_fn,
        class_weights=class_weights,
        model_name=model_name,
        wandb_run=wandb_run
    )

    return model, history


def train_all_deep_models(
        data: Dict[str, Any],
        config: DictConfig,
        wandb_run: Any = None
) -> Dict[str, Any]:
    """
    Train all deep learning models on all signal types.

    Parameters
    ----------
    data : Dict[str, Any]
        Dictionary containing the processed data
    config : DictConfig
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
    signal_types = config.data.signal_types

    # Create timestamp for saving models
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Train models for each signal type
    for signal_type in signal_types:
        logger.info(f"Training models for {signal_type} signal")

        # Extract data for this signal type
        X_train_key = f'X_train_{signal_type}'
        y_train_key = f'y_train_{signal_type}'

        if X_train_key not in data or y_train_key not in data:
            logger.warning(f"Training data not found for {signal_type}")
            continue

        try:
            # Train FCNN model
            fcnn_model, fcnn_history = train_fcnn_model(
                data, signal_type, config, device, wandb_run
            )
            results['models'][f"{signal_type}_fcnn"] = fcnn_model
            results['history'][f"{signal_type}_fcnn"] = fcnn_history

            # Save model
            save_path = os.path.join(
                config.experiment.output_dir,
                'models',
                'deep',
                timestamp,
                f"{signal_type}_fcnn.pt"
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(fcnn_model.state_dict(), save_path)
            logger.info(f"Saved FCNN model to {save_path}")

            # Log model artifact to W&B
            if wandb_run is not None:
                log_artifact(wandb_run, save_path, f"{signal_type}_fcnn_model")

        except Exception as e:
            logger.error(f"Error training FCNN for {signal_type}: {e}", exc_info=True)

        try:
            # Train CNN model
            cnn_model, cnn_history = train_cnn_model(
                data, signal_type, config, device, wandb_run
            )
            results['models'][f"{signal_type}_cnn"] = cnn_model
            results['history'][f"{signal_type}_cnn"] = cnn_history

            # Save model
            save_path = os.path.join(
                config.experiment.output_dir,
                'models',
                'deep',
                timestamp,
                f"{signal_type}_cnn.pt"
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(cnn_model.state_dict(), save_path)
            logger.info(f"Saved CNN model to {save_path}")

            # Log model artifact to W&B
            if wandb_run is not None:
                log_artifact(wandb_run, save_path, f"{signal_type}_cnn_model")

        except Exception as e:
            logger.error(f"Error training CNN for {signal_type}: {e}", exc_info=True)

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

    # Ensure binary classification (0=no footstep, 1=contralateral footstep)
    if len(np.unique(all_targets)) > 2:
        logger.warning(f"Converting multi-class targets to binary for {model_name}")
        binary_targets = (all_targets > 0).astype(int)
        all_targets = binary_targets

        binary_preds = (all_preds > 0).astype(int)
        all_preds = binary_preds

    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision_macro = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0)

    # Calculate ROC AUC if applicable
    roc_auc = None
    if all_probs.shape[1] > 1:  # Multi-class case
        try:
            from sklearn.metrics import roc_auc_score
            roc_auc = roc_auc_score(all_targets, all_probs[:, 1])
        except Exception as e:
            logger.warning(f"Could not calculate ROC AUC: {e}")

    # Calculate confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_targets, all_preds)

    # Store metrics
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'predictions': all_preds,
        'probabilities': all_probs,
        'targets': all_targets,
        'confusion_matrix': cm
    }

    if roc_auc is not None:
        metrics['roc_auc'] = roc_auc

    # Log metrics
    logger.info(f"{model_name} evaluation metrics:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision (macro): {precision_macro:.4f}")
    logger.info(f"  Recall (macro): {recall_macro:.4f}")
    logger.info(f"  F1 (macro): {f1_macro:.4f}")
    if roc_auc is not None:
        logger.info(f"  ROC AUC: {roc_auc:.4f}")

    # Log metrics to Weights & Biases
    if wandb_run is not None:
        metrics_to_log = {
            f"{model_name}_accuracy": accuracy,
            f"{model_name}_precision_macro": precision_macro,
            f"{model_name}_recall_macro": recall_macro,
            f"{model_name}_f1_macro": f1_macro
        }
        if roc_auc is not None:
            metrics_to_log[f"{model_name}_roc_auc"] = roc_auc

        log_metrics(wandb_run, metrics_to_log)

    return metrics


def test_deep_models(
        models: Dict[str, nn.Module],
        data: Dict[str, Any],
        config: DictConfig,
        wandb_run: Any = None
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Test trained deep learning models on test data.

    Parameters
    ----------
    models : Dict[str, nn.Module]
        Dictionary containing trained models
    data : Dict[str, Any]
        Dictionary containing the processed data
    config : DictConfig
        Configuration dictionary
    wandb_run : Any, optional
        Weights & Biases run, by default None

    Returns
    -------
    Dict[str, Dict[str, Dict[str, Any]]]
        Dictionary containing test results
    """
    logger.info("Testing deep learning models on test data")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize results dictionary
    test_results = {}

    # Define signal types and model types
    signal_types = config.data.signal_types
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
            try:
                dataloaders = create_dataloaders(
                    data, signal_type,
                    batch_size=config.training.batch_size,
                    window_size=data['window_size'],
                    reshape=reshape
                )

                # Evaluate model
                metrics = evaluate_deep_model(
                    model, dataloaders['test_loader'], device,
                    f"test_{model_key}", wandb_run
                )

                signal_results[model_type] = metrics

            except Exception as e:
                logger.error(f"Error testing {model_key}: {e}", exc_info=True)

        test_results[signal_type] = signal_results

    return test_results


def save_deep_models(
        models: Dict[str, nn.Module],
        output_dir: str = 'models/deep',
        timestamp: Optional[str] = None
) -> None:
    """
    Save trained deep learning models with timestamp.

    Parameters
    ----------
    models : Dict[str, nn.Module]
        Dictionary containing trained models
    output_dir : str, optional
        Output directory, by default 'models/deep'
    timestamp : Optional[str], optional
        Timestamp to include in the output directory, by default None
    """
    # Create timestamped output directory if timestamp is provided
    if timestamp:
        output_dir = os.path.join(output_dir, timestamp)

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
        config: DictConfig,
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
    config : DictConfig
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
            n_classes = 2  # Binary classification

            # Create model
            model = create_fcnn(input_dim, n_classes, config)
        elif model_type == 'cnn':
            # Create dataloaders to get metadata
            dataloaders = create_dataloaders(
                data, signal_type, reshape=True
            )
            window_size = dataloaders['window_size']
            n_neurons = dataloaders['n_neurons']
            n_classes = 2  # Binary classification

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
        output_file: str = 'results/metrics/deep_learning_results.json',
        timestamp: Optional[str] = None
) -> None:
    """
    Save results to JSON file with timestamp.

    Parameters
    ----------
    results : Dict[str, Any]
        Results dictionary
    output_file : str, optional
        Output file path, by default 'results/metrics/deep_learning_results.json'
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

