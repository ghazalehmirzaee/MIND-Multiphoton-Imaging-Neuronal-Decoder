import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
import logging
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

from mind.utils.experiment_tracking import log_metrics, log_figures

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Implementation of the focal loss from:
    "Focal Loss for Dense Object Detection" by Lin et al.
    https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize FocalLoss.

        Parameters
        ----------
        alpha : float, optional
            Weighting factor, by default 1.0
        gamma : float, optional
            Focusing parameter, by default 2.0
        reduction : str, optional
            Reduction method ('mean', 'sum', 'none'), by default 'mean'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            Predicted logits
        targets : torch.Tensor
            Target labels

        Returns
        -------
        torch.Tensor
            Focal loss
        """
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def train_epoch(
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> Dict[str, float]:
    """
    Train the model for one epoch.

    Parameters
    ----------
    model : nn.Module
        Model to train
    train_loader : torch.utils.data.DataLoader
        Training data loader
    criterion : nn.Module
        Loss function
    optimizer : torch.optim.Optimizer
        Optimizer
    device : torch.device
        Device to use for training
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler], optional
        Learning rate scheduler, by default None

    Returns
    -------
    Dict[str, float]
        Dictionary with training metrics
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in tqdm(train_loader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Update learning rate if using OneCycleLR
        if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        # Calculate metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    # Calculate epoch metrics
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total

    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc
    }


def validate(
        model: nn.Module,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        device: torch.device
) -> Dict[str, float]:
    """
    Validate the model.

    Parameters
    ----------
    model : nn.Module
        Model to validate
    val_loader : torch.utils.data.DataLoader
        Validation data loader
    criterion : nn.Module
        Loss function
    device : torch.device
        Device to use for validation

    Returns
    -------
    Dict[str, float]
        Dictionary with validation metrics
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Validating", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Calculate metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            # Store predictions and targets for additional metrics
            all_preds.append(predicted.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Calculate epoch metrics
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = correct / total

    # Concatenate predictions and targets
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'predictions': all_preds,
        'targets': all_targets
    }


def train_model(
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        config: Dict[str, Any],
        scheduler_fn: Optional[Callable] = None,
        class_weights: Optional[Dict[int, float]] = None,
        model_name: str = "model",
        wandb_run: Any = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Train a model with early stopping.

    Parameters
    ----------
    model : nn.Module
        Model to train
    train_loader : torch.utils.data.DataLoader
        Training data loader
    val_loader : torch.utils.data.DataLoader
        Validation data loader
    criterion : nn.Module
        Loss function
    optimizer : torch.optim.Optimizer
        Optimizer
    device : torch.device
        Device to use for training
    config : Dict[str, Any]
        Configuration dictionary
    scheduler_fn : Optional[Callable], optional
        Function to create learning rate scheduler, by default None
    class_weights : Optional[Dict[int, float]], optional
        Class weights for loss function, by default None
    model_name : str, optional
        Name of the model for saving, by default "model"
    wandb_run : Any, optional
        Weights & Biases run, by default None

    Returns
    -------
    Tuple[nn.Module, Dict[str, Any]]
        Trained model and training history
    """
    # Configuration
    epochs = config['training'].get('epochs', 100)
    patience = config['training'].get('early_stopping', {}).get('patience', 15)
    min_delta = config['training'].get('early_stopping', {}).get('min_delta', 0.001)

    # Initialize learning rate scheduler if provided
    scheduler = None
    if scheduler_fn is not None:
        scheduler = scheduler_fn(optimizer, epochs)

    # Initialize training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    # Initialize early stopping parameters
    best_val_loss = float('inf')
    best_model_state = None
    early_stopping_counter = 0

    # Create directory for checkpoints
    model_dir = os.path.join('models', 'checkpoints')
    os.makedirs(model_dir, exist_ok=True)
    checkpoint_path = os.path.join(model_dir, f"{model_name}.pt")

    # Initialize start time
    start_time = time.time()

    # Training loop
    for epoch in range(epochs):
        # Training phase
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, scheduler)

        # Validation phase
        val_metrics = validate(model, val_loader, criterion, device)

        # Update training history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])

        # Update learning rate based on validation loss (if not OneCycleLR)
        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()

        # Print progress
        elapsed_time = time.time() - start_time
        logger.info(f"Epoch {epoch + 1}/{epochs} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Train Acc: {train_metrics['accuracy']:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Val Acc: {val_metrics['accuracy']:.4f} | "
                    f"Time: {elapsed_time:.2f}s")

        # Log to Weights & Biases
        if wandb_run is not None:
            log_metrics(wandb_run, {
                f"{model_name}_train_loss": train_metrics['loss'],
                f"{model_name}_train_acc": train_metrics['accuracy'],
                f"{model_name}_val_loss": val_metrics['loss'],
                f"{model_name}_val_acc": val_metrics['accuracy'],
                f"{model_name}_learning_rate": optimizer.param_groups[0]['lr']
            })

        # Early stopping
        if val_metrics['loss'] < best_val_loss - min_delta:
            best_val_loss = val_metrics['loss']
            best_model_state = model.state_dict().copy()
            early_stopping_counter = 0

            # Save best model
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Saved best model to {checkpoint_path}")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Plot training history
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    # Plot loss
    ax[0].plot(history['train_loss'], label='Training Loss')
    ax[0].plot(history['val_loss'], label='Validation Loss')
    ax[0].set_title(f'Loss - {model_name}')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    # Plot accuracy
    ax[1].plot(history['train_acc'], label='Training Accuracy')
    ax[1].plot(history['val_acc'], label='Validation Accuracy')
    ax[1].set_title(f'Accuracy - {model_name}')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()

    # Save figure
    plt.tight_layout()
    os.makedirs('results/figures', exist_ok=True)
    fig_path = os.path.join('results', 'figures', f"{model_name}_history.png")
    plt.savefig(fig_path, dpi=300)

    # Log figure to Weights & Biases
    if wandb_run is not None:
        log_figures(wandb_run, {f"{model_name}_history": fig})

    plt.close(fig)

    return model, history






def create_model_name(signal_type: str, model_type: str) -> str:
    """
    Create a standardized model name.

    Parameters
    ----------
    signal_type : str
        Signal type ('calcium', 'deltaf', or 'deconv')
    model_type : str
        Model type ('random_forest', 'svm', 'mlp', 'fcnn', 'cnn')

    Returns
    -------
    str
        Model name
    """
    return f"{signal_type}_{model_type}"


def create_dataloaders(
        data: Dict[str, Any],
        signal_type: str,
        batch_size: int = 32,
        window_size: int = 15,
        reshape: bool = True
) -> Dict[str, Any]:
    """
    Create PyTorch DataLoaders for a specific signal type.

    Parameters
    ----------
    data : Dict[str, Any]
        Dictionary containing the processed data
    signal_type : str
        Type of signal to use ('calcium', 'deltaf', or 'deconv')
    batch_size : int, optional
        Batch size, by default 32
    window_size : int, optional
        Window size, by default 15
    reshape : bool, optional
        Whether to reshape data for CNN/LSTM models, by default True

    Returns
    -------
    Dict[str, Any]
        Dictionary containing train, validation, and test DataLoaders and metadata
    """
    from torch.utils.data import TensorDataset, DataLoader

    # Extract data for the specified signal type
    X_train = data[f'X_train_{signal_type}']
    y_train = data[f'y_train_{signal_type}']
    X_val = data[f'X_val_{signal_type}']
    y_val = data[f'y_val_{signal_type}']
    X_test = data[f'X_test_{signal_type}']
    y_test = data[f'y_test_{signal_type}']

    # Get number of neurons
    n_neurons_key = f'n_{signal_type}_neurons'
    n_neurons = data.get(n_neurons_key, X_train.shape[1] // window_size)

    # Reshape data for CNN/LSTM if requested
    if reshape:
        X_train_reshaped = X_train.reshape(-1, window_size, n_neurons)
        X_val_reshaped = X_val.reshape(-1, window_size, n_neurons)
        X_test_reshaped = X_test.reshape(-1, window_size, n_neurons)
    else:
        X_train_reshaped = X_train
        X_val_reshaped = X_val
        X_test_reshaped = X_test

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_reshaped, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_reshaped, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_reshaped, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Create DataLoaders with pin_memory for faster GPU transfer
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4
    )

    # Get number of classes
    n_classes = len(np.unique(y_train))

    # Return dataloaders and metadata
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'input_dim': X_train.shape[1],  # Flattened dimension for MLP/FCNN
        'window_size': window_size,
        'n_neurons': n_neurons,
        'n_classes': n_classes,
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'X_train_tensor': X_train_tensor,
        'y_train_tensor': y_train_tensor,
        'X_val_tensor': X_val_tensor,
        'y_val_tensor': y_val_tensor,
        'X_test_tensor': X_test_tensor,
        'y_test_tensor': y_test_tensor
    }

