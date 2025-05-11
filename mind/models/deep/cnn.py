# mind/models/deep/cnn.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional, List, Tuple


class CNN(nn.Module):
    """
    1D Convolutional Neural Network model for neural signal classification.
    """

    def __init__(self,
                 input_size: int,
                 window_size: int,
                 num_neurons: int,
                 filter_sizes: List[int] = [64, 128, 256],
                 kernel_size: int = 3,
                 output_size: int = 2,
                 use_residual: bool = True,
                 use_batch_norm: bool = True):
        """
        Initialize CNN model.

        Parameters
        ----------
        input_size : int
            Size of input features
        window_size : int
            Size of the sliding window
        num_neurons : int
            Number of neurons in the input
        filter_sizes : List[int], optional
            Number of filters in each convolutional layer
        kernel_size : int, optional
            Size of convolutional kernel
        output_size : int, optional
            Number of output classes
        use_residual : bool, optional
            Whether to use residual connections
        use_batch_norm : bool, optional
            Whether to use batch normalization
        """
        super(CNN, self).__init__()

        self.window_size = window_size
        self.num_neurons = num_neurons
        self.use_residual = use_residual

        # Reshape input to (batch_size, 1, window_size, num_neurons)
        # for 2D convolution along time and neurons

        # Create convolutional layers
        self.conv_layers = nn.ModuleList()
        input_channels = 1

        for i, num_filters in enumerate(filter_sizes):
            # Convolutional layer
            conv_layer = nn.Conv1d(
                in_channels=input_channels if i == 0 else filter_sizes[i - 1],
                out_channels=num_filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2  # Same padding
            )

            # Create a block with conv, batch norm, and activation
            block = []
            block.append(conv_layer)

            if use_batch_norm:
                block.append(nn.BatchNorm1d(num_filters))

            block.append(nn.ReLU())

            # Add block to conv_layers
            self.conv_layers.append(nn.Sequential(*block))

        # Calculate size after convolutions
        final_conv_size = filter_sizes[-1] * window_size

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_conv_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, window_size * num_neurons)

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        # Reshape input to (batch_size, 1, window_size * num_neurons)
        # Assuming x is (batch_size, window_size * num_neurons)
        batch_size = x.size(0)

        # Reshape to (batch_size, num_neurons, window_size)
        x_reshaped = x.view(batch_size, self.num_neurons, self.window_size)

        # Apply convolutional layers
        residual = None
        for i, conv_layer in enumerate(self.conv_layers):
            if i == 0:
                # Save input for residual connection
                if self.use_residual:
                    residual = x_reshaped

                # Apply first conv layer
                x_reshaped = conv_layer(x_reshaped)
            else:
                # Apply subsequent conv layers
                x_reshaped = conv_layer(x_reshaped)

                # Add residual connection if specified
                if self.use_residual and i == len(self.conv_layers) - 1 and residual is not None:
                    # Ensure residual has the same shape as current output
                    if residual.size(1) != x_reshaped.size(1):
                        # Apply 1x1 convolution to match number of channels
                        residual = nn.Conv1d(
                            residual.size(1), x_reshaped.size(1), kernel_size=1
                        ).to(residual.device)(residual)

                    # Add residual connection
                    x_reshaped = x_reshaped + residual

        # Apply fully connected layers
        output = self.fc_layers(x_reshaped)

        return output


class CNNModel:
    """
    CNN model wrapper for neural signal classification.
    """

    def __init__(self,
                 input_size: Optional[int] = None,
                 window_size: int = 15,
                 num_neurons: Optional[int] = None,
                 filter_sizes: List[int] = [64, 128, 256],
                 kernel_size: int = 3,
                 output_size: int = 2,
                 use_residual: bool = True,
                 use_batch_norm: bool = True,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5,
                 batch_size: int = 32,
                 num_epochs: int = 100,
                 early_stopping_patience: int = 15,
                 use_focal_loss: bool = True,
                 focal_loss_gamma: float = 2.0,
                 focal_loss_alpha: Optional[List[float]] = None,
                 random_state: int = 42,
                 device: Optional[str] = None):
        """
        Initialize CNN model wrapper.

        Parameters
        ----------
        input_size : int, optional
            Size of input features (can be None if determined during training)
        window_size : int, optional
            Size of the sliding window
        num_neurons : int, optional
            Number of neurons in the input (can be None if determined during training)
        filter_sizes : List[int], optional
            Number of filters in each convolutional layer
        kernel_size : int, optional
            Size of convolutional kernel
        output_size : int, optional
            Number of output classes
        use_residual : bool, optional
            Whether to use residual connections
        use_batch_norm : bool, optional
            Whether to use batch normalization
        learning_rate : float, optional
            Learning rate for optimizer
        weight_decay : float, optional
            Weight decay for optimizer
        batch_size : int, optional
            Batch size for training
        num_epochs : int, optional
            Maximum number of epochs for training
        early_stopping_patience : int, optional
            Patience for early stopping
        use_focal_loss : bool, optional
            Whether to use focal loss for imbalanced data
        focal_loss_gamma : float, optional
            Focusing parameter for focal loss
        focal_loss_alpha : List[float], optional
            Class weights for focal loss
        random_state : int, optional
            Random seed for reproducibility
        device : str, optional
            Device to use for training ('cuda' or 'cpu')
        """
        self.model_args = {
            'input_size': input_size,
            'window_size': window_size,
            'num_neurons': num_neurons,
            'filter_sizes': filter_sizes,
            'kernel_size': kernel_size,
            'output_size': output_size,
            'use_residual': use_residual,
            'use_batch_norm': use_batch_norm
        }

        self.training_args = {
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'early_stopping_patience': early_stopping_patience,
            'use_focal_loss': use_focal_loss,
            'focal_loss_gamma': focal_loss_gamma,
            'focal_loss_alpha': focal_loss_alpha,
            'random_state': random_state
        }

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Set random seed
        torch.manual_seed(random_state)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(random_state)
            torch.cuda.manual_seed_all(random_state)

        self.model = None
        self.feature_importances_ = None

    def _focal_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss for classification.

        Parameters
        ----------
        outputs : torch.Tensor
            Model outputs (logits)
        targets : torch.Tensor
            Target labels

        Returns
        -------
        torch.Tensor
            Focal loss
        """
        # Convert targets to one-hot encoding
        num_classes = outputs.size(1)
        one_hot_targets = torch.zeros_like(outputs)
        one_hot_targets.scatter_(1, targets.unsqueeze(1), 1)

        # Apply softmax to get probabilities
        probas = torch.softmax(outputs, dim=1)

        # Compute focal loss
        gamma = self.training_args['focal_loss_gamma']
        alpha = self.training_args['focal_loss_alpha']

        # Element-wise focal loss
        focal_loss = -one_hot_targets * ((1 - probas) ** gamma) * torch.log(probas + 1e-8)

        # Apply class weights if specified
        if alpha is not None:
            alpha_tensor = torch.tensor(alpha, device=outputs.device)
            focal_loss = focal_loss * alpha_tensor.view(1, -1)

        # Sum over classes, mean over batch
        return focal_loss.sum(dim=1).mean()

    def _create_data_loaders(self, X_train: np.ndarray, y_train: np.ndarray,
                             X_val: Optional[np.ndarray] = None,
                             y_val: Optional[np.ndarray] = None) -> Tuple[
        torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader]]:
        """
        Create data loaders for training and validation.

        Parameters
        ----------
        X_train : np.ndarray
            Training data
        y_train : np.ndarray
            Training labels
        X_val : np.ndarray, optional
            Validation data
        y_val : np.ndarray, optional
            Validation labels

        Returns
        -------
        Tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader]]
            Training and validation data loaders
        """
        # Convert numpy arrays to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)

        # Create datasets
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)

        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.training_args['batch_size'], shuffle=True
        )

        # Create validation data loader if validation data is provided
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.LongTensor(y_val)
            val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=self.training_args['batch_size'], shuffle=False
            )

        return train_loader, val_loader

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train the CNN model.

        Parameters
        ----------
        X_train : np.ndarray
            Training data
        y_train : np.ndarray
            Training labels
        X_val : np.ndarray, optional
            Validation data
        y_val : np.ndarray, optional
            Validation labels

        Returns
        -------
        Dict[str, Any]
            Dictionary containing training metrics
        """
        # If input_size or num_neurons is not specified, determine it from X_train
        if self.model_args['input_size'] is None:
            self.model_args['input_size'] = X_train.shape[1]

        if self.model_args['num_neurons'] is None:
            self.model_args['num_neurons'] = X_train.shape[1] // self.model_args['window_size']

        # Create model
        self.model = CNN(**self.model_args).to(self.device)

        # Create data loaders
        train_loader, val_loader = self._create_data_loaders(X_train, y_train, X_val, y_val)

        # Create optimizer
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.training_args['learning_rate'],
            weight_decay=self.training_args['weight_decay']
        )

        # Create learning rate scheduler
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.training_args['learning_rate'],
            epochs=self.training_args['num_epochs'],
            steps_per_epoch=len(train_loader)
        )

        # Set loss function
        if self.training_args['use_focal_loss']:
            # If alpha is not specified, compute it from class frequencies
            if self.training_args['focal_loss_alpha'] is None:
                # Count class frequencies
                class_counts = np.bincount(y_train)
                # Compute inverse class frequencies
                class_weights = 1.0 / class_counts
                # Normalize weights
                class_weights = class_weights / np.sum(class_weights) * len(class_counts)
                self.training_args['focal_loss_alpha'] = class_weights.tolist()

            criterion = self._focal_loss
        else:
            criterion = nn.CrossEntropyLoss()

        # Training loop
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []

        print(f"Training for up to {self.training_args['num_epochs']} epochs...")
        for epoch in range(self.training_args['num_epochs']):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)

                # Compute loss
                if self.training_args['use_focal_loss']:
                    loss = criterion(outputs, targets)
                else:
                    loss = criterion(outputs, targets)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                scheduler.step()

                # Accumulate training loss
                train_loss += loss.item() * inputs.size(0)

                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == targets).sum().item()
                train_total += targets.size(0)

            # Calculate average training loss and accuracy
            train_loss = train_loss / len(train_loader.dataset)
            train_acc = train_correct / train_total
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            # Validation phase if validation data is provided
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)

                        # Forward pass
                        outputs = self.model(inputs)

                        # Compute loss
                        if self.training_args['use_focal_loss']:
                            loss = criterion(outputs, targets)
                        else:
                            loss = criterion(outputs, targets)

                        # Accumulate validation loss
                        val_loss += loss.item() * inputs.size(0)

                        # Calculate accuracy
                        _, predicted = torch.max(outputs, 1)
                        val_correct += (predicted == targets).sum().item()
                        val_total += targets.size(0)

                # Calculate average validation loss and accuracy
                val_loss = val_loss / len(val_loader.dataset)
                val_acc = val_correct / val_total
                val_losses.append(val_loss)
                val_accs.append(val_acc)

                # Check for early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    patience_counter = 0

                    # Save best model
                    best_model_state = self.model.state_dict()
                else:
                    patience_counter += 1

                # Print progress
                if (epoch + 1) % 10 == 0 or epoch == 0 or patience_counter >= self.training_args[
                    'early_stopping_patience']:
                    print(f"Epoch {epoch + 1}/{self.training_args['num_epochs']}, "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

                # Early stopping
                if patience_counter >= self.training_args['early_stopping_patience']:
                    print(f"Early stopping at epoch {epoch + 1}")
                    # Load best model
                    self.model.load_state_dict(best_model_state)
                    break
            else:
                # Print progress without validation data
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"Epoch {epoch + 1}/{self.training_args['num_epochs']}, "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # After training, calculate feature importances
        self._calculate_feature_importances()

        # Return metrics
        metrics = {
            'train_loss': train_losses,
            'train_accuracy': train_accs,
            'final_train_loss': train_losses[-1],
            'final_train_accuracy': train_accs[-1],
        }

        if val_loader is not None:
            metrics.update({
                'val_loss': val_losses,
                'val_accuracy': val_accs,
                'final_val_loss': val_losses[-1],
                'final_val_accuracy': val_accs[-1],
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch
            })

        return metrics

    def _calculate_feature_importances(self) -> None:
        """
        Calculate feature importances using filter activations.
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        # Create a dummy input
        dummy_input = torch.zeros((1, self.model_args['num_neurons'] * self.model_args['window_size'])).to(self.device)

        # Get gradients for each input feature
        dummy_input.requires_grad_(True)

        # Forward pass
        output = self.model(dummy_input)
        output_idx = output.argmax(dim=1)

        # Create one-hot encoding of output
        one_hot = torch.zeros_like(output)
        one_hot[0, output_idx] = 1

        # Backward pass
        self.model.zero_grad()
        output.backward(gradient=one_hot)

        # Extract gradients as feature importances
        feature_importances = dummy_input.grad.data.abs().squeeze().cpu().numpy()

        # Normalize feature importances
        if np.sum(feature_importances) > 0:
            feature_importances = feature_importances / np.sum(feature_importances)

        self.feature_importances_ = feature_importances

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the trained model.

        Parameters
        ----------
        X : np.ndarray
            Input data

        Returns
        -------
        np.ndarray
            Predicted labels
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        # Convert numpy array to PyTorch tensor
        X_tensor = torch.FloatTensor(X).to(self.device)

        # Set model to evaluation mode
        self.model.eval()

        # Make predictions
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)

        return predicted.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.

        Parameters
        ----------
        X : np.ndarray
            Input data

        Returns
        -------
        np.ndarray
            Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        # Convert numpy array to PyTorch tensor
        X_tensor = torch.FloatTensor(X).to(self.device)

        # Set model to evaluation mode
        self.model.eval()

        # Make predictions
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probas = torch.softmax(outputs, dim=1)

        return probas.cpu().numpy()

    def save(self, file_path: str) -> None:
        """
        Save the trained model.

        Parameters
        ----------
        file_path : str
            Path to save the model
        """
        if self.model is None:
            raise ValueError("No trained model to save.")

        # Save model state, model arguments, and feature importances
        save_dict = {
            'model_state': self.model.state_dict(),
            'model_args': self.model_args,
            'feature_importances': self.feature_importances_
        }

        torch.save(save_dict, file_path)
        print(f"Model saved to {file_path}")

    def load(self, file_path: str) -> None:
        """
        Load a trained model.

        Parameters
        ----------
        file_path : str
            Path to the saved model
        """
        # Load saved data
        save_dict = torch.load(file_path, map_location=self.device)

        # Extract model arguments and feature importances
        self.model_args = save_dict['model_args']
        self.feature_importances_ = save_dict['feature_importances']

        # Create model
        self.model = CNN(**self.model_args).to(self.device)

        # Load model state
        self.model.load_state_dict(save_dict['model_state'])

        print(f"Model loaded from {file_path}")

    def get_feature_importances(self) -> np.ndarray:
        """
        Get feature importances from the trained model.

        Returns
        -------
        np.ndarray
            Feature importances
        """
        if self.feature_importances_ is None:
            raise ValueError("Feature importances not available. Train the model first.")

        return self.feature_importances_

    