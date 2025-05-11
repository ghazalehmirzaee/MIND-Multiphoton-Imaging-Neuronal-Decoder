# mind/models/deep/fcnn.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional, List, Tuple


class FCNN(nn.Module):
    """
    Fully Connected Neural Network model for neural signal classification.
    """

    def __init__(self,
                 input_size: int,
                 hidden_sizes: List[int] = [256, 128, 64],
                 output_size: int = 2,
                 dropout_rates: List[float] = [0.3, 0.3, 0.3],
                 batch_norm: bool = True):
        """
        Initialize FCNN model.

        Parameters
        ----------
        input_size : int
            Size of input features
        hidden_sizes : List[int], optional
            Sizes of hidden layers
        output_size : int, optional
            Number of output classes
        dropout_rates : List[float], optional
            Dropout rates for each hidden layer
        batch_norm : bool, optional
            Whether to use batch normalization
        """
        super(FCNN, self).__init__()

        layers = []
        prev_size = input_size

        # Create hidden layers
        for i, (size, dropout_rate) in enumerate(zip(hidden_sizes, dropout_rates)):
            # Linear layer
            layers.append(nn.Linear(prev_size, size))

            # Batch normalization
            if batch_norm:
                layers.append(nn.BatchNorm1d(size))

            # Activation
            layers.append(nn.ReLU())

            # Dropout
            layers.append(nn.Dropout(dropout_rate))

            prev_size = size

        # Output layer
        layers.append(nn.Linear(prev_size, output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        return self.model(x)


class FCNNModel:
    """
    FCNN model wrapper for neural signal classification.
    """

    def __init__(self,
                 input_size: Optional[int] = None,
                 hidden_sizes: List[int] = [256, 128, 64],
                 output_size: int = 2,
                 dropout_rates: List[float] = [0.3, 0.3, 0.3],
                 batch_norm: bool = True,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5,
                 batch_size: int = 32,
                 num_epochs: int = 100,
                 early_stopping_patience: int = 15,
                 random_state: int = 42,
                 device: Optional[str] = None):
        """
        Initialize FCNN model wrapper.

        Parameters
        ----------
        input_size : int, optional
            Size of input features (can be None if determined during training)
        hidden_sizes : List[int], optional
            Sizes of hidden layers
        output_size : int, optional
            Number of output classes
        dropout_rates : List[float], optional
            Dropout rates for each hidden layer
        batch_norm : bool, optional
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
        random_state : int, optional
            Random seed for reproducibility
        device : str, optional
            Device to use for training ('cuda' or 'cpu')
        """
        self.model_args = {
            'input_size': input_size,
            'hidden_sizes': hidden_sizes,
            'output_size': output_size,
            'dropout_rates': dropout_rates,
            'batch_norm': batch_norm
        }

        self.training_args = {
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'early_stopping_patience': early_stopping_patience,
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
        Train the FCNN model.

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
        # If input_size is not specified, determine it from X_train
        if self.model_args['input_size'] is None:
            self.model_args['input_size'] = X_train.shape[1]

        # Create model
        self.model = FCNN(**self.model_args).to(self.device)

        # Create data loaders
        train_loader, val_loader = self._create_data_loaders(X_train, y_train, X_val, y_val)

        # Create optimizer and loss function
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.training_args['learning_rate'],
            weight_decay=self.training_args['weight_decay']
        )
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
                loss = criterion(outputs, targets)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

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
        Calculate feature importances based on the weights of the first layer.
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        # Get weights from the first layer
        first_layer_weights = self.model.model[0].weight.detach().cpu().numpy()

        # Calculate feature importances as the absolute sum of weights connecting each input feature
        self.feature_importances_ = np.abs(first_layer_weights).sum(axis=0)

        # Normalize feature importances
        self.feature_importances_ = self.feature_importances_ / np.sum(self.feature_importances_)

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
        self.model = FCNN(**self.model_args).to(self.device)

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

    