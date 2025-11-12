"""
Shallow Convolutional Neural Network (SCNN) architecture for channel selection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import logging

from config import config

logger = logging.getLogger(__name__)


class ShallowCNN(nn.Module):
    """
    Shallow Convolutional Neural Network for single-channel EEG classification.
    
    Architecture based on the paper specifications:
    - Temporal convolution (50 filters, kernel size 30x1)
    - Two pointwise convolutions (30 and 20 filters)
    - Batch normalization and ELU activations
    - Dropout and fully connected layers
    """
    
    def __init__(
        self, 
        input_time_samples: int = None,
        n_classes: int = None,
        filters_temporal: int = None,
        kernel_temporal: Tuple[int, int] = None,
        filters_pointwise1: int = None,
        filters_pointwise2: int = None,
        dropout_rate: float = None,
        dense_units: int = None,
        l2_reg: float = None
    ):
        super(ShallowCNN, self).__init__()
        
        # Use config defaults if not specified
        self.input_time_samples = input_time_samples or config.time_samples
        self.n_classes = n_classes or config.n_classes
        self.filters_temporal = filters_temporal or config.scnn_filters_temporal
        self.kernel_temporal = kernel_temporal or config.scnn_kernel_temporal
        self.filters_pointwise1 = filters_pointwise1 or config.scnn_filters_pointwise1
        self.filters_pointwise2 = filters_pointwise2 or config.scnn_filters_pointwise2
        self.dropout_rate = dropout_rate or config.scnn_dropout_rate
        self.dense_units = dense_units or config.scnn_dense_units
        self.l2_reg = l2_reg or config.scnn_l2_reg
        
        # Layer 1: Temporal Convolution
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=self.filters_temporal,
            kernel_size=self.kernel_temporal,
            padding='same'
        )
        self.bn1 = nn.BatchNorm2d(self.filters_temporal)
        
        # Layer 2: First Pointwise Convolution
        self.conv2 = nn.Conv2d(
            in_channels=self.filters_temporal,
            out_channels=self.filters_pointwise1,
            kernel_size=(1, 1)
        )
        self.bn2 = nn.BatchNorm2d(self.filters_pointwise1)
        
        # Layer 3: Second Pointwise Convolution
        self.conv3 = nn.Conv2d(
            in_channels=self.filters_pointwise1,
            out_channels=self.filters_pointwise2,
            kernel_size=(1, 1)
        )
        self.bn3 = nn.BatchNorm2d(self.filters_pointwise2)
        
        # Dropout
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Calculate flattened size after convolutions
        self.flattened_size = self.filters_pointwise2 * self.input_time_samples * 1
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, self.dense_units)
        self.fc2 = nn.Linear(self.dense_units, self.n_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, time_samples, 1)
            
        Returns:
            Output tensor of shape (batch_size, n_classes)
        """
        # Reshape input to (batch_size, 1, time_samples, 1) for Conv2D
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension
        
        # Temporal convolution block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.elu(x)
        
        # First pointwise convolution block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x)
        
        # Second pointwise convolution block  
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.elu(x)
        
        # Dropout
        x = self.dropout(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.elu(x)
        
        # Apply L2 regularization to FC1 weights during training
        if self.training:
            l2_reg = torch.norm(self.fc1.weight, p=2) * self.l2_reg
            # Note: L2 regularization will be added to loss externally
        
        # Output layer
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        
        return x
    
    def get_l2_regularization(self):
        """Get L2 regularization term for the first FC layer"""
        return torch.norm(self.fc1.weight, p=2) * self.l2_reg


def build_shallow_cnn_selector(
    input_time_samples: int = None,
    n_classes: int = None,
    **kwargs
) -> ShallowCNN:
    """
    Build and configure a Shallow CNN model for channel selection.
    
    Args:
        input_time_samples: Number of time samples in input
        n_classes: Number of output classes
        **kwargs: Additional model parameters
        
    Returns:
        Configured ShallowCNN model
    """
    model = ShallowCNN(
        input_time_samples=input_time_samples,
        n_classes=n_classes,
        **kwargs
    )
    
    logger.info(f"Built Shallow CNN with {sum(p.numel() for p in model.parameters())} parameters")
    logger.info(f"Input shape: ({input_time_samples or config.time_samples}, 1)")
    logger.info(f"Output classes: {n_classes or config.n_classes}")
    
    return model


class SCNNTrainer:
    """Trainer class for Shallow CNN models"""
    
    def __init__(
        self,
        model: ShallowCNN,
        learning_rate: float = None,
        device: str = None
    ):
        self.model = model
        self.learning_rate = learning_rate or config.scnn_learning_rate
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup optimizer and loss function
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
        logger.info(f"SCNN Trainer initialized on device: {self.device}")
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            
            # Calculate loss including L2 regularization
            loss = self.criterion(output, target)
            l2_reg = self.model.get_l2_regularization()
            total_loss_batch = loss + l2_reg
            
            total_loss_batch.backward()
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                loss = self.criterion(output, target)
                l2_reg = self.model.get_l2_regularization()
                total_loss += (loss + l2_reg).item()
                
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def fit(
        self, 
        train_loader, 
        val_loader, 
        max_epochs: int = None,
        patience: int = None
    ):
        """
        Train the model with early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader  
            max_epochs: Maximum number of epochs
            patience: Early stopping patience
            
        Returns:
            Best validation accuracy achieved
        """
        max_epochs = max_epochs or config.max_epochs
        patience = patience or config.patience
        
        best_val_accuracy = 0.0
        patience_counter = 0
        
        logger.info(f"Training SCNN for up to {max_epochs} epochs")
        
        for epoch in range(max_epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_accuracy = self.validate(val_loader)
            
            # Record history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            
            # Early stopping check
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
                # Save best model state
                self.best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if epoch % 10 == 0 or patience_counter == 0:
                logger.info(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch} (patience: {patience})")
                break
        
        # Load best model
        if hasattr(self, 'best_state'):
            self.model.load_state_dict(self.best_state)
        
        logger.info(f"Training completed. Best validation accuracy: {best_val_accuracy:.4f}")
        return best_val_accuracy