"""
Multi-Layer Fusion CNN architecture for EEG classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import logging

from config import config

logger = logging.getLogger(__name__)


class ParallelCNNBranch(nn.Module):
    """
    Individual CNN branch for processing one frequency band.
    Each branch processes K selected channels for a specific frequency range.
    """
    
    def __init__(
        self,
        K_channels: int,
        input_time_samples: int,
        branch_name: str = "Branch"
    ):
        super(ParallelCNNBranch, self).__init__()
        
        self.K_channels = K_channels
        self.input_time_samples = input_time_samples
        self.branch_name = branch_name
        
        # Spatial convolution layer (adapted for K channels instead of full channel count)
        self.spatial_conv = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=(1, K_channels),  # Spatial filter across K channels
            padding=0
        )
        
        # Temporal convolution layers
        self.temporal_conv1 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(25, 1),  # Temporal filter
            padding=(12, 0)  # Same padding for temporal dimension
        )
        
        self.temporal_conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(15, 1),
            padding=(7, 0)
        )
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Dropout layers
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature dimension (output of this branch)
        self.feature_dim = 128
        
        logger.debug(f"Initialized {branch_name} for {K_channels} channels, "
                    f"{input_time_samples} time samples")
    
    def forward(self, x):
        """
        Forward pass through the CNN branch.
        
        Args:
            x: Input tensor (batch_size, K_channels, time_samples)
            
        Returns:
            Feature vector (batch_size, feature_dim)
        """
        # Reshape for Conv2D: (batch_size, 1, time_samples, K_channels)
        x = x.permute(0, 2, 1).unsqueeze(1)  # (batch, time, channels) -> (batch, 1, time, channels)
        
        # Spatial convolution
        x = self.spatial_conv(x)  # -> (batch, 32, time_samples, 1)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout1(x)
        
        # First temporal convolution
        x = self.temporal_conv1(x)  # -> (batch, 64, time_samples, 1)
        x = self.bn2(x)
        x = F.elu(x)
        
        # Second temporal convolution  
        x = self.temporal_conv2(x)  # -> (batch, 128, time_samples, 1)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.dropout2(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)  # -> (batch, 128, 1, 1)
        
        # Flatten to feature vector
        x = x.view(x.size(0), -1)  # -> (batch, 128)
        
        return x


class FusionCNN(nn.Module):
    """
    Multi-Layer Fusion CNN that combines features from four parallel branches
    processing different frequency bands of the selected K channels.
    """
    
    def __init__(
        self,
        K_channels: int,
        input_time_samples: int,
        n_classes: int = None,
        mlp_units: List[int] = None
    ):
        super(FusionCNN, self).__init__()
        
        self.K_channels = K_channels
        self.input_time_samples = input_time_samples
        self.n_classes = n_classes or config.n_classes
        self.mlp_units = mlp_units or config.fusion_mlp_units.copy()
        
        # Four parallel CNN branches for different frequency bands
        self.branch_A = ParallelCNNBranch(K_channels, input_time_samples, "Branch_A_7-13Hz")
        self.branch_B = ParallelCNNBranch(K_channels, input_time_samples, "Branch_B_13-31Hz")
        self.branch_C = ParallelCNNBranch(K_channels, input_time_samples, "Branch_C_7-31Hz")
        self.branch_D = ParallelCNNBranch(K_channels, input_time_samples, "Branch_D_0-40Hz")
        
        # Calculate total feature dimension after concatenation
        total_feature_dim = (self.branch_A.feature_dim + 
                           self.branch_B.feature_dim + 
                           self.branch_C.feature_dim + 
                           self.branch_D.feature_dim)
        
        # Fusion MLP layers
        self.fusion_layers = nn.ModuleList()
        
        # Input layer to first hidden layer
        self.fusion_layers.append(nn.Linear(total_feature_dim, self.mlp_units[0]))
        
        # Hidden layers
        for i in range(len(self.mlp_units) - 1):
            self.fusion_layers.append(nn.Linear(self.mlp_units[i], self.mlp_units[i + 1]))
        
        # Output layer
        self.output_layer = nn.Linear(self.mlp_units[-1], self.n_classes)
        
        # Dropout for MLP
        self.fusion_dropout = nn.Dropout(0.5)
        
        logger.info(f"Fusion CNN initialized:")
        logger.info(f"  - Input: {K_channels} channels, {input_time_samples} time samples")
        logger.info(f"  - Parallel branches: 4 (for different frequency bands)")
        logger.info(f"  - Total feature dimension: {total_feature_dim}")
        logger.info(f"  - MLP structure: {self.mlp_units} -> {self.n_classes}")
        
    def forward(self, inputs):
        """
        Forward pass through the fusion network.
        
        Args:
            inputs: List of 4 tensors, each of shape (batch_size, K_channels, time_samples)
                   representing different frequency bands
                   
        Returns:
            Output tensor (batch_size, n_classes) with class probabilities
        """
        input_A, input_B, input_C, input_D = inputs
        
        # Extract features from each branch
        features_A = self.branch_A(input_A)
        features_B = self.branch_B(input_B)
        features_C = self.branch_C(input_C)
        features_D = self.branch_D(input_D)
        
        # Concatenate features from all branches
        fused_features = torch.cat([features_A, features_B, features_C, features_D], dim=1)
        
        # Pass through fusion MLP
        x = fused_features
        for i, layer in enumerate(self.fusion_layers):
            x = layer(x)
            x = F.relu(x)
            if i < len(self.fusion_layers) - 1:  # Apply dropout except on last hidden layer
                x = self.fusion_dropout(x)
        
        # Output layer with softmax
        x = self.output_layer(x)
        x = F.softmax(x, dim=1)
        
        return x
    
    def get_feature_representations(self, inputs):
        """
        Extract feature representations from each branch (for analysis).
        
        Args:
            inputs: List of 4 input tensors
            
        Returns:
            Dictionary with features from each branch and fused features
        """
        input_A, input_B, input_C, input_D = inputs
        
        with torch.no_grad():
            features_A = self.branch_A(input_A)
            features_B = self.branch_B(input_B)
            features_C = self.branch_C(input_C)
            features_D = self.branch_D(input_D)
            
            fused_features = torch.cat([features_A, features_B, features_C, features_D], dim=1)
        
        return {
            'branch_A_features': features_A,
            'branch_B_features': features_B,
            'branch_C_features': features_C,
            'branch_D_features': features_D,
            'fused_features': fused_features
        }


def build_fusion_cnn_classifier(
    K_channels: int,
    input_time_samples: int,
    n_classes: int = None,
    mlp_units: List[int] = None
) -> FusionCNN:
    """
    Build and configure a Multi-Layer Fusion CNN classifier.
    
    Args:
        K_channels: Number of selected channels
        input_time_samples: Number of time samples per trial
        n_classes: Number of output classes
        mlp_units: List of units in MLP hidden layers
        
    Returns:
        Configured FusionCNN model
    """
    model = FusionCNN(
        K_channels=K_channels,
        input_time_samples=input_time_samples,
        n_classes=n_classes,
        mlp_units=mlp_units
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Built Fusion CNN with {total_params:,} total parameters "
               f"({trainable_params:,} trainable)")
    
    return model


class FusionCNNTrainer:
    """Trainer class for Fusion CNN models"""
    
    def __init__(
        self,
        model: FusionCNN,
        learning_rate: float = None,
        device: str = None
    ):
        self.model = model
        self.learning_rate = learning_rate or config.fusion_learning_rate
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
        
        logger.info(f"Fusion CNN Trainer initialized on device: {self.device}")
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, batch_data in enumerate(train_loader):
            # Unpack batch data - expecting (inputs_list, targets)
            if len(batch_data) == 2:
                inputs_list, targets = batch_data
            else:
                # Handle different data loader formats
                inputs_list = batch_data[:-1]  # All but last are inputs
                targets = batch_data[-1]       # Last is targets
            
            # Move data to device
            inputs_list = [inp.to(self.device) for inp in inputs_list]
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs_list)
            loss = self.criterion(outputs, targets)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                # Unpack batch data
                if len(batch_data) == 2:
                    inputs_list, targets = batch_data
                else:
                    inputs_list = batch_data[:-1]
                    targets = batch_data[-1]
                
                # Move data to device
                inputs_list = [inp.to(self.device) for inp in inputs_list]
                targets = targets.to(self.device)
                
                outputs = self.model(inputs_list)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
                pred = outputs.argmax(dim=1)
                correct += pred.eq(targets).sum().item()
                total += targets.size(0)
        
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
        
        Returns:
            Best validation accuracy achieved
        """
        max_epochs = max_epochs or config.max_epochs
        patience = patience or config.patience
        
        best_val_accuracy = 0.0
        patience_counter = 0
        
        logger.info(f"Training Fusion CNN for up to {max_epochs} epochs")
        
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
        
        logger.info(f"Fusion CNN training completed. Best validation accuracy: {best_val_accuracy:.4f}")
        return best_val_accuracy