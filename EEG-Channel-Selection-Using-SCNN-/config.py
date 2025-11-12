"""
Configuration class for EEG Channel Selection and Classification
"""
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class EEGConfig:
    """Configuration parameters for EEG processing and model training"""
    
    # Data preprocessing parameters
    sampling_rate: int = 250
    target_freq: Tuple[int, int] = (7, 40)
    segment_duration: float = 3.0  # seconds
    time_samples: int = 1001  # For 3-second segments at 250 Hz
    
    # Channel selection parameters
    n_channels_total: int = 22  # Default for BCI Competition datasets
    K_selected: int = 6
    n_classes: int = 4
    
    # SCNN model parameters
    scnn_filters_temporal: int = 50
    scnn_kernel_temporal: Tuple[int, int] = (30, 1)
    scnn_filters_pointwise1: int = 30
    scnn_filters_pointwise2: int = 20
    scnn_dropout_rate: float = 0.5
    scnn_dense_units: int = 1024
    scnn_l2_reg: float = 0.01
    scnn_learning_rate: float = 1e-4
    
    # Fusion CNN parameters
    fusion_bands: List[Tuple[int, int]] = None
    fusion_mlp_units: List[int] = None
    fusion_learning_rate: float = 1e-4
    
    # Training parameters
    batch_size: int = 32
    max_epochs: int = 100
    patience: int = 15
    validation_split: float = 0.2
    cv_folds: int = 5
    
    def __post_init__(self):
        if self.fusion_bands is None:
            self.fusion_bands = [
                (7, 13),   # Input A: Alpha band
                (13, 31),  # Input B: Beta band  
                (7, 31),   # Input C: Alpha + Beta
                (0, 40)    # Input D: Full band
            ]
        
        if self.fusion_mlp_units is None:
            self.fusion_mlp_units = [50, 50]


# Global config instance
config = EEGConfig()