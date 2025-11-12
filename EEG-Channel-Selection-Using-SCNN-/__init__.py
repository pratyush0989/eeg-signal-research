"""
Two-Stage EEG Channel Selection and Classification

A highly modular PyTorch implementation for EEG motor imagery classification using:
1. Shallow Convolutional Neural Networks (SCNNs) for channel selection
2. Multi-Layer Fusion CNN for final classification

This package provides a complete pipeline for processing EEG data, selecting optimal
channels, and performing classification across multiple frequency bands.
"""

__version__ = "1.0.0"
__author__ = "EEG Analysis Team"

# Import main components for easy access
from .config import config, EEGConfig
from .pipeline import run_full_pipeline, analyze_pipeline_results
from .preprocessing import (
    preprocess_data_for_selection,
    generate_single_channel_inputs, 
    prepare_fusion_input
)
from .channel_selection import select_best_channels
from .scnn_model import build_shallow_cnn_selector, ShallowCNN, SCNNTrainer
from .fusion_cnn import build_fusion_cnn_classifier, FusionCNN, FusionCNNTrainer
from .utils import (
    plot_channel_selection_results,
    plot_training_history,
    evaluate_model_predictions,
    generate_comprehensive_report
)

__all__ = [
    # Configuration
    'config', 'EEGConfig',
    
    # Main pipeline
    'run_full_pipeline', 'analyze_pipeline_results',
    
    # Preprocessing
    'preprocess_data_for_selection',
    'generate_single_channel_inputs',
    'prepare_fusion_input',
    
    # Channel selection
    'select_best_channels',
    
    # Models
    'build_shallow_cnn_selector', 'ShallowCNN', 'SCNNTrainer',
    'build_fusion_cnn_classifier', 'FusionCNN', 'FusionCNNTrainer',
    
    # Utilities
    'plot_channel_selection_results',
    'plot_training_history', 
    'evaluate_model_predictions',
    'generate_comprehensive_report'
]

def get_version():
    """Return package version"""
    return __version__

def get_info():
    """Return package information"""
    return {
        'name': 'EEG Channel Selection',
        'version': __version__,
        'description': __doc__.strip(),
        'author': __author__
    }