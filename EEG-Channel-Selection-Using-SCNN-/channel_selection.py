"""
Channel selection logic using Shallow CNNs
"""
import torch
import numpy as np
from typing import List, Tuple, Dict, Union
import logging
from tqdm import tqdm

from config import config
from scnn_model import build_shallow_cnn_selector, SCNNTrainer
from preprocessing import create_data_loaders

logger = logging.getLogger(__name__)


def select_best_channels(
    channel_data_list: List[Tuple[torch.Tensor, torch.Tensor]],
    model_builder: callable = None,
    K: int = None,
    ch_names: List[str] = None,
    max_epochs: int = None,
    patience: int = None,
    device: str = None
) -> Tuple[List[int], List[str], Dict]:
    """
    Select the best K channels using individual SCNN training.
    
    Args:
        channel_data_list: List of (X_channel, y_labels) tuples for each channel
        model_builder: Function to build SCNN model (default: build_shallow_cnn_selector)
        K: Number of channels to select
        ch_names: List of channel names
        max_epochs: Maximum epochs for training each SCNN
        patience: Early stopping patience
        device: Computing device
        
    Returns:
        Tuple of (selected_indices, selected_names, results_dict)
    """
    if model_builder is None:
        model_builder = build_shallow_cnn_selector
    if K is None:
        K = config.K_selected
    if ch_names is None:
        ch_names = [f"Ch_{i}" for i in range(len(channel_data_list))]
    if max_epochs is None:
        max_epochs = config.max_epochs
    if patience is None:
        patience = config.patience
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    n_channels = len(channel_data_list)
    logger.info(f"Starting channel selection from {n_channels} channels")
    logger.info(f"Target selection: Top {K} channels")
    logger.info(f"Using device: {device}")
    
    channel_results = []
    
    # Train SCNN for each channel
    for ch_idx in tqdm(range(n_channels), desc="Training SCNNs per channel"):
        logger.info(f"\n--- Training SCNN for Channel {ch_idx} ({ch_names[ch_idx]}) ---")
        
        # Get channel data
        X_channel, y_labels = channel_data_list[ch_idx]
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            X_channel, y_labels,
            batch_size=config.batch_size,
            validation_split=config.validation_split
        )
        
        # Build SCNN model
        model = model_builder(
            input_time_samples=X_channel.shape[1],
            n_classes=len(torch.unique(y_labels))
        )
        
        # Create trainer
        trainer = SCNNTrainer(model, device=device)
        
        # Train model
        try:
            best_accuracy = trainer.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                max_epochs=max_epochs,
                patience=patience
            )
            
            # Record results
            channel_results.append({
                'channel_index': ch_idx,
                'channel_name': ch_names[ch_idx],
                'best_accuracy': best_accuracy,
                'train_history': {
                    'train_losses': trainer.train_losses,
                    'val_losses': trainer.val_losses,
                    'val_accuracies': trainer.val_accuracies
                }
            })
            
            logger.info(f"Channel {ch_idx} ({ch_names[ch_idx]}): "
                       f"Best accuracy = {best_accuracy:.4f}")
            
        except Exception as e:
            logger.error(f"Error training SCNN for channel {ch_idx}: {str(e)}")
            # Record failed channel with 0 accuracy
            channel_results.append({
                'channel_index': ch_idx,
                'channel_name': ch_names[ch_idx],
                'best_accuracy': 0.0,
                'error': str(e)
            })
    
    # Sort channels by accuracy (descending)
    sorted_results = sorted(channel_results, key=lambda x: x['best_accuracy'], reverse=True)
    
    # Select top K channels
    selected_indices = [result['channel_index'] for result in sorted_results[:K]]
    selected_names = [result['channel_name'] for result in sorted_results[:K]]
    selected_accuracies = [result['best_accuracy'] for result in sorted_results[:K]]
    
    # Log selection results
    logger.info(f"\n=== CHANNEL SELECTION RESULTS ===")
    logger.info(f"Selected top {K} channels:")
    for i, (idx, name, acc) in enumerate(zip(selected_indices, selected_names, selected_accuracies)):
        logger.info(f"  {i+1}. Channel {idx} ({name}): {acc:.4f}")
    
    # Create comprehensive results dictionary
    results_dict = {
        'all_results': channel_results,
        'sorted_results': sorted_results,
        'selected_indices': selected_indices,
        'selected_names': selected_names,
        'selected_accuracies': selected_accuracies,
        'selection_summary': {
            'total_channels': n_channels,
            'selected_channels': K,
            'best_accuracy': max(selected_accuracies) if selected_accuracies else 0.0,
            'worst_selected_accuracy': min(selected_accuracies) if selected_accuracies else 0.0,
            'mean_selected_accuracy': np.mean(selected_accuracies) if selected_accuracies else 0.0
        }
    }
    
    return selected_indices, selected_names, results_dict


def evaluate_channel_importance(
    channel_data_list: List[Tuple[torch.Tensor, torch.Tensor]],
    selected_indices: List[int],
    ch_names: List[str] = None
) -> Dict:
    """
    Evaluate the importance and characteristics of selected channels.
    
    Args:
        channel_data_list: Original channel data list
        selected_indices: Indices of selected channels
        ch_names: Channel names
        
    Returns:
        Dictionary with channel importance analysis
    """
    if ch_names is None:
        ch_names = [f"Ch_{i}" for i in range(len(channel_data_list))]
    
    analysis = {
        'selected_channels': {},
        'comparison': {}
    }
    
    # Analyze selected channels
    for idx in selected_indices:
        X_channel, y_labels = channel_data_list[idx]
        
        # Basic statistics
        channel_stats = {
            'name': ch_names[idx],
            'index': idx,
            'data_shape': X_channel.shape,
            'mean_amplitude': float(torch.mean(X_channel)),
            'std_amplitude': float(torch.std(X_channel)),
            'min_amplitude': float(torch.min(X_channel)),
            'max_amplitude': float(torch.max(X_channel))
        }
        
        analysis['selected_channels'][idx] = channel_stats
    
    # Overall comparison
    all_means = []
    all_stds = []
    selected_means = []
    selected_stds = []
    
    for idx, (X_channel, _) in enumerate(channel_data_list):
        mean_amp = float(torch.mean(X_channel))
        std_amp = float(torch.std(X_channel))
        
        all_means.append(mean_amp)
        all_stds.append(std_amp)
        
        if idx in selected_indices:
            selected_means.append(mean_amp)
            selected_stds.append(std_amp)
    
    analysis['comparison'] = {
        'all_channels_mean_amplitude': np.mean(all_means),
        'all_channels_std_amplitude': np.mean(all_stds),
        'selected_channels_mean_amplitude': np.mean(selected_means),
        'selected_channels_std_amplitude': np.mean(selected_stds),
        'selection_ratio': len(selected_indices) / len(channel_data_list)
    }
    
    logger.info("Channel importance analysis completed")
    return analysis


def save_channel_selection_results(
    results_dict: Dict,
    filepath: str = "channel_selection_results.npz"
):
    """
    Save channel selection results to file.
    
    Args:
        results_dict: Results from select_best_channels
        filepath: Path to save results
    """
    # Prepare data for saving (convert to numpy arrays where possible)
    save_data = {
        'selected_indices': np.array(results_dict['selected_indices']),
        'selected_names': np.array(results_dict['selected_names']),
        'selected_accuracies': np.array(results_dict['selected_accuracies']),
        'selection_summary': results_dict['selection_summary']
    }
    
    # Save accuracy results for all channels
    all_accuracies = [r['best_accuracy'] for r in results_dict['all_results']]
    all_channel_names = [r['channel_name'] for r in results_dict['all_results']]
    
    save_data['all_accuracies'] = np.array(all_accuracies)
    save_data['all_channel_names'] = np.array(all_channel_names)
    
    np.savez(filepath, **save_data)
    logger.info(f"Channel selection results saved to {filepath}")


def load_channel_selection_results(filepath: str = "channel_selection_results.npz") -> Dict:
    """
    Load channel selection results from file.
    
    Args:
        filepath: Path to load results from
        
    Returns:
        Loaded results dictionary
    """
    loaded = np.load(filepath, allow_pickle=True)
    
    results_dict = {
        'selected_indices': loaded['selected_indices'].tolist(),
        'selected_names': loaded['selected_names'].tolist(),
        'selected_accuracies': loaded['selected_accuracies'].tolist(),
        'selection_summary': loaded['selection_summary'].item(),
        'all_accuracies': loaded['all_accuracies'].tolist(),
        'all_channel_names': loaded['all_channel_names'].tolist()
    }
    
    logger.info(f"Channel selection results loaded from {filepath}")
    return results_dict