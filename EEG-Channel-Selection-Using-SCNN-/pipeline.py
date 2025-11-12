"""
Main pipeline execution for two-stage EEG channel selection and classification
"""
import torch
import numpy as np
from sklearn.model_selection import KFold
from typing import List, Tuple, Dict, Union
import logging
import time

from config import config
from preprocessing import (
    preprocess_data_for_selection, 
    generate_single_channel_inputs,
    prepare_fusion_input,
    create_data_loaders
)
from channel_selection import select_best_channels, save_channel_selection_results
from scnn_model import build_shallow_cnn_selector
from fusion_cnn import build_fusion_cnn_classifier, FusionCNNTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FusionDataLoader:
    """Custom data loader for fusion CNN that handles multiple inputs"""
    
    def __init__(self, fusion_inputs: List[np.ndarray], labels: np.ndarray, batch_size: int = 32, shuffle: bool = True):
        self.fusion_inputs = [torch.FloatTensor(inp) for inp in fusion_inputs]
        self.labels = torch.LongTensor(labels)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(labels)
        
        # Create indices
        self.indices = list(range(self.n_samples))
        if shuffle:
            np.random.shuffle(self.indices)
    
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        for start_idx in range(0, self.n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.n_samples)
            batch_indices = self.indices[start_idx:end_idx]
            
            # Get batch data for each input
            batch_inputs = []
            for fusion_input in self.fusion_inputs:
                batch_inputs.append(fusion_input[batch_indices])
            
            batch_labels = self.labels[batch_indices]
            
            yield batch_inputs + [batch_labels]  # Return as list for unpacking
    
    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size


def create_fusion_data_loaders(
    fusion_inputs: List[np.ndarray],
    labels: np.ndarray,
    validation_split: float = 0.2,
    batch_size: int = None,
    shuffle: bool = True
) -> Tuple[FusionDataLoader, FusionDataLoader]:
    """
    Create train and validation data loaders for fusion CNN.
    
    Args:
        fusion_inputs: List of 4 filtered input arrays
        labels: Target labels
        validation_split: Fraction for validation
        batch_size: Batch size
        shuffle: Whether to shuffle data
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    if batch_size is None:
        batch_size = config.batch_size
    
    n_samples = len(labels)
    n_val = int(validation_split * n_samples)
    n_train = n_samples - n_val
    
    # Create random indices for train/val split
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    # Split each fusion input
    train_fusion_inputs = [inp[train_indices] for inp in fusion_inputs]
    val_fusion_inputs = [inp[val_indices] for inp in fusion_inputs]
    
    train_labels = labels[train_indices]
    val_labels = labels[val_indices]
    
    # Create data loaders
    train_loader = FusionDataLoader(train_fusion_inputs, train_labels, batch_size, shuffle=True)
    val_loader = FusionDataLoader(val_fusion_inputs, val_labels, batch_size, shuffle=False)
    
    return train_loader, val_loader


def run_full_pipeline(
    raw_eeg_data: np.ndarray,
    labels: np.ndarray,
    N_channels: int = None,
    K_selected: int = None,
    ch_names: List[str] = None,
    sampling_rate: int = None,
    cv_folds: int = None,
    save_results: bool = True,
    results_prefix: str = "eeg_pipeline"
) -> Dict:
    """
    Execute the complete two-stage EEG processing pipeline.
    
    Args:
        raw_eeg_data: Raw EEG data (n_trials, n_channels, time_samples)
        labels: Class labels (n_trials,)
        N_channels: Total number of channels  
        K_selected: Number of channels to select
        ch_names: Channel names
        sampling_rate: Sampling rate in Hz
        cv_folds: Number of cross-validation folds
        save_results: Whether to save results to files
        results_prefix: Prefix for result files
        
    Returns:
        Dictionary with complete pipeline results
    """
    start_time = time.time()
    
    # Set defaults from config
    if N_channels is None:
        N_channels = raw_eeg_data.shape[1] if raw_eeg_data.ndim == 3 else raw_eeg_data.shape[0]
    if K_selected is None:
        K_selected = config.K_selected
    if ch_names is None:
        ch_names = [f"Ch_{i}" for i in range(N_channels)]
    if sampling_rate is None:
        sampling_rate = config.sampling_rate
    if cv_folds is None:
        cv_folds = config.cv_folds
    
    logger.info("="*60)
    logger.info("STARTING TWO-STAGE EEG CHANNEL SELECTION AND CLASSIFICATION PIPELINE")
    logger.info("="*60)
    logger.info(f"Input data shape: {raw_eeg_data.shape}")
    logger.info(f"Number of labels: {len(labels)}")
    logger.info(f"Total channels: {N_channels}")
    logger.info(f"Channels to select: {K_selected}")
    logger.info(f"Sampling rate: {sampling_rate} Hz")
    logger.info(f"Cross-validation folds: {cv_folds}")
    
    results = {
        'pipeline_config': {
            'N_channels': N_channels,
            'K_selected': K_selected,
            'sampling_rate': sampling_rate,
            'cv_folds': cv_folds,
            'input_shape': raw_eeg_data.shape,
            'n_classes': len(np.unique(labels))
        },
        'stage1_results': {},
        'stage2_results': {},
        'final_results': {}
    }
    
    # ========================================
    # STAGE 1: CHANNEL SELECTION
    # ========================================
    logger.info("\n" + "="*50)
    logger.info("STAGE 1: CHANNEL SELECTION USING SCNNs")
    logger.info("="*50)
    
    # Step 1.1: Preprocess data for selection
    logger.info("\nStep 1.1: Preprocessing data for channel selection...")
    processed_data, processed_ch_names = preprocess_data_for_selection(
        raw_eeg_data, sampling_rate, ch_names
    )
    
    # Step 1.2: Generate single-channel inputs
    logger.info("\nStep 1.2: Generating single-channel datasets...")
    channel_data_list = generate_single_channel_inputs(processed_data, labels)
    
    # Step 1.3: Select best channels using SCNNs
    logger.info("\nStep 1.3: Training SCNNs and selecting best channels...")
    selected_indices, selected_names, selection_results = select_best_channels(
        channel_data_list=channel_data_list,
        model_builder=build_shallow_cnn_selector,
        K=K_selected,
        ch_names=processed_ch_names
    )
    
    # Save Stage 1 results
    results['stage1_results'] = {
        'selected_indices': selected_indices,
        'selected_names': selected_names,
        'selection_results': selection_results
    }
    
    if save_results:
        save_channel_selection_results(
            selection_results, 
            f"{results_prefix}_stage1_channel_selection.npz"
        )
    
    # ========================================
    # STAGE 2: FUSION CNN CLASSIFICATION
    # ========================================
    logger.info("\n" + "="*50)
    logger.info("STAGE 2: MULTI-LAYER FUSION CNN CLASSIFICATION")
    logger.info("="*50)
    
    # Step 2.1: Filter raw data to selected channels
    logger.info(f"\nStep 2.1: Filtering data to {K_selected} selected channels...")
    if raw_eeg_data.ndim == 3:
        selected_raw_data = raw_eeg_data[:, selected_indices, :]
    else:
        selected_raw_data = raw_eeg_data[selected_indices, :]
    
    logger.info(f"Selected channels data shape: {selected_raw_data.shape}")
    
    # Step 2.2: Prepare fusion inputs (4 frequency bands)
    logger.info("\nStep 2.2: Preparing fusion inputs for frequency bands...")
    fusion_inputs = prepare_fusion_input(selected_raw_data, sampling_rate, selected_names)
    
    # Log fusion input shapes
    for i, (low, high) in enumerate(config.fusion_bands):
        logger.info(f"  Band {i+1} ({low}-{high} Hz): {fusion_inputs[i].shape}")
    
    # Step 2.3: Cross-validation training
    logger.info(f"\nStep 2.3: Training Fusion CNN with {cv_folds}-fold cross-validation...")
    
    cv_results = []
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(labels)):
        logger.info(f"\n--- Cross-Validation Fold {fold + 1}/{cv_folds} ---")
        
        # Split data for this fold
        train_fusion_inputs = [inp[train_idx] for inp in fusion_inputs]
        val_fusion_inputs = [inp[val_idx] for inp in fusion_inputs]
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        
        # Create data loaders for this fold
        train_loader = FusionDataLoader(train_fusion_inputs, train_labels, config.batch_size, shuffle=True)
        val_loader = FusionDataLoader(val_fusion_inputs, val_labels, config.batch_size, shuffle=False)
        
        # Build and train model
        model = build_fusion_cnn_classifier(
            K_channels=K_selected,
            input_time_samples=fusion_inputs[0].shape[2],
            n_classes=len(np.unique(labels))
        )
        
        trainer = FusionCNNTrainer(model)
        fold_accuracy = trainer.fit(train_loader, val_loader)
        
        cv_results.append({
            'fold': fold + 1,
            'accuracy': fold_accuracy,
            'train_history': {
                'train_losses': trainer.train_losses,
                'val_losses': trainer.val_losses,
                'val_accuracies': trainer.val_accuracies
            }
        })
        
        logger.info(f"Fold {fold + 1} accuracy: {fold_accuracy:.4f}")
    
    # Calculate final statistics
    cv_accuracies = [result['accuracy'] for result in cv_results]
    mean_accuracy = np.mean(cv_accuracies)
    std_accuracy = np.std(cv_accuracies)
    
    # Save Stage 2 results
    results['stage2_results'] = {
        'cv_results': cv_results,
        'cv_accuracies': cv_accuracies,
        'fusion_input_shapes': [inp.shape for inp in fusion_inputs]
    }
    
    # ========================================
    # FINAL RESULTS
    # ========================================
    results['final_results'] = {
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'best_fold_accuracy': max(cv_accuracies),
        'worst_fold_accuracy': min(cv_accuracies),
        'selected_channels': {
            'indices': selected_indices,
            'names': selected_names,
            'count': K_selected
        },
        'execution_time': time.time() - start_time
    }
    
    # ========================================
    # FINAL REPORTING
    # ========================================
    logger.info("\n" + "="*60)
    logger.info("PIPELINE EXECUTION COMPLETED")
    logger.info("="*60)
    
    logger.info(f"\nSelected Channels ({K_selected}/{N_channels}):")
    for i, (idx, name) in enumerate(zip(selected_indices, selected_names)):
        acc = selection_results['selected_accuracies'][i]
        logger.info(f"  {i+1}. {name} (Index {idx}): SCNN Accuracy = {acc:.4f}")
    
    logger.info(f"\nCross-Validation Results ({cv_folds} folds):")
    for i, acc in enumerate(cv_accuracies):
        logger.info(f"  Fold {i+1}: {acc:.4f}")
    
    logger.info(f"\nFinal Classification Performance:")
    logger.info(f"  Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    logger.info(f"  Best Fold: {max(cv_accuracies):.4f}")
    logger.info(f"  Worst Fold: {min(cv_accuracies):.4f}")
    
    logger.info(f"\nExecution Time: {results['final_results']['execution_time']:.2f} seconds")
    
    # Save complete results
    if save_results:
        import pickle
        with open(f"{results_prefix}_complete_results.pkl", 'wb') as f:
            pickle.dump(results, f)
        logger.info(f"\nComplete results saved to {results_prefix}_complete_results.pkl")
    
    return results


def load_pipeline_results(filepath: str) -> Dict:
    """
    Load complete pipeline results from file.
    
    Args:
        filepath: Path to results file
        
    Returns:
        Complete results dictionary
    """
    import pickle
    with open(filepath, 'rb') as f:
        results = pickle.load(f)
    
    logger.info(f"Pipeline results loaded from {filepath}")
    return results


def analyze_pipeline_results(results: Dict):
    """
    Analyze and visualize pipeline results.
    
    Args:
        results: Complete results dictionary from run_full_pipeline
    """
    logger.info("\n" + "="*50)
    logger.info("PIPELINE RESULTS ANALYSIS")
    logger.info("="*50)
    
    # Stage 1 Analysis
    stage1 = results['stage1_results']
    selection_summary = stage1['selection_results']['selection_summary']
    
    logger.info("\nStage 1 - Channel Selection Analysis:")
    logger.info(f"  Total channels evaluated: {selection_summary['total_channels']}")
    logger.info(f"  Channels selected: {selection_summary['selected_channels']}")
    logger.info(f"  Best SCNN accuracy: {selection_summary['best_accuracy']:.4f}")
    logger.info(f"  Worst selected accuracy: {selection_summary['worst_selected_accuracy']:.4f}")
    logger.info(f"  Mean selected accuracy: {selection_summary['mean_selected_accuracy']:.4f}")
    
    # Stage 2 Analysis
    stage2 = results['stage2_results']
    final = results['final_results']
    
    logger.info(f"\nStage 2 - Fusion CNN Analysis:")
    logger.info(f"  Cross-validation folds: {len(stage2['cv_accuracies'])}")
    logger.info(f"  Mean accuracy: {final['mean_accuracy']:.4f} ± {final['std_accuracy']:.4f}")
    logger.info(f"  Accuracy range: {final['worst_fold_accuracy']:.4f} - {final['best_fold_accuracy']:.4f}")
    
    # Performance comparison
    improvement = final['mean_accuracy'] - selection_summary['mean_selected_accuracy']
    logger.info(f"\nPerformance Improvement:")
    logger.info(f"  From SCNN selection to Fusion CNN: {improvement:.4f}")
    logger.info(f"  Relative improvement: {improvement/selection_summary['mean_selected_accuracy']*100:.2f}%")


if __name__ == "__main__":
    # Example usage with synthetic data
    logger.info("Running pipeline with synthetic data for demonstration...")
    
    # Generate synthetic EEG data
    n_trials = 200
    n_channels = 22
    time_samples = 1001
    n_classes = 4
    
    # Create synthetic data
    np.random.seed(42)
    synthetic_data = np.random.randn(n_trials, n_channels, time_samples)
    synthetic_labels = np.random.randint(0, n_classes, n_trials)
    
    # Run pipeline
    results = run_full_pipeline(
        raw_eeg_data=synthetic_data,
        labels=synthetic_labels,
        N_channels=n_channels,
        K_selected=6,
        sampling_rate=250,
        cv_folds=3,  # Reduced for demo
        save_results=True,
        results_prefix="synthetic_demo"
    )
    
    # Analyze results
    analyze_pipeline_results(results)