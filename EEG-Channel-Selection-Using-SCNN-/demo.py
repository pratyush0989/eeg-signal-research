"""
Example usage and demo script for the EEG Channel Selection and Classification pipeline
"""
import numpy as np
import torch
import logging
from typing import Optional

from config import config
from pipeline import run_full_pipeline, analyze_pipeline_results, load_pipeline_results
from utils import (
    plot_channel_selection_results, 
    plot_training_history,
    generate_comprehensive_report,
    create_summary_statistics_table
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_synthetic_eeg_data(
    n_trials: int = 200,
    n_channels: int = 22, 
    time_samples: int = 1001,
    n_classes: int = 4,
    sampling_rate: int = 250,
    noise_level: float = 0.1,
    signal_strength: float = 1.0,
    random_seed: Optional[int] = 42
) -> tuple:
    """
    Generate synthetic EEG data for testing the pipeline.
    
    Args:
        n_trials: Number of trials/epochs
        n_channels: Number of EEG channels
        time_samples: Number of time samples per trial
        n_classes: Number of motor imagery classes
        sampling_rate: Sampling rate in Hz
        noise_level: Standard deviation of background noise
        signal_strength: Amplitude of class-specific signals
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (eeg_data, labels, channel_names)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    logger.info(f"Generating synthetic EEG data:")
    logger.info(f"  Trials: {n_trials}, Channels: {n_channels}, Time samples: {time_samples}")
    logger.info(f"  Classes: {n_classes}, Sampling rate: {sampling_rate} Hz")
    
    # Generate time axis
    time_axis = np.linspace(0, time_samples/sampling_rate, time_samples)
    
    # Initialize data array
    eeg_data = np.zeros((n_trials, n_channels, time_samples))
    labels = np.random.randint(0, n_classes, n_trials)
    
    # Define some channels as "important" for each class (adjust for available channels)
    max_ch = min(n_channels - 1, 15)
    important_channels_per_class = {
        0: [i for i in [1, 2, 7, 8] if i < n_channels],      # Class 0: Motor areas
        1: [i for i in [3, 4, 9, 10] if i < n_channels],     # Class 1: Different motor areas  
        2: [i for i in [5, 6, 11, 12] if i < n_channels],    # Class 2: Sensorimotor areas
        3: [i for i in [0, 13, 14, 15] if i < n_channels]    # Class 3: Other regions
    }
    
    # Ensure each class has at least one channel
    for class_id in range(n_classes):
        if len(important_channels_per_class[class_id]) == 0:
            important_channels_per_class[class_id] = [class_id % n_channels]
    
    # Generate data for each trial
    for trial in range(n_trials):
        trial_class = labels[trial]
        
        # Base noise for all channels
        noise = np.random.normal(0, noise_level, (n_channels, time_samples))
        eeg_data[trial] = noise
        
        # Add class-specific signals to important channels
        important_channels = important_channels_per_class[trial_class]
        
        for ch in important_channels:
            # Generate class-specific oscillatory patterns
            freq1 = 8 + trial_class * 3  # Different frequency for each class
            freq2 = 13 + trial_class * 2
            
            # Create signals with different phases and amplitudes
            signal1 = signal_strength * np.sin(2 * np.pi * freq1 * time_axis + 
                                             np.random.uniform(0, 2*np.pi))
            signal2 = signal_strength * 0.7 * np.sin(2 * np.pi * freq2 * time_axis + 
                                                   np.random.uniform(0, 2*np.pi))
            
            # Add event-related potential-like component
            erp_peak_time = 0.3 + np.random.uniform(-0.1, 0.1)  # Around 300ms
            erp_peak_idx = int(erp_peak_time * sampling_rate)
            erp_width = 50  # samples
            
            if erp_peak_idx - erp_width >= 0 and erp_peak_idx + erp_width < time_samples:
                erp_component = signal_strength * 0.5 * np.exp(
                    -((np.arange(time_samples) - erp_peak_idx)**2) / (2 * (erp_width/3)**2)
                )
                eeg_data[trial, ch] += signal1 + signal2 + erp_component
            else:
                eeg_data[trial, ch] += signal1 + signal2
    
    # Generate channel names
    channel_names = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3',
        'Pz', 'P4', 'T6', 'O1', 'O2', 'Ch19', 'Ch20', 'Ch21'
    ][:n_channels]
    
    # Ensure we have enough channel names
    while len(channel_names) < n_channels:
        channel_names.append(f'Ch{len(channel_names)}')
    
    logger.info(f"Synthetic EEG data generated successfully")
    logger.info(f"  Data shape: {eeg_data.shape}")
    logger.info(f"  Labels shape: {labels.shape}")
    logger.info(f"  Class distribution: {np.bincount(labels)}")
    
    return eeg_data, labels, channel_names


def run_demo_with_synthetic_data():
    """
    Run a complete demo of the pipeline with synthetic data.
    """
    logger.info("="*60)
    logger.info("EEG CHANNEL SELECTION AND CLASSIFICATION DEMO")
    logger.info("="*60)
    
    # Generate synthetic data
    eeg_data, labels, ch_names = generate_synthetic_eeg_data(
        n_trials=150,  # Smaller for faster demo
        n_channels=16,  # Reduced for faster processing
        time_samples=1001,
        n_classes=4,
        sampling_rate=250
    )
    
    # Update config for demo
    config.max_epochs = 20  # Reduced for faster demo
    config.patience = 5
    config.cv_folds = 3
    config.K_selected = 6
    
    logger.info("Updated configuration for demo (faster execution)")
    
    # Run the complete pipeline
    results = run_full_pipeline(
        raw_eeg_data=eeg_data,
        labels=labels,
        N_channels=len(ch_names),
        K_selected=config.K_selected,
        ch_names=ch_names,
        sampling_rate=250,
        cv_folds=config.cv_folds,
        save_results=True,
        results_prefix="demo"
    )
    
    # Analyze results
    analyze_pipeline_results(results)
    
    # Generate visualizations
    logger.info("\nGenerating analysis visualizations...")
    
    try:
        # Channel selection plot
        plot_channel_selection_results(
            results['stage1_results']['selection_results'],
            save_path="demo_channel_selection.png"
        )
        
        # Training history plot
        plot_training_history(
            results['stage2_results']['cv_results'],
            save_path="demo_training_history.png"
        )
        
        # Summary statistics table
        summary_table = create_summary_statistics_table(results)
        logger.info("\nSummary Statistics:")
        logger.info("\n" + summary_table.to_string(index=False))
        
        # Generate comprehensive report
        generate_comprehensive_report(results, "demo_analysis_report")
        
    except Exception as e:
        logger.warning(f"Could not generate all visualizations: {str(e)}")
        logger.warning("This may be due to missing matplotlib or seaborn. Install with:")
        logger.warning("pip install matplotlib seaborn pandas")
    
    return results


def run_demo_with_custom_config():
    """
    Demo showing how to customize the configuration.
    """
    logger.info("="*50)
    logger.info("CUSTOM CONFIGURATION DEMO")
    logger.info("="*50)
    
    # Create custom configuration
    from config import EEGConfig
    
    custom_config = EEGConfig(
        # Data parameters
        sampling_rate=500,  # Higher sampling rate
        K_selected=8,       # More channels
        n_classes=2,        # Binary classification
        
        # SCNN parameters  
        scnn_filters_temporal=40,
        scnn_filters_pointwise1=25,
        scnn_filters_pointwise2=15,
        scnn_dropout_rate=0.3,
        
        # Training parameters
        max_epochs=15,
        patience=10,
        batch_size=16,
        cv_folds=5
    )
    
    logger.info("Custom configuration created:")
    logger.info(f"  Sampling rate: {custom_config.sampling_rate} Hz")
    logger.info(f"  Channels to select: {custom_config.K_selected}")
    logger.info(f"  Number of classes: {custom_config.n_classes}")
    logger.info(f"  SCNN temporal filters: {custom_config.scnn_filters_temporal}")
    
    # Generate data matching custom config
    eeg_data, labels, ch_names = generate_synthetic_eeg_data(
        n_trials=100,
        n_channels=20,
        time_samples=int(custom_config.segment_duration * custom_config.sampling_rate) + 1,
        n_classes=custom_config.n_classes,
        sampling_rate=custom_config.sampling_rate
    )
    
    # Temporarily update global config
    original_config = config.__dict__.copy()
    config.__dict__.update(custom_config.__dict__)
    
    try:
        # Run pipeline with custom config
        results = run_full_pipeline(
            raw_eeg_data=eeg_data,
            labels=labels,
            N_channels=len(ch_names),
            K_selected=custom_config.K_selected,
            ch_names=ch_names,
            sampling_rate=custom_config.sampling_rate,
            cv_folds=custom_config.cv_folds,
            save_results=True,
            results_prefix="custom_demo"
        )
        
        logger.info("Custom configuration demo completed successfully!")
        
    finally:
        # Restore original config
        config.__dict__.update(original_config)
    
    return results


def demonstrate_individual_modules():
    """
    Demonstrate individual module functionality.
    """
    logger.info("="*50)
    logger.info("INDIVIDUAL MODULE DEMONSTRATION")
    logger.info("="*50)
    
    # Generate small dataset for quick testing
    eeg_data, labels, ch_names = generate_synthetic_eeg_data(
        n_trials=50, n_channels=8, time_samples=501, n_classes=2
    )
    
    logger.info("Testing individual modules...")
    
    # Test preprocessing
    from preprocessing import preprocess_data_for_selection, generate_single_channel_inputs
    
    logger.info("\n1. Testing preprocessing module...")
    processed_data, processed_ch_names = preprocess_data_for_selection(
        eeg_data, sampling_rate=250, ch_names=ch_names
    )
    logger.info(f"   Processed data shape: {processed_data.shape}")
    
    # Test single channel input generation
    channel_data_list = generate_single_channel_inputs(processed_data, labels)
    logger.info(f"   Generated {len(channel_data_list)} channel datasets")
    
    # Test SCNN model
    from scnn_model import build_shallow_cnn_selector, SCNNTrainer
    
    logger.info("\n2. Testing SCNN model...")
    scnn_model = build_shallow_cnn_selector(
        input_time_samples=processed_data.shape[2],
        n_classes=len(np.unique(labels))
    )
    logger.info(f"   SCNN model built with {sum(p.numel() for p in scnn_model.parameters())} parameters")
    
    # Test channel selection (with limited channels for speed)
    from channel_selection import select_best_channels
    
    logger.info("\n3. Testing channel selection (first 4 channels only)...")
    limited_channel_data = channel_data_list[:4]  # Test with fewer channels
    selected_indices, selected_names, selection_results = select_best_channels(
        limited_channel_data,
        K=2,  # Select top 2
        ch_names=ch_names[:4],
        max_epochs=5,  # Very limited for demo
        patience=3
    )
    logger.info(f"   Selected channels: {selected_names}")
    
    # Test fusion CNN
    from fusion_cnn import build_fusion_cnn_classifier
    
    logger.info("\n4. Testing Fusion CNN...")
    fusion_model = build_fusion_cnn_classifier(
        K_channels=2,
        input_time_samples=processed_data.shape[2],
        n_classes=len(np.unique(labels))
    )
    logger.info(f"   Fusion CNN built with {sum(p.numel() for p in fusion_model.parameters())} parameters")
    
    logger.info("\nAll individual modules tested successfully!")


if __name__ == "__main__":
    """
    Run different demo modes based on command line arguments or interactively.
    """
    import sys
    
    if len(sys.argv) > 1:
        demo_mode = sys.argv[1]
    else:
        print("\nEEG Channel Selection and Classification Demo")
        print("=" * 50)
        print("Select demo mode:")
        print("1. Full pipeline with synthetic data (recommended)")
        print("2. Custom configuration demo")
        print("3. Individual module testing")
        print("4. All demos")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            demo_mode = "full"
        elif choice == "2":
            demo_mode = "custom"
        elif choice == "3":
            demo_mode = "modules"
        elif choice == "4":
            demo_mode = "all"
        else:
            demo_mode = "full"
            print("Invalid choice, running full demo...")
    
    # Run selected demo(s)
    if demo_mode in ["full", "all"]:
        logger.info("Running full pipeline demo...")
        results = run_demo_with_synthetic_data()
    
    if demo_mode in ["custom", "all"]:
        logger.info("\nRunning custom configuration demo...")
        custom_results = run_demo_with_custom_config()
    
    if demo_mode in ["modules", "all"]:
        logger.info("\nRunning individual module demo...")
        demonstrate_individual_modules()
    
    logger.info("\n" + "="*60)
    logger.info("DEMO COMPLETED SUCCESSFULLY!")
    logger.info("="*60)
    logger.info("Check the generated files:")
    logger.info("  - demo_*.pkl: Pipeline results")
    logger.info("  - demo_*.png: Visualization plots") 
    logger.info("  - demo_analysis_report/: Comprehensive analysis")
    logger.info("="*60)