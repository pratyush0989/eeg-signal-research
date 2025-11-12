"""
Comprehensive example script demonstrating the complete EEG Channel Selection pipeline.

This script shows how to:
1. Load and prepare EEG data
2. Configure the pipeline parameters
3. Run the complete two-stage process
4. Analyze and visualize results
5. Save and load results for future use

Author: EEG Analysis Team
Version: 1.0.0
"""

import numpy as np
import logging
from pathlib import Path
import argparse

# Import pipeline components
from pipeline import run_full_pipeline, analyze_pipeline_results, load_pipeline_results
from demo import generate_synthetic_eeg_data
from config import config, EEGConfig
from utils import (
    plot_channel_selection_results,
    plot_training_history, 
    generate_comprehensive_report,
    create_summary_statistics_table
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_1_basic_usage():
    """
    Example 1: Basic pipeline usage with synthetic data
    """
    logger.info("=" * 60)
    logger.info("EXAMPLE 1: BASIC PIPELINE USAGE")
    logger.info("=" * 60)
    
    # Generate synthetic EEG data
    logger.info("Generating synthetic EEG data...")
    eeg_data, labels, ch_names = generate_synthetic_eeg_data(
        n_trials=100,
        n_channels=16,
        time_samples=1001,
        n_classes=4,
        sampling_rate=250
    )
    
    # Run pipeline with default configuration
    logger.info("Running pipeline with default configuration...")
    results = run_full_pipeline(
        raw_eeg_data=eeg_data,
        labels=labels,
        N_channels=len(ch_names),
        ch_names=ch_names,
        save_results=True,
        results_prefix="example1"
    )
    
    # Display results
    logger.info("Pipeline completed! Results summary:")
    final_results = results['final_results']
    logger.info(f"  Selected {len(results['stage1_results']['selected_indices'])} channels")
    logger.info(f"  Final accuracy: {final_results['mean_accuracy']:.4f} ± {final_results['std_accuracy']:.4f}")
    logger.info(f"  Execution time: {final_results['execution_time']:.2f} seconds")
    
    return results


def example_2_custom_configuration():
    """
    Example 2: Using custom configuration
    """
    logger.info("\n" + "=" * 60)
    logger.info("EXAMPLE 2: CUSTOM CONFIGURATION")
    logger.info("=" * 60)
    
    # Create custom configuration
    custom_config = EEGConfig(
        # Channel selection parameters
        K_selected=8,               # Select more channels
        sampling_rate=500,          # Higher sampling rate
        
        # SCNN architecture parameters
        scnn_filters_temporal=40,   # Fewer temporal filters
        scnn_filters_pointwise1=25, # Adjusted pointwise filters
        scnn_filters_pointwise2=15,
        scnn_dropout_rate=0.3,      # Lower dropout
        
        # Training parameters
        max_epochs=30,              # More epochs
        patience=10,                # More patience
        batch_size=16,              # Smaller batch size
        cv_folds=3,                 # Fewer folds for speed
        
        # Fusion CNN parameters  
        fusion_mlp_units=[100, 50], # Larger MLP
    )
    
    logger.info("Custom configuration parameters:")
    logger.info(f"  Channels to select: {custom_config.K_selected}")
    logger.info(f"  Sampling rate: {custom_config.sampling_rate} Hz")
    logger.info(f"  SCNN temporal filters: {custom_config.scnn_filters_temporal}")
    logger.info(f"  Max epochs: {custom_config.max_epochs}")
    logger.info(f"  MLP units: {custom_config.fusion_mlp_units}")
    
    # Generate data matching custom config
    eeg_data, labels, ch_names = generate_synthetic_eeg_data(
        n_trials=80,
        n_channels=20,
        time_samples=int(custom_config.segment_duration * custom_config.sampling_rate) + 1,
        n_classes=custom_config.n_classes,
        sampling_rate=custom_config.sampling_rate
    )
    
    # Temporarily update global config
    original_config_dict = config.__dict__.copy()
    config.__dict__.update(custom_config.__dict__)
    
    try:
        # Run pipeline with custom configuration
        results = run_full_pipeline(
            raw_eeg_data=eeg_data,
            labels=labels,
            N_channels=len(ch_names),
            ch_names=ch_names,
            save_results=True,
            results_prefix="example2_custom"
        )
        
        logger.info("Custom configuration pipeline completed!")
        
    finally:
        # Restore original configuration
        config.__dict__.update(original_config_dict)
    
    return results


def example_3_analysis_and_visualization():
    """
    Example 3: Comprehensive analysis and visualization
    """
    logger.info("\n" + "=" * 60)
    logger.info("EXAMPLE 3: ANALYSIS AND VISUALIZATION")
    logger.info("=" * 60)
    
    # Generate data
    eeg_data, labels, ch_names = generate_synthetic_eeg_data(
        n_trials=120,
        n_channels=12,
        time_samples=751,
        n_classes=3
    )
    
    # Run pipeline
    results = run_full_pipeline(
        raw_eeg_data=eeg_data,
        labels=labels,
        ch_names=ch_names,
        cv_folds=4,
        save_results=True,
        results_prefix="example3_analysis"
    )
    
    # Perform detailed analysis
    logger.info("Performing detailed analysis...")
    analyze_pipeline_results(results)
    
    # Create summary table
    summary_table = create_summary_statistics_table(results)
    logger.info("\nPipeline Summary Statistics:")
    logger.info("\n" + summary_table.to_string(index=False))
    
    # Generate visualizations (if matplotlib available)
    try:
        logger.info("Generating visualizations...")
        
        # Channel selection visualization
        plot_channel_selection_results(
            results['stage1_results']['selection_results'],
            save_path="example3_channel_selection.png"
        )
        
        # Training history visualization
        plot_training_history(
            results['stage2_results']['cv_results'],
            save_path="example3_training_history.png"
        )
        
        # Generate comprehensive report
        generate_comprehensive_report(results, "example3_analysis_report")
        
        logger.info("Visualizations saved successfully!")
        
    except ImportError:
        logger.warning("Matplotlib/seaborn not available. Skipping visualizations.")
    
    return results


def example_4_loading_and_reanalysis():
    """
    Example 4: Loading saved results and performing reanalysis
    """
    logger.info("\n" + "=" * 60)
    logger.info("EXAMPLE 4: LOADING AND REANALYSIS")
    logger.info("=" * 60)
    
    # Check if previous results exist
    result_files = list(Path(".").glob("example*_complete_results.pkl"))
    
    if not result_files:
        logger.info("No previous results found. Running a quick pipeline first...")
        eeg_data, labels, ch_names = generate_synthetic_eeg_data(
            n_trials=50, n_channels=8, time_samples=501, n_classes=2
        )
        results = run_full_pipeline(
            eeg_data, labels, ch_names=ch_names, cv_folds=2, 
            save_results=True, results_prefix="example4_temp"
        )
        result_file = "example4_temp_complete_results.pkl"
    else:
        result_file = str(result_files[0])
        logger.info(f"Loading existing results from {result_file}")
    
    # Load results
    loaded_results = load_pipeline_results(result_file)
    
    # Reanalyze loaded results
    logger.info("Reanalyzing loaded results...")
    analyze_pipeline_results(loaded_results)
    
    # Extract key insights
    stage1 = loaded_results['stage1_results']['selection_results']
    stage2 = loaded_results['stage2_results']
    final = loaded_results['final_results']
    
    logger.info("\nKey Insights from Reanalysis:")
    logger.info(f"  Dataset: {loaded_results['pipeline_config']['input_shape']}")
    logger.info(f"  Selected channels: {stage1['selected_names']}")
    logger.info(f"  Channel selection improved performance by: {final['mean_accuracy'] - stage1['selection_summary']['mean_selected_accuracy']:.4f}")
    logger.info(f"  Best performing fold: {final['best_fold_accuracy']:.4f}")
    logger.info(f"  Cross-validation consistency (std): {final['std_accuracy']:.4f}")
    
    return loaded_results


def example_5_batch_processing():
    """
    Example 5: Batch processing multiple datasets
    """
    logger.info("\n" + "=" * 60)
    logger.info("EXAMPLE 5: BATCH PROCESSING")
    logger.info("=" * 60)
    
    # Simulate multiple datasets with different characteristics
    datasets = [
        {"name": "Dataset_A", "n_trials": 80, "n_channels": 16, "n_classes": 2},
        {"name": "Dataset_B", "n_trials": 120, "n_channels": 22, "n_classes": 4},
        {"name": "Dataset_C", "n_trials": 60, "n_channels": 14, "n_classes": 3},
    ]
    
    batch_results = []
    
    for i, dataset_config in enumerate(datasets):
        logger.info(f"\nProcessing {dataset_config['name']} ({i+1}/{len(datasets)})...")
        
        # Generate dataset
        eeg_data, labels, ch_names = generate_synthetic_eeg_data(
            n_trials=dataset_config['n_trials'],
            n_channels=dataset_config['n_channels'],
            n_classes=dataset_config['n_classes'],
            time_samples=501,  # Shorter for speed
            random_seed=42 + i  # Different seed for each dataset
        )
        
        # Run pipeline with reduced parameters for speed
        results = run_full_pipeline(
            raw_eeg_data=eeg_data,
            labels=labels,
            ch_names=ch_names,
            cv_folds=3,  # Reduced for speed
            save_results=True,
            results_prefix=f"batch_{dataset_config['name']}"
        )
        
        # Store results
        batch_results.append({
            'dataset_name': dataset_config['name'],
            'config': dataset_config,
            'results': results
        })
        
        logger.info(f"  {dataset_config['name']}: {results['final_results']['mean_accuracy']:.4f} accuracy")
    
    # Compare batch results
    logger.info("\nBatch Processing Summary:")
    logger.info("-" * 40)
    for batch_result in batch_results:
        name = batch_result['dataset_name']
        config_info = batch_result['config']
        accuracy = batch_result['results']['final_results']['mean_accuracy']
        
        logger.info(f"{name}: {accuracy:.4f} "
                   f"({config_info['n_trials']} trials, {config_info['n_channels']} channels, "
                   f"{config_info['n_classes']} classes)")
    
    return batch_results


def main():
    """
    Main function to run examples
    """
    parser = argparse.ArgumentParser(description="EEG Channel Selection Examples")
    parser.add_argument(
        "--example", type=int, choices=[1, 2, 3, 4, 5], default=1,
        help="Example to run (1-5)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all examples sequentially"
    )
    
    args = parser.parse_args()
    
    logger.info("EEG CHANNEL SELECTION AND CLASSIFICATION - COMPREHENSIVE EXAMPLES")
    logger.info("=" * 70)
    
    examples = {
        1: example_1_basic_usage,
        2: example_2_custom_configuration, 
        3: example_3_analysis_and_visualization,
        4: example_4_loading_and_reanalysis,
        5: example_5_batch_processing
    }
    
    if args.all:
        logger.info("Running all examples...")
        for example_num in sorted(examples.keys()):
            try:
                examples[example_num]()
            except Exception as e:
                logger.error(f"Example {example_num} failed: {e}")
    else:
        logger.info(f"Running Example {args.example}...")
        try:
            results = examples[args.example]()
            logger.info(f"\nExample {args.example} completed successfully!")
        except Exception as e:
            logger.error(f"Example {args.example} failed: {e}")
            raise
    
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLES COMPLETED")
    logger.info("=" * 70)
    logger.info("Check generated files:")
    logger.info("  - example*_complete_results.pkl: Full pipeline results")
    logger.info("  - example*_stage1_channel_selection.npz: Channel selection results")
    logger.info("  - example*.png: Visualization plots")
    logger.info("  - example*_analysis_report/: Comprehensive reports")


if __name__ == "__main__":
    main()