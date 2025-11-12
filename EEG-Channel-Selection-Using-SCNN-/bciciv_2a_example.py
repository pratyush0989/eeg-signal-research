"""
Example script for training and testing the EEG pipeline with BCI Competition IV Dataset 2a

This script demonstrates:
1. Loading BCICIV-2a dataset
2. Preprocessing for the pipeline
3. Running the complete two-stage pipeline
4. Evaluating results with train/test split
5. Comparing results across different subjects

Usage:
    python bciciv_2a_example.py --data_path /path/to/bciciv_2a --subject A01
    python bciciv_2a_example.py --data_path /path/to/bciciv_2a --all_subjects
"""

import numpy as np
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Dict

# Import pipeline components
from pipeline import run_full_pipeline, analyze_pipeline_results
from config import config, EEGConfig
from bciciv_2a_loader import (
    load_bciciv_2a_subject,
    load_all_subjects_bciciv_2a, 
    create_train_test_split_bciciv_2a,
    preprocess_bciciv_2a_for_pipeline,
    save_bciciv_2a_processed,
    BCICIV_2A_CHANNELS
)
from utils import (
    plot_channel_selection_results,
    plot_training_history,
    generate_comprehensive_report,
    evaluate_model_predictions
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_bciciv_2a_config() -> EEGConfig:
    """
    Create optimized configuration for BCI Competition IV Dataset 2a.
    
    Returns:
        EEGConfig optimized for BCICIV-2a dataset
    """
    bciciv_config = EEGConfig(
        # Dataset specific parameters
        sampling_rate=250,
        n_channels_total=22,  # EEG channels only
        K_selected=8,         # Select 8 most informative channels
        n_classes=4,          # 4 motor imagery classes
        
        # Optimized training parameters for BCICIV-2a
        max_epochs=50,
        patience=15,
        batch_size=32,
        cv_folds=5,
        
        # SCNN parameters tuned for motor imagery
        scnn_learning_rate=5e-4,
        scnn_filters_temporal=40,
        scnn_filters_pointwise1=25,
        scnn_filters_pointwise2=15,
        scnn_dropout_rate=0.4,
        
        # Fusion CNN parameters
        fusion_learning_rate=1e-3,
        fusion_mlp_units=[100, 50],
        
        # Preprocessing parameters
        target_freq=(8, 30),  # Focus on mu and beta rhythms for motor imagery
        segment_duration=3.0   # 3-second segments
    )
    
    logger.info("BCICIV-2a optimized configuration created:")
    logger.info(f"  Channels to select: {bciciv_config.K_selected}")
    logger.info(f"  Sampling rate: {bciciv_config.sampling_rate} Hz")
    logger.info(f"  Target frequency band: {bciciv_config.target_freq} Hz")
    logger.info(f"  Cross-validation folds: {bciciv_config.cv_folds}")
    
    return bciciv_config


def train_single_subject(
    data_path: str,
    subject_id: str,
    config: EEGConfig,
    save_results: bool = True
) -> Dict:
    """
    Train the pipeline on a single BCICIV-2a subject.
    
    Args:
        data_path: Path to BCICIV-2a dataset
        subject_id: Subject ID (e.g., 'A01')
        config: Configuration to use
        save_results: Whether to save results
        
    Returns:
        Dictionary with pipeline results
    """
    logger.info(f"="*60)
    logger.info(f"TRAINING SINGLE SUBJECT: {subject_id}")
    logger.info(f"="*60)
    
    # Load training data for the subject
    logger.info(f"Loading training data for {subject_id}...")
    eeg_data, labels, channel_names = load_bciciv_2a_subject(
        data_path=data_path,
        subject_id=subject_id,
        session='T',  # Training session
        use_eeg_only=True
    )
    
    # Preprocess for pipeline
    processed_data, processed_labels, channel_names = preprocess_bciciv_2a_for_pipeline(
        eeg_data, labels, channel_names, 
        target_length_sec=config.segment_duration,
        sampling_rate=config.sampling_rate
    )
    
    # Update global config temporarily
    original_config = config.__dict__.copy()
    config.__dict__.update(config.__dict__)
    
    try:
        # Run the complete pipeline
        results = run_full_pipeline(
            raw_eeg_data=processed_data,
            labels=processed_labels,
            N_channels=len(channel_names),
            K_selected=config.K_selected,
            ch_names=channel_names,
            sampling_rate=config.sampling_rate,
            cv_folds=config.cv_folds,
            save_results=save_results,
            results_prefix=f"bciciv_2a_{subject_id}"
        )
        
        # Add subject-specific information
        results['subject_info'] = {
            'subject_id': subject_id,
            'dataset': 'BCI Competition IV Dataset 2a',
            'session': 'Training',
            'original_trials': len(labels),
            'processed_trials': len(processed_labels),
            'channel_names': channel_names
        }
        
        # Analyze results
        logger.info(f"\n{subject_id} Results Summary:")
        analyze_pipeline_results(results)
        
        return results
        
    finally:
        # Restore original config
        config.__dict__.update(original_config)


def train_test_split_evaluation(
    data_path: str,
    subject_ids: Optional[List[str]] = None,
    config: EEGConfig = None,
    test_ratio: float = 0.3
) -> Dict:
    """
    Evaluate pipeline using train/test split on BCICIV-2a data.
    
    Args:
        data_path: Path to BCICIV-2a dataset
        subject_ids: List of subject IDs to include
        config: Configuration to use
        test_ratio: Fraction of data to use for testing
        
    Returns:
        Dictionary with evaluation results
    """
    if config is None:
        config = setup_bciciv_2a_config()
    
    logger.info(f"="*60)
    logger.info(f"TRAIN/TEST SPLIT EVALUATION")
    logger.info(f"="*60)
    
    # Create train/test split
    logger.info("Creating train/test split...")
    (train_data, train_labels, channel_names), (test_data, test_labels, _) = create_train_test_split_bciciv_2a(
        data_path=data_path,
        subject_ids=subject_ids,
        test_ratio=test_ratio,
        use_eeg_only=True
    )
    
    # Update global config
    original_config_dict = globals()['config'].__dict__.copy()
    globals()['config'].__dict__.update(config.__dict__)
    
    try:
        # Train on training set
        logger.info("Training on training set...")
        train_results = run_full_pipeline(
            raw_eeg_data=train_data,
            labels=train_labels,
            N_channels=len(channel_names),
            K_selected=config.K_selected,
            ch_names=channel_names,
            sampling_rate=config.sampling_rate,
            cv_folds=config.cv_folds,
            save_results=True,
            results_prefix="bciciv_2a_train_test"
        )
        
        # Get selected channels from training
        selected_indices = train_results['stage1_results']['selected_indices']
        selected_names = train_results['stage1_results']['selected_names']
        
        logger.info(f"Selected channels: {selected_names}")
        
        # Evaluate on test set (simplified - just report what we would do)
        logger.info(f"Test set available with {len(test_labels)} trials")
        logger.info("For full test evaluation, you would:")
        logger.info("1. Use the selected channels from training")  
        logger.info("2. Retrain fusion CNN on selected channels from training data")
        logger.info("3. Evaluate the trained model on test data")
        
        # Create comprehensive results
        evaluation_results = {
            'train_results': train_results,
            'selected_channels': {
                'indices': selected_indices,
                'names': selected_names
            },
            'data_info': {
                'train_shape': train_data.shape,
                'test_shape': test_data.shape,
                'train_class_dist': np.bincount(train_labels).tolist(),
                'test_class_dist': np.bincount(test_labels).tolist(),
                'subjects_included': subject_ids or "all available",
                'test_ratio': test_ratio
            }
        }
        
        return evaluation_results
        
    finally:
        # Restore original config
        globals()['config'].__dict__.update(original_config_dict)


def compare_subjects(
    data_path: str,
    subject_ids: List[str],
    config: EEGConfig = None
) -> Dict:
    """
    Compare pipeline performance across multiple BCICIV-2a subjects.
    
    Args:
        data_path: Path to BCICIV-2a dataset
        subject_ids: List of subject IDs to compare
        config: Configuration to use
        
    Returns:
        Dictionary with comparison results
    """
    if config is None:
        config = setup_bciciv_2a_config()
    
    logger.info(f"="*60)
    logger.info(f"COMPARING SUBJECTS: {', '.join(subject_ids)}")
    logger.info(f"="*60)
    
    comparison_results = {
        'subjects': {},
        'summary': {}
    }
    
    all_accuracies = []
    all_selected_channels = []
    
    for subject_id in subject_ids:
        logger.info(f"\nProcessing {subject_id}...")
        
        try:
            # Train on this subject
            subject_results = train_single_subject(
                data_path=data_path,
                subject_id=subject_id,
                config=config,
                save_results=True
            )
            
            # Store results
            comparison_results['subjects'][subject_id] = {
                'final_accuracy': subject_results['final_results']['mean_accuracy'],
                'std_accuracy': subject_results['final_results']['std_accuracy'],
                'selected_channels': subject_results['stage1_results']['selected_names'],
                'best_fold_accuracy': subject_results['final_results']['best_fold_accuracy'],
                'execution_time': subject_results['final_results']['execution_time']
            }
            
            all_accuracies.append(subject_results['final_results']['mean_accuracy'])
            all_selected_channels.extend(subject_results['stage1_results']['selected_names'])
            
            logger.info(f"{subject_id} completed: {subject_results['final_results']['mean_accuracy']:.4f} accuracy")
            
        except Exception as e:
            logger.error(f"Failed to process {subject_id}: {str(e)}")
            comparison_results['subjects'][subject_id] = {'error': str(e)}
    
    # Calculate summary statistics
    if all_accuracies:
        comparison_results['summary'] = {
            'mean_accuracy_across_subjects': np.mean(all_accuracies),
            'std_accuracy_across_subjects': np.std(all_accuracies),
            'best_subject_accuracy': max(all_accuracies),
            'worst_subject_accuracy': min(all_accuracies),
            'most_selected_channels': {},
            'total_subjects': len(subject_ids),
            'successful_subjects': len(all_accuracies)
        }
        
        # Count most frequently selected channels
        from collections import Counter
        channel_counts = Counter(all_selected_channels)
        comparison_results['summary']['most_selected_channels'] = dict(channel_counts.most_common(10))
        
        # Log summary
        logger.info(f"\n" + "="*50)
        logger.info("CROSS-SUBJECT COMPARISON SUMMARY")
        logger.info("="*50)
        logger.info(f"Subjects processed: {len(all_accuracies)}/{len(subject_ids)}")
        logger.info(f"Mean accuracy: {comparison_results['summary']['mean_accuracy_across_subjects']:.4f} ± {comparison_results['summary']['std_accuracy_across_subjects']:.4f}")
        logger.info(f"Best subject: {comparison_results['summary']['best_subject_accuracy']:.4f}")
        logger.info(f"Worst subject: {comparison_results['summary']['worst_subject_accuracy']:.4f}")
        
        logger.info(f"\nMost frequently selected channels:")
        for channel, count in list(channel_counts.most_common(5)):
            logger.info(f"  {channel}: {count} times")
    
    return comparison_results


def generate_bciciv_2a_report(
    results: Dict,
    output_dir: str = "bciciv_2a_analysis_report"
):
    """
    Generate comprehensive analysis report for BCICIV-2a results.
    
    Args:
        results: Pipeline results dictionary
        output_dir: Directory to save report
    """
    logger.info(f"Generating comprehensive BCICIV-2a analysis report...")
    
    try:
        # Generate standard report
        generate_comprehensive_report(results, output_dir)
        
        # Add BCICIV-2a specific analysis
        import json
        from pathlib import Path
        
        report_dir = Path(output_dir)
        report_dir.mkdir(exist_ok=True)
        
        # Save detailed results as JSON
        if 'subject_info' in results:
            with open(report_dir / "subject_info.json", 'w') as f:
                json.dump(results['subject_info'], f, indent=2)
        
        # Create summary text report
        with open(report_dir / "bciciv_2a_summary.txt", 'w') as f:
            f.write("BCI Competition IV Dataset 2a - Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            
            if 'subject_info' in results:
                f.write(f"Subject: {results['subject_info']['subject_id']}\n")
                f.write(f"Original trials: {results['subject_info']['original_trials']}\n")
                f.write(f"Processed trials: {results['subject_info']['processed_trials']}\n\n")
            
            # Pipeline results
            final = results['final_results']
            f.write(f"Final Classification Performance:\n")
            f.write(f"  Mean Accuracy: {final['mean_accuracy']:.4f} ± {final['std_accuracy']:.4f}\n")
            f.write(f"  Best Fold: {final['best_fold_accuracy']:.4f}\n")
            f.write(f"  Execution Time: {final['execution_time']:.2f} seconds\n\n")
            
            # Selected channels
            stage1 = results['stage1_results']
            f.write(f"Selected Channels:\n")
            for i, (name, acc) in enumerate(zip(stage1['selected_names'], 
                                              stage1['selection_results']['selected_accuracies'])):
                f.write(f"  {i+1}. {name}: {acc:.4f}\n")
        
        logger.info(f"BCICIV-2a analysis report generated in {output_dir}/")
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")


def main():
    """Main function for BCICIV-2a pipeline execution"""
    parser = argparse.ArgumentParser(description="BCI Competition IV Dataset 2a EEG Pipeline")
    parser.add_argument("--data_path", type=str, required=True, help="Path to BCICIV-2a dataset directory")
    parser.add_argument("--subject", type=str, help="Single subject ID (e.g., A01)")
    parser.add_argument("--subjects", nargs='+', help="Multiple subject IDs (e.g., A01 A02 A03)")
    parser.add_argument("--all_subjects", action="store_true", help="Use all available subjects (A01-A09)")
    parser.add_argument("--mode", type=str, choices=['single', 'compare', 'train_test'], default='single',
                       help="Execution mode")
    parser.add_argument("--test_ratio", type=float, default=0.3, help="Test set ratio for train_test mode")
    parser.add_argument("--output_dir", type=str, default="bciciv_2a_results", help="Output directory")
    parser.add_argument("--config_file", type=str, help="Path to custom configuration file")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == 'single' and not args.subject:
        parser.error("Single mode requires --subject argument")
    
    # Setup configuration
    bciciv_config = setup_bciciv_2a_config()
    
    # Determine subjects to process
    if args.all_subjects:
        subject_ids = [f'A{i:02d}' for i in range(1, 10)]
    elif args.subjects:
        subject_ids = args.subjects
    elif args.subject:
        subject_ids = [args.subject]
    else:
        subject_ids = ['A01']  # Default
    
    logger.info(f"BCICIV-2a Pipeline Execution")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Subjects: {subject_ids}")
    
    # Execute based on mode
    try:
        if args.mode == 'single':
            results = train_single_subject(
                data_path=args.data_path,
                subject_id=subject_ids[0],
                config=bciciv_config,
                save_results=True
            )
            generate_bciciv_2a_report(results, f"{args.output_dir}_single")
            
        elif args.mode == 'compare':
            results = compare_subjects(
                data_path=args.data_path,
                subject_ids=subject_ids,
                config=bciciv_config
            )
            
            # Save comparison results
            import json
            with open(f"{args.output_dir}_comparison.json", 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
        elif args.mode == 'train_test':
            results = train_test_split_evaluation(
                data_path=args.data_path,
                subject_ids=subject_ids,
                config=bciciv_config,
                test_ratio=args.test_ratio
            )
            generate_bciciv_2a_report(results['train_results'], f"{args.output_dir}_train_test")
        
        logger.info("BCICIV-2a pipeline execution completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()