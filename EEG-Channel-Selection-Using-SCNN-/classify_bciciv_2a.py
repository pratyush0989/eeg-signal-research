"""
Simple BCICIV-2a Classification Script

This script shows you exactly how to:
1. Load your BCICIV-2a dataset
2. Pass it through the complete pipeline
3. Get classification results

Usage:
    python classify_bciciv_2a.py --data_path /path/to/bciciv_2a --subject A01
"""

import argparse
import logging
import numpy as np
import json
from pathlib import Path

# Setup simple logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def classify_bciciv_2a_subject(data_path: str, subject_id: str = "A01"):
    """
    Complete classification pipeline for BCICIV-2a subject.
    
    Args:
        data_path: Path to your BCICIV-2a dataset folder
        subject_id: Subject to classify (A01, A02, ..., A09)
        
    Returns:
        Dictionary with classification results
    """
    
    print(f"\n🧠 EEG Classification Pipeline for Subject {subject_id}")
    print("=" * 60)
    
    # Step 1: Load the dataset
    print(f"\n📂 Step 1: Loading BCICIV-2a data for {subject_id}...")
    try:
        from bciciv_2a_loader import load_bciciv_2a_subject, preprocess_bciciv_2a_for_pipeline
        
        # Load training session data
        eeg_data, labels, channel_names = load_bciciv_2a_subject(
            data_path=data_path,
            subject_id=subject_id,
            session='T',  # Training session
            use_eeg_only=True  # Use only EEG channels (22 channels)
        )
        
        print(f"✅ Data loaded successfully!")
        print(f"   📊 Data shape: {eeg_data.shape}")
        print(f"   🏷️  Labels: {len(labels)} trials")
        print(f"   📡 Channels: {len(channel_names)} EEG channels")
        print(f"   🎯 Classes: {len(np.unique(labels))} (Left hand, Right hand, Feet, Tongue)")
        print(f"   📈 Class distribution: {np.bincount(labels)}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("💡 Make sure your dataset path is correct and contains .mat files like A01T.mat")
        return None
    
    # Step 2: Preprocess for pipeline
    print(f"\n🔄 Step 2: Preprocessing data for pipeline...")
    try:
        processed_data, processed_labels, channel_names = preprocess_bciciv_2a_for_pipeline(
            eeg_data, labels, channel_names,
            target_length_sec=3.0,  # 3-second segments
            sampling_rate=250
        )
        
        print(f"✅ Preprocessing completed!")
        print(f"   📊 Processed shape: {processed_data.shape}")
        print(f"   ⏱️  Segment length: 3.0 seconds (750 samples)")
        
    except Exception as e:
        print(f"❌ Error preprocessing: {e}")
        return None
    
    # Step 3: Configure pipeline
    print(f"\n⚙️  Step 3: Setting up pipeline configuration...")
    try:
        from config import config
        
        # Optimize for BCICIV-2a dataset
        config.sampling_rate = 250
        config.K_selected = 6  # Select top 6 channels
        config.n_classes = 4   # 4 motor imagery classes
        config.max_epochs = 15  # Reasonable training time
        config.patience = 8
        config.cv_folds = 3    # 3-fold CV for faster results
        config.target_freq = (8, 30)  # Motor imagery frequency band
        
        print(f"✅ Configuration set!")
        print(f"   🎯 Channels to select: {config.K_selected}")
        print(f"   📊 Cross-validation folds: {config.cv_folds}")
        print(f"   🌊 Frequency band: {config.target_freq} Hz")
        
    except Exception as e:
        print(f"❌ Error configuring pipeline: {e}")
        return None
    
    # Step 4: Run the complete pipeline
    print(f"\n🚀 Step 4: Running EEG Channel Selection & Classification Pipeline...")
    print("   This will take a few minutes...")
    
    try:
        from pipeline import run_full_pipeline
        
        # Run the complete two-stage pipeline
        results = run_full_pipeline(
            raw_eeg_data=processed_data,
            labels=processed_labels,
            N_channels=len(channel_names),
            K_selected=config.K_selected,
            ch_names=channel_names,
            sampling_rate=config.sampling_rate,
            cv_folds=config.cv_folds,
            save_results=True,
            results_prefix=f"classification_{subject_id}"
        )
        
        print(f"✅ Pipeline completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running pipeline: {e}")
        return None
    
    # Step 5: Display Classification Results
    print(f"\n📊 Step 5: Classification Results for {subject_id}")
    print("=" * 60)
    
    # Extract results
    stage1_results = results['stage1_results']
    final_results = results['final_results']
    
    # Selected channels
    selected_channels = stage1_results['selected_names']
    selected_accuracies = stage1_results['selection_results']['selected_accuracies']
    
    print(f"\n🎯 SELECTED CHANNELS (Top {len(selected_channels)}):")
    for i, (channel, acc) in enumerate(zip(selected_channels, selected_accuracies)):
        print(f"   {i+1}. {channel}: {acc:.4f} accuracy")
    
    # Classification performance
    mean_accuracy = final_results['mean_accuracy']
    std_accuracy = final_results['std_accuracy']
    best_fold = final_results['best_fold_accuracy']
    worst_fold = final_results['worst_fold_accuracy']
    
    print(f"\n🏆 FINAL CLASSIFICATION RESULTS:")
    print(f"   📈 Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"   🥇 Best Fold:     {best_fold:.4f}")
    print(f"   📉 Worst Fold:    {worst_fold:.4f}")
    print(f"   ⏱️  Runtime:       {final_results['execution_time']:.1f} seconds")
    
    # Performance interpretation
    print(f"\n💡 PERFORMANCE INTERPRETATION:")
    if mean_accuracy >= 0.80:
        print(f"   🌟 EXCELLENT! Very high classification accuracy")
    elif mean_accuracy >= 0.70:
        print(f"   ✨ GOOD! Above-average performance for motor imagery")
    elif mean_accuracy >= 0.60:
        print(f"   👍 DECENT! Reasonable performance, room for improvement")
    else:
        print(f"   📚 NEEDS WORK! Below chance level, consider different subjects/parameters")
    
    # Channel analysis
    motor_channels = ['C3', 'C4', 'Cz', 'CP3', 'CP4']
    selected_motor = [ch for ch in selected_channels if ch in motor_channels]
    
    print(f"\n🧠 CHANNEL ANALYSIS:")
    print(f"   🎯 Motor cortex channels selected: {len(selected_motor)}/{len(motor_channels)}")
    if selected_motor:
        print(f"   📍 Motor channels: {', '.join(selected_motor)}")
    
    # Save summary
    summary = {
        'subject_id': subject_id,
        'dataset': 'BCI Competition IV Dataset 2a',
        'mean_accuracy': float(mean_accuracy),
        'std_accuracy': float(std_accuracy),
        'selected_channels': selected_channels,
        'selected_accuracies': [float(x) for x in selected_accuracies],
        'motor_channels_selected': selected_motor,
        'class_names': ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
    }
    
    # Save results
    import json
    output_file = f"classification_results_{subject_id}.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    return results


def classify_and_compare_subjects(data_path: str, subjects: list = None):
    """
    Classify multiple subjects and compare results.
    
    Args:
        data_path: Path to BCICIV-2a dataset
        subjects: List of subject IDs to compare
    """
    
    if subjects is None:
        subjects = ['A01', 'A02', 'A03']  # Default subjects
    
    print(f"\n🔬 MULTI-SUBJECT COMPARISON")
    print("=" * 60)
    print(f"Subjects to process: {', '.join(subjects)}")
    
    all_results = {}
    
    for subject in subjects:
        print(f"\n{'='*20} {subject} {'='*20}")
        
        try:
            results = classify_bciciv_2a_subject(data_path, subject)
            if results:
                all_results[subject] = {
                    'accuracy': results['final_results']['mean_accuracy'],
                    'std': results['final_results']['std_accuracy'],
                    'channels': results['stage1_results']['selected_names']
                }
        except Exception as e:
            print(f"❌ Failed to process {subject}: {e}")
            continue
    
    # Compare results
    if len(all_results) > 1:
        print(f"\n📊 COMPARISON SUMMARY")
        print("=" * 40)
        
        accuracies = [(subj, data['accuracy']) for subj, data in all_results.items()]
        accuracies.sort(key=lambda x: x[1], reverse=True)
        
        print(f"📈 Subject Rankings:")
        for rank, (subject, acc) in enumerate(accuracies, 1):
            print(f"   {rank}. {subject}: {acc:.4f}")
        
        # Best subject analysis
        best_subject = accuracies[0][0]
        best_channels = all_results[best_subject]['channels']
        
        print(f"\n🏆 Best Subject: {best_subject}")
        print(f"   Selected channels: {', '.join(best_channels)}")
        
        # Save comparison
        comparison_file = "multi_subject_comparison.json"
        with open(comparison_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n💾 Comparison saved to: {comparison_file}")


def main():
    parser = argparse.ArgumentParser(description="BCICIV-2a EEG Classification")
    parser.add_argument("--data_path", type=str, required=True, 
                       help="Path to BCICIV-2a dataset folder")
    parser.add_argument("--subject", type=str, default="A01",
                       help="Subject ID to classify (A01-A09)")
    parser.add_argument("--compare", nargs='+', 
                       help="Compare multiple subjects (e.g., --compare A01 A02 A03)")
    
    args = parser.parse_args()
    
    # Verify dataset path
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"❌ Error: Dataset path does not exist: {data_path}")
        return
    
    # Check for .mat files
    mat_files = list(data_path.glob("*.mat"))
    if not mat_files:
        print(f"❌ Error: No .mat files found in {data_path}")
        print("💡 Make sure your folder contains files like A01T.mat, A01E.mat, etc.")
        return
    
    print(f"✅ Found {len(mat_files)} .mat files in dataset")
    
    try:
        if args.compare:
            # Multi-subject comparison
            classify_and_compare_subjects(str(data_path), args.compare)
        else:
            # Single subject classification
            results = classify_bciciv_2a_subject(str(data_path), args.subject)
            
            if results:
                print(f"\n🎉 SUCCESS! Classification completed for {args.subject}")
                print(f"📁 Check the generated files for detailed results")
            else:
                print(f"❌ Classification failed for {args.subject}")
    
    except KeyboardInterrupt:
        print(f"\n⛔ Classification interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("💡 Try running with a single subject first to debug")


if __name__ == "__main__":
    main()