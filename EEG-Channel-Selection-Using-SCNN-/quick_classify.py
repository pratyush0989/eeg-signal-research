"""
QUICK BCICIV-2a Classification

The simplest way to get classification results from your BCICIV-2a dataset.
Just run this script and get results in minutes!

Usage:
    python quick_classify.py /path/to/bciciv_2a A01
"""

import sys
import numpy as np
from pathlib import Path

def quick_classify(data_path, subject_id="A01"):
    """Get classification results quickly with minimal setup"""
    
    print(f"🚀 Quick EEG Classification for {subject_id}")
    print("-" * 50)
    
    try:
        # Step 1: Load data
        print(f"📂 Loading {subject_id}...")
        from bciciv_2a_loader import load_bciciv_2a_subject, preprocess_bciciv_2a_for_pipeline
        
        eeg_data, labels, channels = load_bciciv_2a_subject(
            data_path, subject_id, session='T'
        )
        
        processed_data, processed_labels, channels = preprocess_bciciv_2a_for_pipeline(
            eeg_data, labels, channels
        )
        
        print(f"✅ Loaded: {processed_data.shape} | Classes: {len(np.unique(processed_labels))}")
        
        # Step 2: Quick config
        print("⚙️ Setting up pipeline...")
        from config import config
        config.K_selected = 4      # Select top 4 channels for speed
        config.max_epochs = 8      # Fast training
        config.patience = 4
        config.cv_folds = 3        # 3-fold CV
        
        # Step 3: Run pipeline
        print("🧠 Running classification...")
        from pipeline import run_full_pipeline
        
        results = run_full_pipeline(
            raw_eeg_data=processed_data,
            labels=processed_labels,
            K_selected=4,
            ch_names=channels,
            cv_folds=3,
            save_results=False  # Don't save for quick run
        )
        
        # Step 4: Show results
        accuracy = results['final_results']['mean_accuracy']
        selected = results['stage1_results']['selected_names']
        
        print(f"\n🎯 RESULTS:")
        print(f"   Accuracy: {accuracy:.1%}")
        print(f"   Selected Channels: {', '.join(selected)}")
        
        if accuracy > 0.7:
            print(f"   🌟 Great performance!")
        elif accuracy > 0.6:
            print(f"   👍 Good performance!")
        else:
            print(f"   📈 Room for improvement")
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quick_classify.py /path/to/bciciv_2a [subject]")
        print("Example: python quick_classify.py C:/datasets/bciciv_2a A01")
        sys.exit(1)
    
    data_path = sys.argv[1]
    subject = sys.argv[2] if len(sys.argv) > 2 else "A01"
    
    quick_classify(data_path, subject)