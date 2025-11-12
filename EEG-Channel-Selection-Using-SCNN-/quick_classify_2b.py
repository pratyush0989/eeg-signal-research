"""
Quick Binary Classification for BCICIV-2b Dataset

Simple way to get binary classification results (left hand vs right hand)
from your BCICIV-2b dataset with only 3 EEG channels.

Usage:
    python quick_classify_2b.py /path/to/bciciv_2b B01
"""

import sys
import numpy as np
from pathlib import Path

def quick_classify_2b(data_path, subject_id="B01"):
    """Get binary classification results quickly from BCICIV-2b"""
    
    print(f"🧠 BCICIV-2b Binary Classification: {subject_id}")
    print("-" * 50)
    print("📊 Task: Left Hand vs Right Hand")
    print("📡 Channels: C3, Cz, C4 (motor cortex)")
    print()
    
    try:
        # Step 1: Load data
        print(f"📂 Loading {subject_id} training sessions...")
        from bciciv_2b_loader import load_bciciv_2b_subject, preprocess_bciciv_2b_for_pipeline
        
        eeg_data, labels, channels = load_bciciv_2b_subject(
            data_path, subject_id, sessions=['01T', '02T', '03T']
        )
        
        processed_data, processed_labels, channels = preprocess_bciciv_2b_for_pipeline(
            eeg_data, labels, channels
        )
        
        left_trials = sum(processed_labels == 0)
        right_trials = sum(processed_labels == 1)
        print(f"✅ Loaded: {processed_data.shape}")
        print(f"   Left Hand: {left_trials} trials")
        print(f"   Right Hand: {right_trials} trials")
        
        # Step 2: Quick config for binary classification
        print("⚙️ Setting up binary classification pipeline...")
        from config import config
        config.K_selected = 2      # Select best 2 out of 3 channels
        config.max_epochs = 5      # Slightly more training for binary task
        config.patience = 3
        config.cv_folds = 3
        config.n_classes = 2       # Binary classification
        
        # Step 3: Run pipeline
        print("🧠 Running binary classification...")
        from pipeline import run_full_pipeline
        
        results = run_full_pipeline(
            raw_eeg_data=processed_data,
            labels=processed_labels,
            K_selected=2,  # Only 2 channels from 3 available
            ch_names=channels,
            cv_folds=3,
            save_results=False
        )
        
        # Step 4: Show results
        accuracy = results['final_results']['mean_accuracy']
        selected = results['stage1_results']['selected_names']
        
        print(f"\n🎯 BCICIV-2b RESULTS:")
        print(f"   Binary Accuracy: {accuracy:.1%}")
        print(f"   Selected Channels: {', '.join(selected)}")
        print(f"   Available Channels: {', '.join(channels)}")
        
        # Binary classification interpretation
        if accuracy > 0.75:
            print(f"   🌟 Excellent binary classification!")
        elif accuracy > 0.65:
            print(f"   👍 Good binary classification!")
        elif accuracy > 0.55:
            print(f"   📈 Above chance level (50%)")
        else:
            print(f"   ⚠️ Near chance level - may need more data")
        
        # Channel interpretation for motor imagery
        motor_channels = ['C3', 'C4']  # Left/right motor cortex
        selected_motor = [ch for ch in selected if ch in motor_channels]
        if selected_motor:
            print(f"   🎪 Motor cortex channels selected: {selected_motor}")
        if 'Cz' in selected:
            print(f"   🎯 Central channel (Cz) selected - good for motor imagery")
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quick_classify_2b.py /path/to/bciciv_2b [subject]")
        print("Example: python quick_classify_2b.py BCICIV_2b_gdf B01")
        sys.exit(1)
    
    data_path = sys.argv[1]
    subject = sys.argv[2] if len(sys.argv) > 2 else "B01"
    
    quick_classify_2b(data_path, subject)