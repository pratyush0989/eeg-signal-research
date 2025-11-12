"""
Quick start script for BCI Competition IV Dataset 2b with EEG Pipeline

This script helps you quickly test BCICIV-2b dataset (binary classification)
with the EEG channel selection pipeline.

Key differences from Dataset 2a:
- Only 3 EEG channels (C3, Cz, C4) 
- Binary classification: left hand vs right hand
- Multiple sessions per subject (3 training + 2 evaluation)

Usage:
    python bciciv_2b_quickstart.py --data_path /path/to/bciciv_2b_gdf --subject B01
"""

import sys
import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_dataset_structure_2b(data_path: str):
    """
    Check if the BCICIV-2b dataset structure is correct.
    
    Expected files per subject:
    - B0101T.gdf, B0102T.gdf, B0103T.gdf (Training sessions)
    - B0104E.gdf, B0105E.gdf (Evaluation sessions)
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        logger.error(f"Dataset path does not exist: {data_path}")
        return False
    
    logger.info(f"Checking BCICIV-2b dataset structure in: {data_path}")
    
    # Check for expected files
    expected_subjects = [f'B{i:02d}' for i in range(1, 10)]  # B01-B09
    found_files = []
    missing_files = []
    
    for subject in expected_subjects:
        for session in ['01T', '02T', '03T', '04E', '05E']:
            filename = f"{subject}{session}.gdf"
            filepath = data_path / filename
            
            if filepath.exists():
                found_files.append(str(filepath))
            else:
                missing_files.append(str(filepath))
    
    logger.info(f"Found {len(found_files)} files:")
    for file in found_files[:5]:  # Show first 5
        logger.info(f"  ✓ {Path(file).name}")
    if len(found_files) > 5:
        logger.info(f"  ... and {len(found_files) - 5} more")
    
    if missing_files:
        logger.warning(f"Missing {len(missing_files)} files:")
        for file in missing_files[:3]:  # Show first 3
            logger.warning(f"  ✗ {Path(file).name}")
        if len(missing_files) > 3:
            logger.warning(f"  ... and {len(missing_files) - 3} more")
    
    return len(found_files) > 0


def test_single_subject_loading_2b(data_path: str, subject_id: str = "B01"):
    """Test loading a single subject to verify BCICIV-2b data format."""
    
    try:
        from bciciv_2b_loader import load_bciciv_2b_subject
        
        logger.info(f"Testing BCICIV-2b data loading for subject {subject_id}...")
        
        # Load training sessions
        eeg_data, labels, channel_names = load_bciciv_2b_subject(
            data_path=data_path,
            subject_id=subject_id,
            sessions=['01T', '02T', '03T']  # Training sessions
        )
        
        logger.info(f"✓ Successfully loaded {subject_id} training data:")
        logger.info(f"  Data shape: {eeg_data.shape}")
        logger.info(f"  Labels shape: {labels.shape}")
        logger.info(f"  Number of channels: {len(channel_names)}")
        logger.info(f"  Channel names: {channel_names}")
        logger.info(f"  Classes: {len(set(labels))} (binary classification)")
        logger.info(f"  Class distribution: Left={sum(labels==0)}, Right={sum(labels==1)}")
        
        # Test evaluation sessions
        try:
            eval_data, eval_labels, _ = load_bciciv_2b_subject(
                data_path=data_path,
                subject_id=subject_id,
                sessions=['04E', '05E']  # Evaluation sessions
            )
            logger.info(f"✓ Evaluation sessions: {eval_data.shape}")
        except Exception as e:
            logger.warning(f"Evaluation sessions not available: {e}")
        
        return True
        
    except ImportError as e:
        logger.error(f"Cannot import BCICIV-2b loader: {e}")
        return False
    except Exception as e:
        logger.error(f"Error loading {subject_id}: {e}")
        return False


def run_quick_pipeline_test_2b(data_path: str, subject_id: str = "B01"):
    """Run a quick pipeline test with BCICIV-2b data."""
    
    try:
        from bciciv_2b_loader import load_bciciv_2b_subject, preprocess_bciciv_2b_for_pipeline
        from pipeline import run_full_pipeline
        from config import config
        
        logger.info(f"Running quick pipeline test with BCICIV-2b {subject_id}...")
        
        # Load and preprocess data
        eeg_data, labels, channel_names = load_bciciv_2b_subject(
            data_path=data_path,
            subject_id=subject_id,
            sessions=['01T', '02T', '03T']  # Training sessions
        )
        
        processed_data, processed_labels, channel_names = preprocess_bciciv_2b_for_pipeline(
            eeg_data, labels, channel_names
        )
        
        # Set fast parameters for testing - Binary classification
        config.max_epochs = 3
        config.patience = 2
        config.cv_folds = 2
        config.K_selected = 2  # Select 2 out of 3 channels for binary task
        config.n_classes = 2   # Binary classification
        
        logger.info("Running pipeline with BCICIV-2b parameters (binary classification)...")
        
        # Run pipeline
        results = run_full_pipeline(
            raw_eeg_data=processed_data,
            labels=processed_labels,
            K_selected=config.K_selected,
            ch_names=channel_names,
            cv_folds=config.cv_folds,
            save_results=False  # Don't save for quick test
        )
        
        logger.info(f"✓ BCICIV-2b pipeline completed successfully!")
        logger.info(f"Selected channels: {results['stage1_results']['selected_names']}")
        logger.info(f"Final accuracy: {results['final_results']['mean_accuracy']:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"BCICIV-2b pipeline test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="BCICIV-2b Quick Start")
    parser.add_argument("--data_path", type=str, required=True, 
                       help="Path to BCICIV-2b dataset directory")
    parser.add_argument("--subject", type=str, default="B01",
                       help="Subject to test (default: B01)")
    parser.add_argument("--skip_pipeline", action="store_true",
                       help="Skip pipeline test, only check data loading")
    
    args = parser.parse_args()
    
    print("🧠 BCI Competition IV Dataset 2b - Quick Start")
    print("=" * 50)
    print("📊 Binary Classification: Left Hand vs Right Hand")
    print("📡 3 EEG Channels: C3, Cz, C4")
    print("📁 5 Sessions per Subject: 3 Training + 2 Evaluation")
    print()
    
    # Step 1: Check dataset structure
    print("1️⃣ Checking BCICIV-2b dataset structure...")
    if not check_dataset_structure_2b(args.data_path):
        print("❌ Dataset structure check failed")
        return False
    
    # Step 2: Test data loading
    print(f"\n2️⃣ Testing BCICIV-2b data loading for {args.subject}...")
    if not test_single_subject_loading_2b(args.data_path, args.subject):
        print("❌ Data loading test failed")
        return False
    
    # Step 3: Test pipeline (optional)
    if not args.skip_pipeline:
        print(f"\n3️⃣ Testing BCICIV-2b pipeline with {args.subject}...")
        if not run_quick_pipeline_test_2b(args.data_path, args.subject):
            print("❌ Pipeline test failed")
            return False
        else:
            print("✅ BCICIV-2b pipeline test passed!")
    
    print("\n🎉 BCICIV-2b quick start completed successfully!")
    print("\n📋 Next steps:")
    print("1. Compare with Dataset 2a:")
    print(f"   python bciciv_2a_quickstart.py --data_path gdf --subject A01")
    print("\n2. Run full BCICIV-2b analysis:")
    print(f"   python quick_classify_2b.py {args.data_path} {args.subject}")
    print("\n3. Test all BCICIV-2b subjects:")
    print(f"   python bciciv_2b_loader.py {args.data_path}")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)