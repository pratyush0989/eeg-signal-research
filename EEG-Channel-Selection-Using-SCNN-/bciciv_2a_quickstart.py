"""
Quick start script for BCI Competition IV Dataset 2a with EEG Pipeline

This script helps you quickly test if your dataset is correctly formatted
and runs a fast demo with your actual data.

Usage:
    python bciciv_2a_quickstart.py --data_path /path/to/your/bciciv_2a_dataset
"""

import sys
import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_dataset_structure(data_path: str):
    """
    Check if the BCICIV-2a dataset structure is correct.
    
    Expected files:
    - A01T.gdf, A01E.gdf (Subject 1 training and evaluation)
    - A02T.gdf, A02E.gdf (Subject 2 training and evaluation)
    - ... up to A09T.gdf, A09E.gdf
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        logger.error(f"Dataset path does not exist: {data_path}")
        return False
    
    logger.info(f"Checking dataset structure in: {data_path}")
    
    # Check for expected files
    expected_subjects = [f'A{i:02d}' for i in range(1, 10)]
    found_files = []
    missing_files = []
    
    for subject in expected_subjects:
        train_file = data_path / f"{subject}T.gdf"
        eval_file = data_path / f"{subject}E.gdf"
        
        if train_file.exists():
            found_files.append(str(train_file))
        else:
            missing_files.append(str(train_file))
            
        if eval_file.exists():
            found_files.append(str(eval_file))
        else:
            missing_files.append(str(eval_file))
    
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


def test_single_subject_loading(data_path: str, subject_id: str = "A01"):
    """Test loading a single subject to verify data format."""
    
    try:
        from bciciv_2a_loader import load_bciciv_2a_subject
        
        logger.info(f"Testing data loading for subject {subject_id}...")
        
        # Load training data
        eeg_data, labels, channel_names = load_bciciv_2a_subject(
            data_path=data_path,
            subject_id=subject_id,
            session='T',
            use_eeg_only=True
        )
        
        logger.info(f"✓ Successfully loaded {subject_id} training data:")
        logger.info(f"  Data shape: {eeg_data.shape}")
        logger.info(f"  Labels shape: {labels.shape}")
        logger.info(f"  Number of channels: {len(channel_names)}")
        logger.info(f"  Channel names: {channel_names[:5]}...")  # First 5 channels
        logger.info(f"  Classes: {len(set(labels))}")
        logger.info(f"  Class distribution: {[int(x) for x in labels[:10]]}...")  # First 10 labels
        
        return True
        
    except ImportError as e:
        logger.error(f"Cannot import BCICIV-2a loader: {e}")
        logger.error("Make sure you're in the correct directory with bciciv_2a_loader.py")
        return False
    except Exception as e:
        logger.error(f"Error loading {subject_id}: {e}")
        logger.error("Please check your dataset format and file paths")
        return False


def run_quick_pipeline_test(data_path: str, subject_id: str = "A01"):
    """Run a quick pipeline test with minimal parameters."""
    
    try:
        from bciciv_2a_loader import load_bciciv_2a_subject, preprocess_bciciv_2a_for_pipeline
        from pipeline import run_full_pipeline
        from config import config
        
        logger.info(f"Running quick pipeline test with {subject_id}...")
        
        # Load and preprocess data
        eeg_data, labels, channel_names = load_bciciv_2a_subject(
            data_path=data_path,
            subject_id=subject_id,
            session='T'
        )
        
        processed_data, processed_labels, channel_names = preprocess_bciciv_2a_for_pipeline(
            eeg_data, labels, channel_names
        )
        
        # Set fast parameters for testing
        config.max_epochs = 3
        config.patience = 2
        config.cv_folds = 2
        config.K_selected = 4  # Select fewer channels for speed
        
        logger.info("Running pipeline with fast parameters...")
        
        # Run pipeline
        results = run_full_pipeline(
            raw_eeg_data=processed_data,
            labels=processed_labels,
            N_channels=len(channel_names),
            K_selected=config.K_selected,
            ch_names=channel_names,
            cv_folds=config.cv_folds,
            save_results=False  # Don't save for quick test
        )
        
        logger.info(f"✓ Pipeline completed successfully!")
        logger.info(f"Selected channels: {results['stage1_results']['selected_names']}")
        logger.info(f"Final accuracy: {results['final_results']['mean_accuracy']:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="BCICIV-2a Quick Start")
    parser.add_argument("--data_path", type=str, required=True, 
                       help="Path to BCICIV-2a dataset directory")
    parser.add_argument("--subject", type=str, default="A01",
                       help="Subject to test (default: A01)")
    parser.add_argument("--skip_pipeline", action="store_true",
                       help="Skip pipeline test, only check data loading")
    
    args = parser.parse_args()
    
    print("🧠 BCI Competition IV Dataset 2a - Quick Start")
    print("=" * 50)
    
    # Step 1: Check dataset structure
    print("\n1️⃣ Checking dataset structure...")
    if not check_dataset_structure(args.data_path):
        print("❌ Dataset structure check failed")
        return False
    
    # Step 2: Test data loading
    print(f"\n2️⃣ Testing data loading for {args.subject}...")
    if not test_single_subject_loading(args.data_path, args.subject):
        print("❌ Data loading test failed")
        return False
    
    # Step 3: Test pipeline (optional)
    if not args.skip_pipeline:
        print(f"\n3️⃣ Testing pipeline with {args.subject}...")
        if not run_quick_pipeline_test(args.data_path, args.subject):
            print("❌ Pipeline test failed")
            return False
        else:
            print("✅ Pipeline test passed!")
    
    print("\n🎉 Quick start completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run full analysis:")
    print(f"   python bciciv_2a_example.py --data_path {args.data_path} --subject {args.subject}")
    print("\n2. Compare multiple subjects:")
    print(f"   python bciciv_2a_example.py --data_path {args.data_path} --mode compare --subjects A01 A02 A03")
    print("\n3. Train/test evaluation:")
    print(f"   python bciciv_2a_example.py --data_path {args.data_path} --mode train_test --all_subjects")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)