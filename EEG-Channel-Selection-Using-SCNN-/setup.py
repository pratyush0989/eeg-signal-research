#!/usr/bin/env python
"""
Setup script for EEG Channel Selection and Classification package
"""
from pathlib import Path

def setup_environment():
    """
    Set up the environment for the EEG pipeline.
    """
    print("Setting up EEG Channel Selection and Classification Environment")
    print("=" * 60)
    
    # Check Python version
    import sys
    print(f"Python version: {sys.version}")
    
    # Check required packages
    required_packages = [
        'torch', 'numpy', 'scipy', 'sklearn', 
        'matplotlib', 'seaborn', 'pandas', 'tqdm'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    # Test basic functionality
    print("\nTesting basic functionality...")
    try:
        from config import config
        print(f"✓ Configuration loaded (K={config.K_selected}, SR={config.sampling_rate}Hz)")
        
        from preprocessing import preprocess_data_for_selection
        print("✓ Preprocessing module")
        
        from scnn_model import build_shallow_cnn_selector
        print("✓ SCNN model module")
        
        from fusion_cnn import build_fusion_cnn_classifier
        print("✓ Fusion CNN module")
        
        from pipeline import run_full_pipeline
        print("✓ Main pipeline")
        
        print("\n✓ All modules loaded successfully!")
        
    except Exception as e:
        print(f"\n✗ Error loading modules: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nQuick start options:")
    print("1. Run demo:           python demo.py")
    print("2. Run tests:          python test_pipeline.py")
    print("3. Use in your code:   from pipeline import run_full_pipeline")
    print("\nSee README.md for detailed usage instructions.")
    
    return True


def run_quick_test():
    """
    Run a quick test to ensure everything works.
    """
    print("\nRunning quick functionality test...")
    
    try:
        from demo import generate_synthetic_eeg_data
        
        # Generate minimal test data
        eeg_data, labels, ch_names = generate_synthetic_eeg_data(
            n_trials=10, n_channels=4, time_samples=125, n_classes=2
        )
        
        # Test preprocessing
        from preprocessing import preprocess_data_for_selection
        processed_data, _ = preprocess_data_for_selection(eeg_data)
        
        print(f"✓ Test data: {eeg_data.shape} -> {processed_data.shape}")
        print("✓ Quick test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Quick test failed: {e}")
        return False


if __name__ == "__main__":
    """
    Main setup execution
    """
    success = setup_environment()
    
    if success:
        test_success = run_quick_test()
        if not test_success:
            print("\nWarning: Quick test failed. Check your installation.")
    else:
        print("\nSetup failed. Please install missing dependencies.")
        print("Run: pip install -r requirements.txt")
    
    print(f"\nSetup {'SUCCESS' if success else 'FAILED'}")