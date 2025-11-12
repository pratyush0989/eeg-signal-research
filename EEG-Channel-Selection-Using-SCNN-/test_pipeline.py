"""
Test suite for the EEG Channel Selection and Classification pipeline
"""
import unittest
import numpy as np
import torch
import tempfile
import os
from pathlib import Path

# Import modules to test
from config import config, EEGConfig
from preprocessing import (
    preprocess_data_for_selection,
    generate_single_channel_inputs,
    prepare_fusion_input
)
from scnn_model import build_shallow_cnn_selector, ShallowCNN, SCNNTrainer
from fusion_cnn import build_fusion_cnn_classifier, FusionCNN
from channel_selection import select_best_channels
from pipeline import run_full_pipeline


class TestConfiguration(unittest.TestCase):
    """Test configuration management"""
    
    def test_default_config(self):
        """Test default configuration values"""
        self.assertEqual(config.sampling_rate, 250)
        self.assertEqual(config.K_selected, 6)
        self.assertEqual(config.n_classes, 4)
        self.assertEqual(len(config.fusion_bands), 4)
    
    def test_custom_config(self):
        """Test custom configuration creation"""
        custom_config = EEGConfig(
            sampling_rate=500,
            K_selected=8,
            n_classes=2
        )
        self.assertEqual(custom_config.sampling_rate, 500)
        self.assertEqual(custom_config.K_selected, 8)
        self.assertEqual(custom_config.n_classes, 2)


class TestPreprocessing(unittest.TestCase):
    """Test preprocessing functions"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.n_trials = 50
        self.n_channels = 8
        self.time_samples = 501
        self.n_classes = 4
        
        # Generate test data
        self.eeg_data = np.random.randn(self.n_trials, self.n_channels, self.time_samples)
        self.labels = np.random.randint(0, self.n_classes, self.n_trials)
        self.ch_names = [f'Ch_{i}' for i in range(self.n_channels)]
    
    def test_preprocess_data_for_selection(self):
        """Test data preprocessing function"""
        processed_data, processed_ch_names = preprocess_data_for_selection(
            self.eeg_data, sampling_rate=250, ch_names=self.ch_names
        )
        
        # Check output shapes
        self.assertEqual(processed_data.shape[0], self.n_trials)
        self.assertEqual(processed_data.shape[1], self.n_channels)
        self.assertEqual(len(processed_ch_names), self.n_channels)
        
        # Check that data is standardized (approximately zero mean)
        for ch in range(self.n_channels):
            channel_mean = np.mean(processed_data[:, ch, :])
            self.assertAlmostEqual(channel_mean, 0.0, places=1)
    
    def test_generate_single_channel_inputs(self):
        """Test single channel input generation"""
        processed_data, _ = preprocess_data_for_selection(self.eeg_data)
        channel_data_list = generate_single_channel_inputs(processed_data, self.labels)
        
        # Check number of channel datasets
        self.assertEqual(len(channel_data_list), self.n_channels)
        
        # Check each channel dataset
        for i, (X_channel, y_labels) in enumerate(channel_data_list):
            self.assertEqual(X_channel.shape[0], self.n_trials)
            self.assertEqual(X_channel.shape[1], processed_data.shape[2])
            self.assertEqual(X_channel.shape[2], 1)
            self.assertEqual(len(y_labels), self.n_trials)
    
    def test_prepare_fusion_input(self):
        """Test fusion input preparation"""
        fusion_inputs = prepare_fusion_input(
            self.eeg_data, sampling_rate=250, ch_names=self.ch_names
        )
        
        # Check number of frequency bands
        self.assertEqual(len(fusion_inputs), 4)
        
        # Check shapes
        for fusion_input in fusion_inputs:
            self.assertEqual(fusion_input.shape[0], self.n_trials)
            self.assertEqual(fusion_input.shape[1], self.n_channels)


class TestSCNNModel(unittest.TestCase):
    """Test Shallow CNN model"""
    
    def setUp(self):
        """Set up test parameters"""
        self.input_time_samples = 501
        self.n_classes = 4
        self.batch_size = 10
        
    def test_build_shallow_cnn_selector(self):
        """Test SCNN model building"""
        model = build_shallow_cnn_selector(
            input_time_samples=self.input_time_samples,
            n_classes=self.n_classes
        )
        
        self.assertIsInstance(model, ShallowCNN)
        self.assertEqual(model.n_classes, self.n_classes)
        self.assertEqual(model.input_time_samples, self.input_time_samples)
    
    def test_scnn_forward_pass(self):
        """Test SCNN forward pass"""
        model = build_shallow_cnn_selector(
            input_time_samples=self.input_time_samples,
            n_classes=self.n_classes
        )
        
        # Create test input
        x = torch.randn(self.batch_size, self.input_time_samples, 1)
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        self.assertEqual(output.shape[0], self.batch_size)
        self.assertEqual(output.shape[1], self.n_classes)
        
        # Check that output is probability distribution
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output <= 1))
        self.assertTrue(torch.allclose(torch.sum(output, dim=1), torch.ones(self.batch_size)))
    
    def test_scnn_trainer_initialization(self):
        """Test SCNN trainer initialization"""
        model = build_shallow_cnn_selector(
            input_time_samples=self.input_time_samples,
            n_classes=self.n_classes
        )
        
        trainer = SCNNTrainer(model)
        
        self.assertIsNotNone(trainer.optimizer)
        self.assertIsNotNone(trainer.criterion)
        self.assertEqual(len(trainer.train_losses), 0)


class TestFusionCNN(unittest.TestCase):
    """Test Fusion CNN model"""
    
    def setUp(self):
        """Set up test parameters"""
        self.K_channels = 6
        self.input_time_samples = 501
        self.n_classes = 4
        self.batch_size = 8
    
    def test_build_fusion_cnn_classifier(self):
        """Test Fusion CNN building"""
        model = build_fusion_cnn_classifier(
            K_channels=self.K_channels,
            input_time_samples=self.input_time_samples,
            n_classes=self.n_classes
        )
        
        self.assertIsInstance(model, FusionCNN)
        self.assertEqual(model.K_channels, self.K_channels)
        self.assertEqual(model.n_classes, self.n_classes)
    
    def test_fusion_cnn_forward_pass(self):
        """Test Fusion CNN forward pass"""
        model = build_fusion_cnn_classifier(
            K_channels=self.K_channels,
            input_time_samples=self.input_time_samples,
            n_classes=self.n_classes
        )
        
        # Create test inputs (4 frequency bands)
        inputs = [
            torch.randn(self.batch_size, self.K_channels, self.input_time_samples)
            for _ in range(4)
        ]
        
        # Forward pass
        output = model(inputs)
        
        # Check output shape
        self.assertEqual(output.shape[0], self.batch_size)
        self.assertEqual(output.shape[1], self.n_classes)
        
        # Check that output is probability distribution
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output <= 1))
        self.assertTrue(torch.allclose(torch.sum(output, dim=1), torch.ones(self.batch_size)))


class TestChannelSelection(unittest.TestCase):
    """Test channel selection functionality"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        torch.manual_seed(42)
        
        self.n_trials = 30  # Small for fast testing
        self.n_channels = 6
        self.time_samples = 251
        self.n_classes = 2
        
        # Generate test data
        eeg_data = np.random.randn(self.n_trials, self.n_channels, self.time_samples)
        labels = np.random.randint(0, self.n_classes, self.n_trials)
        
        # Preprocess data
        processed_data, self.ch_names = preprocess_data_for_selection(eeg_data)
        self.channel_data_list = generate_single_channel_inputs(processed_data, labels)
    
    def test_select_best_channels(self):
        """Test channel selection function"""
        K = 3
        
        selected_indices, selected_names, results = select_best_channels(
            channel_data_list=self.channel_data_list,
            K=K,
            ch_names=self.ch_names,
            max_epochs=3,  # Very limited for testing
            patience=2
        )
        
        # Check outputs
        self.assertEqual(len(selected_indices), K)
        self.assertEqual(len(selected_names), K)
        self.assertIn('selected_accuracies', results)
        self.assertIn('selection_summary', results)
        
        # Check that indices are valid
        self.assertTrue(all(0 <= idx < self.n_channels for idx in selected_indices))


class TestPipeline(unittest.TestCase):
    """Test complete pipeline"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        
        # Very small dataset for fast testing
        self.n_trials = 40
        self.n_channels = 8
        self.time_samples = 251
        self.n_classes = 2
        
        self.eeg_data = np.random.randn(self.n_trials, self.n_channels, self.time_samples)
        self.labels = np.random.randint(0, self.n_classes, self.n_trials)
        self.ch_names = [f'Ch_{i}' for i in range(self.n_channels)]
    
    def test_run_full_pipeline(self):
        """Test complete pipeline execution"""
        # Use very limited parameters for fast testing
        results = run_full_pipeline(
            raw_eeg_data=self.eeg_data,
            labels=self.labels,
            N_channels=self.n_channels,
            K_selected=3,  # Select only 3 channels
            ch_names=self.ch_names,
            sampling_rate=250,
            cv_folds=2,  # Only 2 folds for speed
            save_results=False  # Don't save during tests
        )
        
        # Check that results contain expected keys
        expected_keys = ['pipeline_config', 'stage1_results', 'stage2_results', 'final_results']
        for key in expected_keys:
            self.assertIn(key, results)
        
        # Check stage 1 results
        stage1 = results['stage1_results']
        self.assertIn('selected_indices', stage1)
        self.assertIn('selected_names', stage1)
        self.assertEqual(len(stage1['selected_indices']), 3)
        
        # Check stage 2 results
        stage2 = results['stage2_results']
        self.assertIn('cv_results', stage2)
        self.assertEqual(len(stage2['cv_results']), 2)  # 2 folds
        
        # Check final results
        final = results['final_results']
        self.assertIn('mean_accuracy', final)
        self.assertIn('std_accuracy', final)
        self.assertTrue(0 <= final['mean_accuracy'] <= 1)


class TestUtilities(unittest.TestCase):
    """Test utility functions"""
    
    def test_imports(self):
        """Test that all utility functions can be imported"""
        try:
            from utils import (
                plot_channel_selection_results,
                plot_training_history,
                evaluate_model_predictions,
                create_summary_statistics_table
            )
        except ImportError as e:
            # Skip if matplotlib/seaborn not available
            if "matplotlib" in str(e) or "seaborn" in str(e):
                self.skipTest("Matplotlib/seaborn not available for testing")
            else:
                raise


def run_tests():
    """Run all tests"""
    # Configure for faster testing
    config.max_epochs = 3
    config.patience = 2
    config.batch_size = 8
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestConfiguration,
        TestPreprocessing, 
        TestSCNNModel,
        TestFusionCNN,
        TestChannelSelection,
        TestPipeline,
        TestUtilities
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    """Run tests when script is executed directly"""
    print("Running EEG Channel Selection Pipeline Tests")
    print("=" * 50)
    
    success = run_tests()
    
    if success:
        print("\nAll tests passed successfully!")
    else:
        print("\nSome tests failed. Check the output above for details.")
    
    print("=" * 50)