# Two-Stage EEG Channel Selection and Classification

A comprehensive PyTorch implementation for EEG motor imagery classification using a two-stage approach:

1. **Channel Selection Stage**: Uses Shallow Convolutional Neural Networks (SCNNs) to evaluate individual channels and select the most informative ones
2. **Classification Stage**: Employs a Multi-Layer Fusion CNN that processes selected channels across multiple frequency bands

## Features

- **Highly Modular Design**: Each component is implemented as a separate, reusable module
- **Configurable Parameters**: Centralized configuration system for easy customization
- **Complete Pipeline**: End-to-end processing from raw EEG data to final classification
- **Cross-Validation**: Built-in k-fold cross-validation for robust performance evaluation
- **Visualization Tools**: Comprehensive plotting and analysis utilities
- **Logging Support**: Detailed logging throughout the pipeline for monitoring progress

## Quick Start

### Basic Usage

```python
import numpy as np
from pipeline import run_full_pipeline

# Load your EEG data (n_trials, n_channels, time_samples)
eeg_data = np.load('your_eeg_data.npy')
labels = np.load('your_labels.npy')

# Run the complete pipeline
results = run_full_pipeline(
    raw_eeg_data=eeg_data,
    labels=labels,
    K_selected=6,  # Number of channels to select
    sampling_rate=250,
    cv_folds=5
)

# Analyze results
from pipeline import analyze_pipeline_results
analyze_pipeline_results(results)
```

### Demo with Synthetic Data

```python
from demo import run_demo_with_synthetic_data

# Run a complete demo with generated synthetic EEG data
results = run_demo_with_synthetic_data()
```

### BCI Competition IV Dataset 2a Support

```python
from bciciv_2a_example import train_single_subject, setup_bciciv_2a_config

# Load and train on BCICIV-2a dataset
config = setup_bciciv_2a_config()
results = train_single_subject(
    data_path="/bciciv_2a",
    subject_id="A01",
    config=config
)
```

## Architecture Overview

### Stage 1: Channel Selection (SCNN)

- **Input**: Individual channel data (n_trials, time_samples, 1)
- **Architecture**:
  - Temporal convolution (50 filters, 30x1 kernel)
  - Two pointwise convolutions (30, 20 filters)
  - Batch normalization and ELU activations
  - Dropout and fully connected layers
- **Output**: Classification accuracy for each channel

### Stage 2: Multi-Layer Fusion CNN

- **Input**: Selected K channels across 4 frequency bands
  - Band A: 7-13 Hz (Alpha)
  - Band B: 13-31 Hz (Beta)
  - Band C: 7-31 Hz (Alpha + Beta)
  - Band D: 0-40 Hz (Full band)
- **Architecture**:
  - 4 parallel CNN branches (one per frequency band)
  - Spatial convolution adapted for K channels
  - Feature fusion and MLP classification head
- **Output**: Final classification with 4-class probabilities

## Module Structure

```
EEG-Channel_Selection/
├── config.py              # Configuration management
├── preprocessing.py        # Data preprocessing utilities
├── scnn_model.py          # Shallow CNN implementation
├── channel_selection.py   # Channel selection logic
├── fusion_cnn.py          # Multi-Layer Fusion CNN
├── pipeline.py            # Main pipeline orchestration
├── utils.py               # Visualization and analysis tools
├── demo.py                # Example usage and demos
└── __init__.py            # Package initialization
```

## Configuration

The pipeline uses a centralized configuration system. Key parameters can be customized:

```python
from config import config, EEGConfig

# Use default configuration
print(config.K_selected)  # 6
print(config.sampling_rate)  # 250

# Create custom configuration
custom_config = EEGConfig(
    K_selected=8,
    sampling_rate=500,
    max_epochs=50,
    scnn_learning_rate=5e-4
)
```

## Key Functions

### Data Preprocessing

- `preprocess_data_for_selection()`: Bandpass filtering, standardization, segmentation
- `generate_single_channel_inputs()`: Create individual channel datasets
- `prepare_fusion_input()`: Filter data into frequency bands

### Channel Selection

- `select_best_channels()`: Train SCNNs on each channel and select top K
- `build_shallow_cnn_selector()`: Create SCNN model architecture

### Classification

- `build_fusion_cnn_classifier()`: Create Multi-Layer Fusion CNN
- `run_full_pipeline()`: Execute complete two-stage pipeline

### Analysis & Visualization

- `plot_channel_selection_results()`: Visualize channel selection outcomes
- `plot_training_history()`: Show training progress across folds
- `generate_comprehensive_report()`: Create complete analysis report

## Requirements

```
torch >= 1.9.0
numpy >= 1.19.0
scipy >= 1.6.0
scikit-learn >= 0.24.0
matplotlib >= 3.3.0  # For visualization
seaborn >= 0.11.0    # For visualization
pandas >= 1.2.0      # For data analysis
tqdm >= 4.60.0       # For progress bars
```

## Supported Datasets

### BCI Competition IV Dataset 2a

Built-in support for the popular BCI Competition IV Dataset 2a:
- **Subjects**: 9 subjects (A01-A09)
- **Classes**: 4 motor imagery tasks (left hand, right hand, feet, tongue)
- **Channels**: 22 EEG channels
- **Sampling Rate**: 250 Hz
- **Sessions**: Training (288 trials) + Evaluation (288 trials) per subject

**Quick Start with BCICIV-2a:**
```bash
# Test your dataset
python bciciv_2a_quickstart.py --data_path /path/to/bciciv_2a

# Train single subject
python bciciv_2a_example.py --data_path /path/to/bciciv_2a --subject A01

# Compare multiple subjects
python bciciv_2a_example.py --data_path /path/to/bciciv_2a --mode compare --subjects A01 A02 A03
```

## Example Output

```
=== CHANNEL SELECTION RESULTS ===
Selected top 6 channels:
  1. Channel 8 (C3): 0.8945
  2. Channel 10 (C4): 0.8723
  3. Channel 2 (F7): 0.8456
  4. Channel 14 (P3): 0.8234
  5. Channel 16 (P4): 0.8123
  6. Channel 9 (Cz): 0.7998

Cross-Validation Results (5 folds):
  Fold 1: 0.9234
  Fold 2: 0.9123
  Fold 3: 0.9345
  Fold 4: 0.9078
  Fold 5: 0.9267

Final Classification Performance:
  Mean Accuracy: 0.9209 ± 0.0108
  Best Fold: 0.9345
  Worst Fold: 0.9078
```

## Advanced Usage

### Custom Model Architecture

```python
from scnn_model import ShallowCNN
from fusion_cnn import FusionCNN

# Custom SCNN
scnn = ShallowCNN(
    input_time_samples=1001,
    n_classes=4,
    filters_temporal=60,  # More filters
    dropout_rate=0.3      # Less dropout
)

# Custom Fusion CNN
fusion_cnn = FusionCNN(
    K_channels=8,
    input_time_samples=1001,
    n_classes=4,
    mlp_units=[100, 50]   # Larger MLP
)
```

### Cross-Validation Analysis

```python
from sklearn.model_selection import StratifiedKFold
from pipeline import run_full_pipeline

# Custom cross-validation strategy
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

results = run_full_pipeline(
    raw_eeg_data=eeg_data,
    labels=labels,
    cv_folds=10,  # 10-fold CV
    save_results=True,
    results_prefix="detailed_analysis"
)
```

### Batch Processing

```python
import glob

# Process multiple datasets
datasets = glob.glob("data/*.npz")

for dataset_path in datasets:
    data = np.load(dataset_path)
    eeg_data = data['eeg_data']
    labels = data['labels']

    results = run_full_pipeline(
        raw_eeg_data=eeg_data,
        labels=labels,
        results_prefix=f"analysis_{Path(dataset_path).stem}"
    )
```

## Performance Tips

1. **GPU Acceleration**: The pipeline automatically uses CUDA if available
2. **Memory Management**: Large datasets are processed in batches
3. **Early Stopping**: Training uses validation-based early stopping
4. **Parallel Processing**: Channel selection can be parallelized across channels

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{eeg_channel_selection_2024,
  title={Two-Stage EEG Channel Selection and Classification},
  author={EEG Analysis Team},
  year={2024},
  version={1.0.0},
  url={https://github.com/your-repo/eeg-channel-selection}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please read the contributing guidelines and submit pull requests for any improvements.

## Support

For questions and support, please open an issue on the GitHub repository or contact the development team.
# EEG-Channel-Selection-Using-SCNN-
