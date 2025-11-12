# 🧠 BCI Competition IV Dataset 2a - Complete Integration Guide

## What You Now Have

I've created a complete integration for the BCI Competition IV Dataset 2a with your EEG Channel Selection pipeline! Here's everything that's been added:

## 📁 New Files Created

### Core BCICIV-2a Support
- **`bciciv_2a_loader.py`** - Complete data loader for BCICIV-2a format
- **`bciciv_2a_example.py`** - Comprehensive examples and training scripts  
- **`bciciv_2a_quickstart.py`** - Quick validation and testing script

## 🚀 Quick Start Commands

### 1. Test Your Dataset Structure
```bash
python bciciv_2a_quickstart.py --data_path /path/to/your/bciciv_2a_dataset
```
This will:
- Check if your .mat files are correctly placed
- Test loading a subject (default A01)
- Run a quick pipeline validation

### 2. Train Single Subject
```bash
python bciciv_2a_example.py --data_path /path/to/bciciv_2a --subject A01
```

### 3. Compare Multiple Subjects
```bash
python bciciv_2a_example.py --data_path /path/to/bciciv_2a --mode compare --subjects A01 A02 A03
```

### 4. Train/Test Evaluation
```bash
python bciciv_2a_example.py --data_path /path/to/bciciv_2a --mode train_test --all_subjects
```

## 📊 Dataset Information

The BCICIV-2a dataset contains:
- **9 Subjects**: A01, A02, ..., A09
- **4 Motor Imagery Classes**: 
  - Class 1: Left Hand
  - Class 2: Right Hand  
  - Class 3: Feet
  - Class 4: Tongue
- **22 EEG Channels**: Fz, FC3, FC1, FCz, FC2, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3, CP1, CPz, CP2, CP4, P1, Pz, P2, POz
- **Sampling Rate**: 250 Hz
- **Sessions**: Training (T) and Evaluation (E) files for each subject

## 💡 Expected File Structure

Your dataset directory should contain files like:
```
/path/to/bciciv_2a/
├── A01T.mat  # Subject 1 training
├── A01E.mat  # Subject 1 evaluation  
├── A02T.mat  # Subject 2 training
├── A02E.mat  # Subject 2 evaluation
├── ...
├── A09T.mat  # Subject 9 training
└── A09E.mat  # Subject 9 evaluation
```

## 🔧 Optimized Configuration

The pipeline automatically uses optimized settings for motor imagery:
- **Frequency Band**: 8-30 Hz (focuses on mu and beta rhythms)
- **Channel Selection**: 8 most informative channels (default)
- **Segment Length**: 3 seconds
- **Cross-Validation**: 5 folds

## 📝 Example Usage in Python

```python
from bciciv_2a_loader import load_bciciv_2a_subject
from bciciv_2a_example import train_single_subject, setup_bciciv_2a_config

# Load data for a single subject
eeg_data, labels, channel_names = load_bciciv_2a_subject(
    data_path="/path/to/bciciv_2a",
    subject_id="A01",
    session='T'  # Training session
)

print(f"Loaded data shape: {eeg_data.shape}")
print(f"Number of trials: {len(labels)}")
print(f"Channel names: {channel_names}")

# Run complete pipeline
config = setup_bciciv_2a_config()
results = train_single_subject(
    data_path="/path/to/bciciv_2a",
    subject_id="A01", 
    config=config
)

print(f"Final accuracy: {results['final_results']['mean_accuracy']:.4f}")
print(f"Selected channels: {results['stage1_results']['selected_names']}")
```

## 🎯 What The Pipeline Will Do

1. **Load BCICIV-2a Data**: Automatically handles .mat file format
2. **Preprocess**: Apply motor imagery optimized filtering (8-30 Hz)
3. **Stage 1 - Channel Selection**: 
   - Train individual SCNNs on each of 22 EEG channels
   - Select top K channels (default: 8) based on classification performance
4. **Stage 2 - Fusion Classification**:
   - Process selected channels across 4 frequency bands
   - Train Multi-Layer Fusion CNN
   - 5-fold cross-validation for robust evaluation
5. **Generate Results**: 
   - Classification accuracy
   - Selected channel analysis
   - Comprehensive visualizations and reports

## 📈 Expected Results

For BCICIV-2a dataset, you can expect:
- **Channel Selection**: Typically selects central motor channels (C3, C4, Cz, etc.)
- **Accuracy Range**: Usually 60-85% depending on subject
- **Best Subjects**: Some subjects perform much better than others
- **Selected Channels**: Often includes C3, C4, Cz (motor cortex areas)

## 🛠️ Troubleshooting

### Common Issues:

1. **"FileNotFoundError"**: Check that your .mat files are in the correct path
2. **"ValueError in filter"**: Some frequency bands might need adjustment for your data
3. **"Memory Error"**: Reduce batch size or use fewer subjects simultaneously

### Debug Commands:
```bash
# Check if dataset is readable
python bciciv_2a_quickstart.py --data_path /your/path --skip_pipeline

# Test with minimal parameters
python bciciv_2a_quickstart.py --data_path /your/path --subject A01
```

## 🔄 Next Steps After Getting Your Results

1. **Analyze Selected Channels**: See which brain regions are most informative
2. **Compare Subjects**: Identify best and worst performing subjects
3. **Optimize Parameters**: Adjust frequency bands, K value, or model architecture
4. **Evaluate on Test Set**: Use evaluation sessions for final performance assessment

## 📊 Example Output

```
=== CHANNEL SELECTION RESULTS ===
Selected top 8 channels:
  1. C3 (Index 7): SCNN Accuracy = 0.8234  
  2. C4 (Index 11): SCNN Accuracy = 0.8156
  3. Cz (Index 9): SCNN Accuracy = 0.7891
  4. CP3 (Index 13): SCNN Accuracy = 0.7654
  5. CP4 (Index 17): SCNN Accuracy = 0.7432
  ...

Final Classification Performance:
  Mean Accuracy: 0.7823 ± 0.0234
  Best Fold: 0.8156
  Worst Fold: 0.7345
```

You're all set to train and test with your BCICIV-2a dataset! 🎉