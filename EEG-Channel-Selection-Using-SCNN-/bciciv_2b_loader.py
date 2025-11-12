"""
BCI Competition IV Dataset 2b data loader for EEG Channel Selection Pipeline

This module provides functions to load, preprocess, and format the BCICIV-2b dataset
for use with the two-stage EEG channel selection and classification pipeline.

Dataset Information:
- 9 subjects (B01-B09) 
- 2 motor imagery classes: left hand vs right hand (binary classification)
- 3 EEG channels: C3, Cz, C4 (motor cortex area)
- Sampling rate: 250 Hz
- Session structure per subject:
  * B01T, B02T, B03T: Training sessions
  * B04E, B05E: Evaluation sessions

Key Differences from Dataset 2a:
- Only 3 EEG channels (vs 22 in 2a)
- Binary classification (vs 4-class in 2a)  
- Multiple training sessions per subject (3 vs 1 in 2a)

Author: EEG Analysis Team
"""

import numpy as np
import mne
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union
import logging

# Setup logging
logger = logging.getLogger(__name__)

# BCI Competition IV Dataset 2b channel information
BCICIV_2B_CHANNELS = {
    'eeg_channels': ['C3', 'Cz', 'C4'],  # Only 3 motor cortex channels
    'sampling_rate': 250,
    'n_classes': 2,
    'class_names': ['left_hand', 'right_hand'],
    'class_ids': [1, 2]  # Original class IDs in the dataset
}

def load_bciciv_2b_subject(
    data_path: Union[str, Path], 
    subject_id: str,  # Format: 'B01', 'B02', etc.
    sessions: List[str] = None,  # List of sessions: ['01T', '02T', '03T'] or ['04E', '05E']
    trial_length_sec: float = 4.0,
    baseline_length_sec: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load BCI Competition IV Dataset 2b data for a specific subject.
    
    Args:
        data_path: Path to the directory containing GDF files
        subject_id: Subject identifier (e.g., 'B01', 'B02', ..., 'B09')
        sessions: List of session IDs to load. If None, loads all training sessions.
                 Examples: ['01T', '02T', '03T'] for training
                          ['04E', '05E'] for evaluation
        trial_length_sec: Length of each trial in seconds
        baseline_length_sec: Length of baseline period to skip
    
    Returns:
        Tuple of (eeg_data, labels, channel_names)
        - eeg_data: (n_trials, n_channels, n_samples) 
        - labels: (n_trials,) with class labels 0-1 (remapped from 1-2)
        - channel_names: List of channel names ['C3', 'Cz', 'C4']
    """
    data_path = Path(data_path)
    
    # Default sessions: all training sessions
    if sessions is None:
        sessions = ['01T', '02T', '03T']  # Training sessions
    
    all_trials = []
    all_labels = []
    channel_names = BCICIV_2B_CHANNELS['eeg_channels']
    
    logger.info(f"Loading BCICIV-2b subject {subject_id} sessions: {sessions}")
    
    for session in sessions:
        # Construct filename: B0101T.gdf, B0102T.gdf, etc.
        filename = f"{subject_id}{session}.gdf"
        filepath = data_path / filename
        
        if not filepath.exists():
            logger.warning(f"Session file not found: {filepath}")
            continue
            
        logger.info(f"Loading session: {filename}")
        
        # Load GDF file using MNE
        raw = mne.io.read_raw_gdf(str(filepath), verbose=False, preload=True)
        
        # Get raw EEG data (channels x samples) - only first 3 channels are EEG
        raw_data = raw.get_data()[:3, :]  # C3, Cz, C4
        
        # Get events from annotations
        try:
            events = mne.find_events(raw, verbose=False)
            event_id = None
        except ValueError:
            events, event_id = mne.events_from_annotations(raw, verbose=False)
        
        # Map event codes for BCICIV-2b
        if event_id is not None:
            left_code = event_id.get('769', None)   # Left hand
            right_code = event_id.get('770', None)  # Right hand
            trial_start_code = event_id.get('768', None)  # Trial start
        else:
            left_code, right_code = 769, 770
            trial_start_code = 768
        
        # Find class events (left/right hand cues)
        class_codes = [left_code, right_code]
        class_events = events[np.isin(events[:, 2], class_codes)]
        
        # Extract trials
        sampling_rate = int(raw.info['sfreq'])
        n_samples_per_trial = int(trial_length_sec * sampling_rate)
        
        session_trials = []
        session_labels = []
        
        for event in class_events:
            class_code = event[2]
            
            # Map class code to binary label
            if class_code == left_code:
                class_label = 0  # Left hand
            elif class_code == right_code:
                class_label = 1  # Right hand  
            else:
                continue
            
            # Extract trial data
            cue_sample = event[0]
            trial_start = cue_sample
            trial_end = trial_start + n_samples_per_trial
            
            if trial_end <= raw_data.shape[1]:
                trial_data = raw_data[:, trial_start:trial_end]  # (3, n_samples)
                session_trials.append(trial_data)
                session_labels.append(class_label)
        
        logger.info(f"Session {session}: {len(session_trials)} trials")
        all_trials.extend(session_trials)
        all_labels.extend(session_labels)
    
    # Convert to numpy arrays
    if len(all_trials) == 0:
        raise ValueError(f"No trials found for subject {subject_id}")
        
    eeg_data = np.array(all_trials)  # (n_trials, 3, n_samples)
    labels = np.array(all_labels, dtype=int)
    
    logger.info(f"Total loaded: {eeg_data.shape[0]} trials, {eeg_data.shape[1]} channels")
    logger.info(f"Class distribution: {np.bincount(labels)}")
    
    return eeg_data, labels, channel_names


def load_all_subjects_bciciv_2b(
    data_path: Union[str, Path],
    subjects: Optional[List[str]] = None,
    sessions: List[str] = None,
    trial_length_sec: float = 4.0
) -> Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]]:
    """
    Load multiple subjects from BCICIV-2b dataset.
    
    Args:
        data_path: Path to dataset directory
        subjects: List of subject IDs. If None, loads all subjects B01-B09
        sessions: List of sessions to load per subject
        trial_length_sec: Trial length in seconds
        
    Returns:
        Dictionary mapping subject_id -> (eeg_data, labels, channel_names)
    """
    data_path = Path(data_path)
    
    if subjects is None:
        subjects = [f'B{i:02d}' for i in range(1, 10)]  # B01-B09
    
    if sessions is None:
        sessions = ['01T', '02T', '03T']  # Training sessions by default
    
    results = {}
    
    for subject_id in subjects:
        try:
            eeg_data, labels, channel_names = load_bciciv_2b_subject(
                data_path, subject_id, sessions, trial_length_sec
            )
            results[subject_id] = (eeg_data, labels, channel_names)
            logger.info(f"✓ Loaded {subject_id}: {eeg_data.shape}")
        except Exception as e:
            logger.error(f"✗ Failed to load {subject_id}: {e}")
    
    return results


def preprocess_bciciv_2b_for_pipeline(
    eeg_data: np.ndarray,
    labels: np.ndarray, 
    channel_names: List[str],
    target_length_sec: float = 3.0,
    target_sampling_rate: int = 250
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Preprocess BCICIV-2b data for the EEG pipeline.
    
    Args:
        eeg_data: Raw EEG data (n_trials, n_channels, n_samples)
        labels: Class labels (n_trials,)
        channel_names: Channel names
        target_length_sec: Target trial length in seconds
        target_sampling_rate: Target sampling rate
    
    Returns:
        Preprocessed (eeg_data, labels, channel_names)
    """
    logger.info("Preprocessing BCICIV-2b data for pipeline")
    logger.info(f"Original shape: {eeg_data.shape}")
    
    # Calculate target number of samples
    target_samples = int(target_length_sec * target_sampling_rate)
    
    # Truncate or pad to target length
    current_samples = eeg_data.shape[2]
    if current_samples > target_samples:
        eeg_data = eeg_data[:, :, :target_samples]
        logger.info(f"Truncated data to {target_samples} samples")
    elif current_samples < target_samples:
        pad_width = ((0, 0), (0, 0), (0, target_samples - current_samples))
        eeg_data = np.pad(eeg_data, pad_width, mode='constant')
        logger.info(f"Padded data to {target_samples} samples")
    
    logger.info(f"Final processed shape: {eeg_data.shape}")
    
    return eeg_data, labels, channel_names


def create_train_test_split_bciciv_2b(
    data_path: Union[str, Path],
    subject_id: str,
    test_ratio: float = 0.3
) -> Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]]:
    """
    Create train/test split for BCICIV-2b using different sessions.
    
    Uses training sessions (01T, 02T, 03T) for training
    and evaluation sessions (04E, 05E) for testing.
    
    Args:
        data_path: Path to dataset directory  
        subject_id: Subject identifier (e.g., 'B01')
        test_ratio: Not used - split is based on session type
        
    Returns:
        Dictionary with 'train' and 'test' keys containing data tuples
    """
    logger.info(f"Creating train/test split for {subject_id}")
    
    # Load training sessions
    train_data, train_labels, channel_names = load_bciciv_2b_subject(
        data_path, subject_id, sessions=['01T', '02T', '03T']
    )
    
    # Load evaluation sessions  
    try:
        test_data, test_labels, _ = load_bciciv_2b_subject(
            data_path, subject_id, sessions=['04E', '05E']
        )
    except Exception as e:
        logger.warning(f"Could not load evaluation sessions: {e}")
        logger.info("Using training data split instead")
        
        # Fallback: split training data
        n_train = int(len(train_data) * (1 - test_ratio))
        indices = np.random.permutation(len(train_data))
        
        test_data = train_data[indices[n_train:]]
        test_labels = train_labels[indices[n_train:]]
        train_data = train_data[indices[:n_train]]
        train_labels = train_labels[indices[:n_train]]
    
    return {
        'train': (train_data, train_labels, channel_names),
        'test': (test_data, test_labels, channel_names)
    }


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python bciciv_2b_loader.py /path/to/bciciv_2b_gdf [subject]")
        sys.exit(1)
    
    data_path = sys.argv[1]
    subject_id = sys.argv[2] if len(sys.argv) > 2 else "B01"
    
    print(f"Testing BCICIV-2b loader with {subject_id}")
    
    # Test loading single subject
    eeg_data, labels, channels = load_bciciv_2b_subject(data_path, subject_id)
    print(f"Loaded: {eeg_data.shape}, Classes: {np.bincount(labels)}")
    
    # Test preprocessing
    processed_data, processed_labels, channels = preprocess_bciciv_2b_for_pipeline(
        eeg_data, labels, channels
    )
    print(f"Processed: {processed_data.shape}")