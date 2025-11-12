"""
BCI Competition IV Dataset 2a data loader for EEG Channel Selection Pipeline

This module provides functions to load, preprocess, and format the BCICIV-2a dataset
for use with the two-stage EEG channel selection and classification pipeline.

Dataset Information:
- 9 subjects (A01-A09)
- 4 motor imagery classes: left hand (1), right hand (2), feet (3), tongue (4)
- 22 EEG channels + 3 EOG channels
- Sampling rate: 250 Hz
- Session structure: Training (288 trials) + Evaluation (288 trials) per subject

Author: EEG Analysis Team
"""

import numpy as np
import mne
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union
import logging
import pickle

# Setup logging
logger = logging.getLogger(__name__)

# BCI Competition IV Dataset 2a channel information
BCICIV_2A_CHANNELS = {
    'eeg_channels': [
        'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 
        'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'
    ],
    'eog_channels': ['EOG1', 'EOG2', 'EOG3'],
    'sampling_rate': 250,
    'n_classes': 4,
    'class_names': ['left_hand', 'right_hand', 'feet', 'tongue'],
    'class_ids': [1, 2, 3, 4]  # Original class IDs in the dataset
}

def load_bciciv_2a_subject(
    data_path: Union[str, Path], 
    subject_id: str,
    session: str = 'T',  # 'T' for training, 'E' for evaluation
    use_eeg_only: bool = True,
    trial_length_sec: float = 4.0,
    baseline_length_sec: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load BCI Competition IV Dataset 2a data for a specific subject.
    
    Args:
        data_path: Path to the dataset directory
        subject_id: Subject identifier ('A01' to 'A09')
        session: Session type ('T' for training, 'E' for evaluation)
        use_eeg_only: If True, use only EEG channels (exclude EOG)
        trial_length_sec: Length of each trial in seconds
        baseline_length_sec: Baseline period length in seconds
        
    Returns:
        Tuple of (eeg_data, labels, channel_names)
        - eeg_data: (n_trials, n_channels, n_samples)
        - labels: (n_trials,) with class labels 0-3 (remapped from 1-4)
        - channel_names: List of channel names
    """
    data_path = Path(data_path)
    
    # Construct filename
    filename = f"{subject_id}{session}.gdf"
    filepath = data_path / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    
    logger.info(f"Loading BCI Competition IV Dataset 2a: {filename}")
    
    # Load GDF file using MNE
    raw = mne.io.read_raw_gdf(str(filepath), verbose=False, preload=True)
    
    # Get channel information - handle the mangled channel names
    all_channel_names = raw.ch_names
    
    # The first 22 channels are EEG channels, next 3 are EOG
    if use_eeg_only:
        eeg_indices = list(range(22))  # First 22 channels are EEG
        # Use the standard channel names
        channel_names = BCICIV_2A_CHANNELS['eeg_channels']
    else:
        eeg_indices = list(range(25))  # 22 EEG + 3 EOG
        channel_names = BCICIV_2A_CHANNELS['eeg_channels'] + BCICIV_2A_CHANNELS['eog_channels']
    
    # Get raw EEG data (channels x samples)
    raw_data = raw.get_data()[eeg_indices, :]  # Shape: (n_channels, n_samples)
    
    # Get events from annotations (GDF files store events as annotations)
    try:
        events = mne.find_events(raw, verbose=False)
        event_id = None
    except ValueError:
        # If no event channel found, convert from annotations
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        logger.info(f"Found events from annotations: {list(event_id.keys())}")
    
    # Map event codes - MNE remaps annotation codes to sequential integers
    if event_id is not None:
        # Find the remapped codes for our events of interest
        trial_start_code = event_id.get('768', None)  # Trial start
        left_code = event_id.get('769', None)         # Left hand
        right_code = event_id.get('770', None)        # Right hand  
        feet_code = event_id.get('771', None)         # Feet
        tongue_code = event_id.get('772', None)       # Tongue
        
        logger.info(f"Event code mapping: start={trial_start_code}, classes=[{left_code},{right_code},{feet_code},{tongue_code}]")
    else:
        # Use original codes if no remapping
        trial_start_code = 768
        left_code, right_code, feet_code, tongue_code = 769, 770, 771, 772
    
    # Extract trial information
    if session == 'T':  # Training session has class labels        
        # Find trial start events and class events
        if trial_start_code is not None:
            trial_starts = events[events[:, 2] == trial_start_code, 0]
        else:
            # Fallback: use class events as trial markers
            trial_starts = events[(events[:, 2] >= min(left_code, right_code, feet_code, tongue_code)) & 
                                (events[:, 2] <= max(left_code, right_code, feet_code, tongue_code)), 0]
            
        class_codes = [left_code, right_code, feet_code, tongue_code]
        class_events = events[np.isin(events[:, 2], class_codes)]
        
        labels = []
        trial_info = []
        
        if len(trial_starts) > 0:
            # Method 1: Use trial starts + following class events
            for start_sample in trial_starts:
                # Find the class event that follows this trial start
                class_event_idx = np.where(class_events[:, 0] > start_sample)[0]
                if len(class_event_idx) > 0:
                    class_code = class_events[class_event_idx[0], 2]
                    # Map class code to label
                    if class_code == left_code:
                        class_label = 0
                    elif class_code == right_code:
                        class_label = 1
                    elif class_code == feet_code:
                        class_label = 2
                    elif class_code == tongue_code:
                        class_label = 3
                    else:
                        continue  # Skip unknown class
                        
                    labels.append(class_label)
                    
                    # Trial spans from cue (class event) to end of trial period
                    cue_sample = class_events[class_event_idx[0], 0]
                    trial_end = cue_sample + int(trial_length_sec * raw.info['sfreq'])
                    trial_info.append([cue_sample, trial_end])
        else:
            # Method 2: Use class events directly as trials
            logger.info("No trial start events found, using class events as trials")
            for event in class_events:
                class_code = event[2]
                # Map class code to label
                if class_code == left_code:
                    class_label = 0
                elif class_code == right_code:
                    class_label = 1
                elif class_code == feet_code:
                    class_label = 2
                elif class_code == tongue_code:
                    class_label = 3
                else:
                    continue
                    
                labels.append(class_label)
                cue_sample = event[0]
                trial_end = cue_sample + int(trial_length_sec * raw.info['sfreq'])
                trial_info.append([cue_sample, trial_end])
        
        labels = np.array(labels)
        trial_info = np.array(trial_info)
        
    else:  # Evaluation session - no labels available
        trial_starts = events[events[:, 2] == 768, 0]
        labels = np.zeros(len(trial_starts))  # Placeholder labels
        trial_info = []
        
        for start_sample in trial_starts:
            trial_end = start_sample + int(trial_length_sec * raw.info['sfreq'])
            trial_info.append([start_sample, trial_end])
        
        trial_info = np.array(trial_info)
    
    # Parameters
    sampling_rate = int(raw.info['sfreq'])  # Use actual sampling rate from GDF
    n_samples_per_trial = int(trial_length_sec * sampling_rate)
    baseline_samples = int(baseline_length_sec * sampling_rate)
    
    # Extract trials
    n_trials = len(trial_info)
    n_channels = len(channel_names)
    eeg_data = np.zeros((n_trials, n_channels, n_samples_per_trial))
    
    logger.info(f"Extracting {n_trials} trials with {n_channels} channels")
    
    for trial_idx in range(n_trials):
        # Get trial start and end indices  
        trial_start = int(trial_info[trial_idx, 0])  # Already in samples
        trial_end = int(trial_info[trial_idx, 1])
        
        # Make sure we don't exceed data bounds
        if trial_start + n_samples_per_trial <= raw_data.shape[1]:
            # Extract trial data (channels x samples format from MNE)
            trial_data = raw_data[:, trial_start:trial_start + n_samples_per_trial]
            eeg_data[trial_idx] = trial_data  # Shape: (n_channels, n_samples)
        else:
            logger.warning(f"Trial {trial_idx} extends beyond data length, skipping")
            eeg_data[trial_idx] = np.zeros((n_channels, n_samples_per_trial))
    
    # Labels are already in 0-3 format from the conversion above
    labels = labels.astype(int)
    
    logger.info(f"Loaded data shape: {eeg_data.shape}")
    logger.info(f"Labels shape: {labels.shape}")
    if session == 'T':
        logger.info(f"Class distribution: {np.bincount(labels)}")
    
    return eeg_data, labels, channel_names


def load_all_subjects_bciciv_2a(
    data_path: Union[str, Path],
    subjects: Optional[List[str]] = None,
    session: str = 'T',
    use_eeg_only: bool = True,
    combine_subjects: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray, List[str]], Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]]]:
    """
    Load BCI Competition IV Dataset 2a data for multiple subjects.
    
    Args:
        data_path: Path to the dataset directory
        subjects: List of subject IDs to load (default: all A01-A09)
        session: Session type ('T' for training, 'E' for evaluation)
        use_eeg_only: If True, use only EEG channels
        combine_subjects: If True, combine all subjects into single arrays
        
    Returns:
        If combine_subjects=True: Tuple of (combined_data, combined_labels, channel_names)
        If combine_subjects=False: Dict mapping subject_id -> (data, labels, channel_names)
    """
    if subjects is None:
        subjects = [f'A{i:02d}' for i in range(1, 10)]  # A01 to A09
    
    logger.info(f"Loading BCI Competition IV Dataset 2a for subjects: {subjects}")
    
    subject_data = {}
    
    for subject_id in subjects:
        try:
            eeg_data, labels, channel_names = load_bciciv_2a_subject(
                data_path=data_path,
                subject_id=subject_id,
                session=session,
                use_eeg_only=use_eeg_only
            )
            subject_data[subject_id] = (eeg_data, labels, channel_names)
            logger.info(f"Successfully loaded {subject_id}: {eeg_data.shape}")
            
        except Exception as e:
            logger.error(f"Failed to load subject {subject_id}: {str(e)}")
            continue
    
    if not subject_data:
        raise ValueError("No subjects were successfully loaded")
    
    if combine_subjects:
        # Combine all subjects
        all_data = []
        all_labels = []
        
        for subject_id, (data, labels, _) in subject_data.items():
            all_data.append(data)
            if session == 'T':  # Only combine labels for training session
                all_labels.append(labels)
        
        combined_data = np.concatenate(all_data, axis=0)
        combined_labels = np.concatenate(all_labels, axis=0) if all_labels else np.array([])
        
        logger.info(f"Combined data shape: {combined_data.shape}")
        logger.info(f"Combined labels shape: {combined_labels.shape}")
        
        return combined_data, combined_labels, channel_names
    
    else:
        return subject_data


def preprocess_bciciv_2a_for_pipeline(
    eeg_data: np.ndarray,
    labels: np.ndarray,
    channel_names: List[str],
    target_length_sec: float = 3.0,
    sampling_rate: int = 250
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Preprocess BCI Competition IV Dataset 2a data for the EEG pipeline.
    
    Args:
        eeg_data: EEG data (n_trials, n_channels, n_samples)
        labels: Class labels (n_trials,)
        channel_names: List of channel names
        target_length_sec: Target trial length in seconds
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Tuple of (processed_data, processed_labels, channel_names)
    """
    target_samples = int(target_length_sec * sampling_rate)
    current_samples = eeg_data.shape[2]
    
    logger.info(f"Preprocessing BCICIV-2a data for pipeline")
    logger.info(f"Original shape: {eeg_data.shape}")
    logger.info(f"Target length: {target_length_sec}s ({target_samples} samples)")
    
    if current_samples == target_samples:
        logger.info("Data already has target length")
        return eeg_data, labels, channel_names
    
    elif current_samples > target_samples:
        # Truncate to target length (take middle portion)
        start_idx = (current_samples - target_samples) // 2
        end_idx = start_idx + target_samples
        processed_data = eeg_data[:, :, start_idx:end_idx]
        logger.info(f"Truncated data to {target_samples} samples")
        
    else:
        # Pad with zeros if too short
        pad_samples = target_samples - current_samples
        pad_left = pad_samples // 2
        pad_right = pad_samples - pad_left
        processed_data = np.pad(eeg_data, ((0, 0), (0, 0), (pad_left, pad_right)), mode='constant')
        logger.info(f"Padded data to {target_samples} samples")
    
    logger.info(f"Final processed shape: {processed_data.shape}")
    
    return processed_data, labels, channel_names


def create_train_test_split_bciciv_2a(
    data_path: Union[str, Path],
    subject_ids: Optional[List[str]] = None,
    test_ratio: float = 0.2,
    use_eeg_only: bool = True,
    random_seed: int = 42
) -> Tuple[Tuple[np.ndarray, np.ndarray, List[str]], Tuple[np.ndarray, np.ndarray, List[str]]]:
    """
    Create train/test split from BCI Competition IV Dataset 2a training data.
    
    Args:
        data_path: Path to dataset directory
        subject_ids: List of subjects to include
        test_ratio: Fraction of data to use for testing
        use_eeg_only: If True, use only EEG channels
        random_seed: Random seed for reproducible splits
        
    Returns:
        Tuple of ((train_data, train_labels, channel_names), (test_data, test_labels, channel_names))
    """
    # Load training data
    combined_data, combined_labels, channel_names = load_all_subjects_bciciv_2a(
        data_path=data_path,
        subjects=subject_ids,
        session='T',  # Training session
        use_eeg_only=use_eeg_only,
        combine_subjects=True
    )
    
    # Preprocess for pipeline (3-second segments)
    processed_data, processed_labels, channel_names = preprocess_bciciv_2a_for_pipeline(
        combined_data, combined_labels, channel_names
    )
    
    # Create stratified split
    from sklearn.model_selection import train_test_split
    
    np.random.seed(random_seed)
    
    train_data, test_data, train_labels, test_labels = train_test_split(
        processed_data, processed_labels,
        test_size=test_ratio,
        stratify=processed_labels,
        random_state=random_seed
    )
    
    logger.info(f"Train/test split created:")
    logger.info(f"  Train: {train_data.shape}, Labels: {len(train_labels)}")
    logger.info(f"  Test:  {test_data.shape}, Labels: {len(test_labels)}")
    logger.info(f"  Train class distribution: {np.bincount(train_labels)}")
    logger.info(f"  Test class distribution:  {np.bincount(test_labels)}")
    
    return (train_data, train_labels, channel_names), (test_data, test_labels, channel_names)


def save_bciciv_2a_processed(
    data: Tuple[np.ndarray, np.ndarray, List[str]],
    save_path: Union[str, Path],
    description: str = "BCICIV-2a processed data"
):
    """
    Save processed BCI Competition IV Dataset 2a data.
    
    Args:
        data: Tuple of (eeg_data, labels, channel_names)
        save_path: Path to save the data
        description: Description of the data
    """
    eeg_data, labels, channel_names = data
    
    save_dict = {
        'eeg_data': eeg_data,
        'labels': labels,
        'channel_names': channel_names,
        'dataset_info': {
            'dataset': 'BCI Competition IV Dataset 2a',
            'description': description,
            'n_trials': len(labels),
            'n_channels': len(channel_names),
            'n_samples': eeg_data.shape[2],
            'n_classes': len(np.unique(labels)),
            'class_distribution': np.bincount(labels).tolist(),
            'sampling_rate': BCICIV_2A_CHANNELS['sampling_rate']
        }
    }
    
    save_path = Path(save_path)
    
    if save_path.suffix == '.npz':
        np.savez_compressed(save_path, **save_dict)
    elif save_path.suffix == '.pkl':
        with open(save_path, 'wb') as f:
            pickle.dump(save_dict, f)
    else:
        # Default to npz
        save_path = save_path.with_suffix('.npz')
        np.savez_compressed(save_path, **save_dict)
    
    logger.info(f"Saved processed BCICIV-2a data to {save_path}")
    logger.info(f"Data shape: {eeg_data.shape}, Labels: {len(labels)}")


def load_bciciv_2a_processed(load_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load previously processed BCI Competition IV Dataset 2a data.
    
    Args:
        load_path: Path to the saved data file
        
    Returns:
        Tuple of (eeg_data, labels, channel_names)
    """
    load_path = Path(load_path)
    
    if load_path.suffix == '.npz':
        data = np.load(load_path, allow_pickle=True)
        eeg_data = data['eeg_data']
        labels = data['labels']
        channel_names = data['channel_names'].tolist()
        
    elif load_path.suffix == '.pkl':
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        eeg_data = data['eeg_data']
        labels = data['labels'] 
        channel_names = data['channel_names']
        
    else:
        raise ValueError(f"Unsupported file format: {load_path.suffix}")
    
    logger.info(f"Loaded processed BCICIV-2a data from {load_path}")
    logger.info(f"Data shape: {eeg_data.shape}, Labels: {len(labels)}")
    
    return eeg_data, labels, channel_names


# Example usage and test functions
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="BCI Competition IV Dataset 2a Data Loader")
    parser.add_argument("--data_path", type=str, required=True, help="Path to BCICIV-2a dataset")
    parser.add_argument("--subject", type=str, default="A01", help="Subject ID (A01-A09)")
    parser.add_argument("--session", type=str, default="T", choices=['T', 'E'], help="Session type")
    parser.add_argument("--output", type=str, help="Output path to save processed data")
    
    args = parser.parse_args()
    
    # Test loading single subject
    try:
        eeg_data, labels, channel_names = load_bciciv_2a_subject(
            data_path=args.data_path,
            subject_id=args.subject,
            session=args.session
        )
        
        print(f"Successfully loaded {args.subject}:")
        print(f"  Data shape: {eeg_data.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Channels: {len(channel_names)}")
        print(f"  Channel names: {channel_names}")
        
        if args.session == 'T':
            print(f"  Class distribution: {np.bincount(labels)}")
        
        if args.output:
            save_bciciv_2a_processed(
                (eeg_data, labels, channel_names),
                args.output,
                f"Subject {args.subject} session {args.session}"
            )
            
    except Exception as e:
        print(f"Error loading data: {e}")
        raise