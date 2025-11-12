"""
Data preprocessing utilities for EEG channel selection and classification
"""
import numpy as np
import torch
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Union
import logging

from config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_data_for_selection(
    data_raw: np.ndarray, 
    sampling_rate: int = None,
    ch_names: List[str] = None,
    target_freq: Tuple[int, int] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Preprocess raw EEG data for channel selection stage.
    
    Args:
        data_raw: Raw EEG data of shape (n_channels, n_samples) or (n_trials, n_channels, n_samples)
        sampling_rate: Sampling rate in Hz
        ch_names: List of channel names
        target_freq: Frequency band for filtering (low, high)
        
    Returns:
        Tuple of (processed_data, channel_names)
        - processed_data: Standardized, segmented data (n_trials, n_channels, time_samples)
        - channel_names: List of channel names
    """
    if sampling_rate is None:
        sampling_rate = config.sampling_rate
    if target_freq is None:
        target_freq = config.target_freq
    if ch_names is None:
        ch_names = [f"Ch_{i}" for i in range(data_raw.shape[-2] if data_raw.ndim == 3 else data_raw.shape[0])]
    
    logger.info(f"Preprocessing data with shape {data_raw.shape}")
    logger.info(f"Applying bandpass filter: {target_freq[0]}-{target_freq[1]} Hz")
    
    # Task 1: Filtering - Apply Butterworth bandpass filter
    nyquist = sampling_rate / 2
    low = target_freq[0] / nyquist
    high = target_freq[1] / nyquist
    
    # Design Butterworth filter
    if low <= 0:
        # Use lowpass filter if low frequency is 0
        b, a = butter(N=4, Wn=high, btype='low')
    elif high >= 1:
        # Use highpass filter if high frequency is above Nyquist
        b, a = butter(N=4, Wn=low, btype='high')
    else:
        # Use bandpass filter
        b, a = butter(N=4, Wn=[low, high], btype='band')
    
    # Handle different input shapes
    if data_raw.ndim == 2:  # (n_channels, n_samples)
        filtered_data = np.zeros_like(data_raw)
        for ch in range(data_raw.shape[0]):
            filtered_data[ch, :] = filtfilt(b, a, data_raw[ch, :])
        # Convert to trial format by segmenting
        filtered_data = _segment_continuous_data(filtered_data, sampling_rate)
    elif data_raw.ndim == 3:  # (n_trials, n_channels, n_samples)
        filtered_data = np.zeros_like(data_raw)
        for trial in range(data_raw.shape[0]):
            for ch in range(data_raw.shape[1]):
                filtered_data[trial, ch, :] = filtfilt(b, a, data_raw[trial, ch, :])
    else:
        raise ValueError(f"Unsupported data shape: {data_raw.shape}")
    
    # Task 2: Standardization - Z-score normalization per channel
    logger.info("Applying standardization (z-score normalization)")
    standardized_data = np.zeros_like(filtered_data)
    
    for ch in range(filtered_data.shape[1]):
        channel_data = filtered_data[:, ch, :].flatten()
        scaler = StandardScaler()
        normalized_flat = scaler.fit_transform(channel_data.reshape(-1, 1)).flatten()
        standardized_data[:, ch, :] = normalized_flat.reshape(filtered_data.shape[0], -1)
    
    # Task 3: Ensure correct segmentation (already handled above for continuous data)
    logger.info(f"Final processed data shape: {standardized_data.shape}")
    
    return standardized_data, ch_names


def _segment_continuous_data(continuous_data: np.ndarray, sampling_rate: int) -> np.ndarray:
    """
    Segment continuous EEG data into trials.
    
    Args:
        continuous_data: Continuous EEG data (n_channels, n_samples)
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Segmented data (n_trials, n_channels, time_samples)
    """
    segment_samples = int(config.segment_duration * sampling_rate)
    n_channels, total_samples = continuous_data.shape
    
    # Calculate number of complete segments
    n_trials = total_samples // segment_samples
    
    # Extract segments
    segmented_data = np.zeros((n_trials, n_channels, segment_samples))
    
    for trial in range(n_trials):
        start_idx = trial * segment_samples
        end_idx = start_idx + segment_samples
        segmented_data[trial, :, :] = continuous_data[:, start_idx:end_idx]
    
    logger.info(f"Segmented data into {n_trials} trials of {segment_samples} samples each")
    return segmented_data


def generate_single_channel_inputs(
    data_processed: np.ndarray, 
    labels: np.ndarray
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Generate single-channel datasets for SCNN training.
    
    Args:
        data_processed: Processed EEG data (n_trials, n_channels, time_samples)
        labels: Labels for each trial (n_trials,)
        
    Returns:
        List of tuples (X_channel, y_labels) for each channel
    """
    n_trials, n_channels, time_samples = data_processed.shape
    logger.info(f"Generating single-channel inputs for {n_channels} channels")
    
    channel_datasets = []
    
    for ch in range(n_channels):
        # Extract single channel data
        X_channel = data_processed[:, ch, :].reshape(n_trials, time_samples, 1)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_channel)
        y_tensor = torch.LongTensor(labels)
        
        channel_datasets.append((X_tensor, y_tensor))
        logger.debug(f"Channel {ch}: X shape {X_tensor.shape}, y shape {y_tensor.shape}")
    
    logger.info(f"Generated {len(channel_datasets)} single-channel datasets")
    return channel_datasets


def prepare_fusion_input(
    raw_data_selected_channels: np.ndarray,
    sampling_rate: int = None,
    ch_names: List[str] = None
) -> List[np.ndarray]:
    """
    Prepare fusion inputs by filtering selected channels into different frequency bands.
    
    Args:
        raw_data_selected_channels: Raw EEG data for selected channels (n_trials, K_channels, time_samples)
        sampling_rate: Sampling rate in Hz
        ch_names: Channel names (optional)
        
    Returns:
        List of four filtered data arrays for different frequency bands
    """
    if sampling_rate is None:
        sampling_rate = config.sampling_rate
    
    logger.info(f"Preparing fusion inputs for {len(config.fusion_bands)} frequency bands")
    logger.info(f"Input data shape: {raw_data_selected_channels.shape}")
    
    fusion_inputs = []
    
    for i, (low_freq, high_freq) in enumerate(config.fusion_bands):
        logger.info(f"Processing band {i+1}: {low_freq}-{high_freq} Hz")
        
        # Apply bandpass filter for this frequency band
        filtered_data, _ = preprocess_data_for_selection(
            raw_data_selected_channels,
            sampling_rate=sampling_rate,
            ch_names=ch_names,
            target_freq=(low_freq, high_freq)
        )
        
        fusion_inputs.append(filtered_data)
    
    logger.info(f"Generated {len(fusion_inputs)} fusion input arrays")
    return fusion_inputs


def create_data_loaders(
    X: torch.Tensor, 
    y: torch.Tensor, 
    batch_size: int = None,
    validation_split: float = None,
    shuffle: bool = True
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        X: Input tensor
        y: Target tensor  
        batch_size: Batch size for data loaders
        validation_split: Fraction of data to use for validation
        shuffle: Whether to shuffle the data
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    if batch_size is None:
        batch_size = config.batch_size
    if validation_split is None:
        validation_split = config.validation_split
    
    dataset = torch.utils.data.TensorDataset(X, y)
    
    # Split into train and validation
    n_samples = len(dataset)
    n_val = int(validation_split * n_samples)
    n_train = n_samples - n_val
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val]
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, val_loader