import torch
import librosa
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from utils.utils import extract_mel_spectrogram

class UrbanSound8KDataset(Dataset):
    def __init__(self, annotations_file, root_dir, folds, mode='train', transform=None):
        """
        UrbanSound8K dataset class.

        Args:
            annotations_file (str): Path to the annotations CSV file.
            root_dir (str): Root directory containing audio files.
            folds (list): List of fold numbers to include in the dataset.
            mode (str): 'train' for mel-spectrograms or 'attack' for waveforms.
            transform (callable, optional): Transformation function for data augmentation.
        """
        self.annotations = pd.read_csv(annotations_file)
        self.root_dir = root_dir
        self.transform = transform
        self.folds = folds
        self.mode = mode  # Mode: 'train' or 'attack'
        self.sr = 22050
        self.target_length = 4 * self.sr  # 4 seconds at 22050 Hz
        # Filter annotations to include only the specified folds
        self.annotations = self.annotations[self.annotations['fold'].isin(self.folds)]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset.

        Args:
            idx (int): Index of the item.

        Returns:
            tuple: (features, label, file path), where features can be a mel-spectrogram or waveform.
        """
        # Get the file path
        fold = self.annotations.iloc[idx, 5]
        file_name = self.annotations.iloc[idx, 0]
        audio_file_path = os.path.join(self.root_dir, f"fold{fold}", file_name)
        
        # Load the audio file
        audio, sample_rate = librosa.load(audio_file_path, sr=self.sr)
        
        # Ensure all audio has the same length (4 seconds)
        if len(audio) < self.target_length:
            padding = self.target_length - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        elif len(audio) > self.target_length:
            audio = audio[:self.target_length]
        
        # Get the label
        label = int(self.annotations.iloc[idx, 6])
        label = torch.tensor(label, dtype=torch.long)

        # Depending on the mode, return the appropriate features
        if self.mode == 'train' or self.mode == "evaluate":
            features = extract_mel_spectrogram(audio, sample_rate)
            if self.transform:
                features = self.transform(features)
        elif self.mode == 'attack':
            features = audio
        elif self.mode == 'AudioCLIP':
            # Normalize waveform between -1 and 1
            audio = audio / np.max(np.abs(audio))
            audio = torch.tensor(audio, dtype=torch.float32)
            features = audio
            
        return features, label, audio_file_path

def get_data_loaders(annotations_file, root_dir, train_folds, val_folds, test_folds, batch_size=32, transform=None, mode="train"):
    # Create datasets for training, validation, and testing
    train_dataset = UrbanSound8KDataset(annotations_file, root_dir, train_folds, mode= mode, transform=transform)
    val_dataset = UrbanSound8KDataset(annotations_file, root_dir, val_folds, mode=mode, transform=transform)
    test_dataset = UrbanSound8KDataset(annotations_file, root_dir, test_folds, mode=mode, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
