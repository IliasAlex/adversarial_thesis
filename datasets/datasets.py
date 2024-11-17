import torch
import torch.nn.functional as F
import librosa
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class UrbanSound8KDataset(Dataset):
    def __init__(self, annotations_file, root_dir, folds, transform=None):
        self.annotations = pd.read_csv(annotations_file)
        self.root_dir = root_dir
        self.transform = transform
        self.folds = folds

        # Filter annotations to include only the specified folds
        self.annotations = self.annotations[self.annotations['fold'].isin(self.folds)]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Get the file path
        fold = self.annotations.iloc[idx, 5]
        file_name = self.annotations.iloc[idx, 0]
        audio_file_path = os.path.join(self.root_dir, f"fold{fold}", file_name)
        
        # Load the audio file
        audio, sample_rate = librosa.load(audio_file_path, sr=22050)
        
        # Extract features (mel-spectrogram)
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128, n_fft=4096, hop_length=1024)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Convert to torch tensor
        mel_spectrogram_db = torch.tensor(mel_spectrogram_db, dtype=torch.float32)

        # Ensure the time dimension matches the expected number of time hops (84)
        if mel_spectrogram_db.shape[1] < 84:
            padding = 84 - mel_spectrogram_db.shape[1]
            mel_spectrogram_db = F.pad(mel_spectrogram_db, (0, padding))
        elif mel_spectrogram_db.shape[1] > 84:
            mel_spectrogram_db = mel_spectrogram_db[:, :84]

        # Apply transformation if any (e.g., data augmentation)
        if self.transform:
            mel_spectrogram_db = self.transform(mel_spectrogram_db)

        # Get the label
        label = int(self.annotations.iloc[idx, 6])

        # Convert to torch tensor
        label = torch.tensor(label, dtype=torch.long)

        return mel_spectrogram_db, label, audio_file_path

def get_data_loaders(annotations_file, root_dir, train_folds, val_folds, test_folds, batch_size=32, transform=None):
    # Create datasets for training, validation, and testing
    train_dataset = UrbanSound8KDataset(annotations_file, root_dir, train_folds, transform=transform)
    val_dataset = UrbanSound8KDataset(annotations_file, root_dir, val_folds, transform=transform)
    test_dataset = UrbanSound8KDataset(annotations_file, root_dir, test_folds, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
