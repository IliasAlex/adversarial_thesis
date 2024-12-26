import yaml
import torch
import numpy as np
import random
import librosa
import torch.nn.functional as F
import logging

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging(filename='training.log'):
    logging.basicConfig(
        filename=filename,
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logging.getLogger().addHandler(logging.StreamHandler())

def toUrbanClass(id):
    classes = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling", "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]
    return classes[id]

                # dog_bark, children_playing, air_conditioner, street_music, gun_shot, siren, engine_idling, jackhammer, drilling, car_horn




def extract_mel_spectrogram(audio, sample_rate=22010, device='cuda'):
        """
        Extracts mel-spectrogram features from a waveform.

        Args:
            audio (numpy.ndarray): The audio waveform.
            sample_rate (int): The sample rate of the audio.

        Returns:
            torch.Tensor: Mel-spectrogram tensor with a fixed size.
        """
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio, sr=sample_rate, n_mels=128, n_fft=4096, hop_length=1024
        )
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Convert to torch tensor
        mel_spectrogram_db = torch.tensor(mel_spectrogram_db, dtype=torch.float32)

        # Ensure the time dimension matches the expected number of time hops (84)
        if mel_spectrogram_db.shape[1] < 84:
            padding = 84 - mel_spectrogram_db.shape[1]
            mel_spectrogram_db = F.pad(mel_spectrogram_db, (0, padding))
        elif mel_spectrogram_db.shape[1] > 84:
            mel_spectrogram_db = mel_spectrogram_db[:, :84]

        mel_tensor = torch.tensor(mel_spectrogram_db, dtype=torch.float32).unsqueeze(0).to(device)

        return mel_tensor


def calculate_snr(signal, noise):
    """
    Calculate the Signal-to-Noise Ratio (SNR) in decibels.

    Args:
        signal (np.ndarray): Original signal.
        noise (np.ndarray): Perturbation (noise).

    Returns:
        float: SNR in decibels.
    """
    # Calculate power of signal and noise
    power_signal = np.mean(signal ** 2)
    power_noise = np.mean(noise ** 2)

    # Avoid division by zero
    if power_noise == 0:
        return float('inf')

    # Calculate SNR in decibels
    snr = 10 * np.log10(power_signal / power_noise)
    return snr