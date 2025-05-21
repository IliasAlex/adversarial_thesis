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

def evaluate_with_predictions(model_name, model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for waveforms, labels in data_loader:
            spectrograms = torch.stack([extract_mel_spectrogram(waveform.numpy()) for waveform in waveforms])
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            waveforms.to(device)
            if model_name == 'Baseline' or model_name == 'BaselineAvgPooling':
                outputs = model(spectrograms)
            elif model_name == "AudioCLIP":
                outputs = model(waveforms.to(device))
            elif model_name == 'Passt':
                data = model.mel(waveforms.to(device))
                data = data.unsqueeze(1)
                outputs = model.net(data)[0]
            else:
                raise "Incorrect model name"                

            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    loss = running_loss / total
    accuracy = correct / total
    return loss, accuracy

def add_normalized_noise(y: np.ndarray, y_noise: np.ndarray, SNR: float) -> np.ndarray:
    """Apply the background noise y_noise to y with a given SNR
    
    Args:
        y (np.ndarray): The original signal
        y_noise (np.ndarray): The noisy signal
        SNR (float): Signal to Noise ratio (in dB)
        
    Returns:
        np.ndarray: The original signal with the noise added.
    """
    if y.size < y_noise.size:
        y_noise = y_noise[:y.size]
    else:
        y_noise = np.resize(y_noise, y.shape)
    snr = 10**(SNR / 10)
    E_y, E_n = np.sum(y**2), np.sum(y_noise**2)
    scale_factor = np.sqrt((E_y / E_n) * (1 / snr))
    z = y + scale_factor * y_noise

    return {"adversary": z / z.max(), "clean_audio": y / z.max(), "noise": (z - y)/z.max()}

from torchaudio.transforms import MelScale
from scipy.signal import get_window

def preprocess_waveform_to_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int = 22050,
    n_fft: int = 1024,
    hop_length: int = 512,
    win_length: int = 1024,
    n_mels: int = 128,
    spec_height: int = 224,
    spec_width: int = 224,
    window: str = "hann",
    f_min: float = 0.0,
    f_max: float = None,
    log10_eps: float = 1e-6,
):
    """
    Converts a raw audio waveform into a log-Mel spectrogram compatible with AudioCLIP.
    """
    batch_size, T = waveform.shape

    # Compute STFT
    win = torch.from_numpy(get_window(window, Nx=win_length, fftbins=True)).to(waveform.device, waveform.dtype)
    spec = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=win,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    
    # Convert to magnitude spectrogram
    magnitude_spec = spec.abs() ** 2  # Power spectrogram

    # Convert to Mel spectrogram
    mel_filter = MelScale(
        n_mels=n_mels,
        sample_rate=sample_rate,
        f_min=f_min,
        f_max=f_max or sample_rate / 2,
        n_stft=n_fft // 2 + 1,
    ).to(waveform.device)
    mel_spec = mel_filter(magnitude_spec)

    # Convert to log scale
    log_mel_spec = torch.log10(mel_spec + log10_eps) * 10.0  # Convert to decibels

    # Resize to target dimensions
    log_mel_spec = log_mel_spec.unsqueeze(1)  # Add channel dimension: [B, 1, F, T]
    log_mel_spec = F.interpolate(
        log_mel_spec, size=(spec_height, spec_width), mode="bilinear", align_corners=False
    )

    return log_mel_spec