import yaml
import torch
import numpy as np
import random
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

def setup_logging():
    logging.basicConfig(
        filename='training.log',
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logging.getLogger().addHandler(logging.StreamHandler())

def toUrbanClass(id):
    classes = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling", "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]
    return classes[id]

def calculate_snr(signal, noise):
    """
    Calculate the Signal-to-Noise Ratio (SNR) in decibels.

    Args:
        signal (np.ndarray): Original signal.
        noise (np.ndarray): Perturbation (noise).

    Returns:
        float: SNR in decibels.
    """
    # Ensure signal and noise are numpy arrays
    signal = np.array(signal)
    noise = np.array(noise)

    # Calculate power of signal and noise
    power_signal = np.mean(signal ** 2)
    power_noise = np.mean(noise ** 2)

    # Avoid division by zero
    if power_noise == 0:
        return float('inf')

    # Calculate SNR in decibels
    snr = 10 * np.log10(power_signal / power_noise)
    return snr
