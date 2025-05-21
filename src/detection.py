import os
import glob
import torch
import logging
import librosa
import torch.nn.functional as F
import numpy as np
from models.models import BaselineCNN, BaselineCNN2, Autoencoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.utils import extract_mel_spectrogram, toUrbanClass

def setup_logging():
    logging.basicConfig(
        filename='detection.log',
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logging.getLogger().addHandler(logging.StreamHandler())
    
def load_model(config, device):
    if config['model_name'] == 'Baseline':
        model = BaselineCNN(num_classes=10)
    elif config['model_name'] == 'BaselineAvgPooling':
        model = BaselineCNN2(num_classes=10)

    if config['model_path'] is not None:
        logging.info(f"Loading pre-trained model from {config['model_path']}")
        model.load_state_dict(torch.load(config['model_path'], map_location=device))
        model.eval()

    return model

def run_detection_experiment(config, device="cuda"):
    """
    Runs the adversarial detection experiment.
    
    Args:
        config (dict): Configuration dictionary containing model details.
        device (str): Device to use for computation ("cpu" or "cuda").
    """
    setup_logging()
    detection_folder = config["detection_folder"]

    # Load model
    model = load_model(config, device)
    model.to(device)
    model.eval()
    
    # Load autoencoder
    autoencoder = Autoencoder()
    autoencoder.load_state_dict(torch.load('/home/ilias/projects/adversarial_thesis/src/models/best_autoencoder_model.pth'))
    autoencoder.to(device)
    autoencoder.eval()

    # Get all adversarial files
    adv_files = sorted(glob.glob(os.path.join(detection_folder, "adv_*.wav")))

    if not adv_files:
        logging.error(f"[ERROR] No adversarial files found in: {detection_folder}")
        return
    
    results = []
    y_true, y_pred = [], []

    for adv_file in adv_files:
        # Load adversarial waveform
        adv_waveform, sr = librosa.load(adv_file, sr=22050)

        # Extract spectrogram and send to model
        adv_spectrogram = extract_mel_spectrogram(adv_waveform).unsqueeze(0)

        # Predict label before autoencoder
        with torch.no_grad():
            outputs = model(adv_spectrogram)  # Get logits
            adv_prediction = outputs.argmax(dim=1).item()

        # Pass through autoencoder
        with torch.no_grad():
            reconstructed_spectrogram = autoencoder(adv_spectrogram)

        # Predict label after autoencoder
        with torch.no_grad():
            outputs = model(reconstructed_spectrogram)  # Get logits
            new_prediction = outputs.argmax(dim=1).item()

        # Compare predictions
        detected = adv_prediction != new_prediction
        results.append({
            "adv_file": adv_file,
            "adv_prediction": toUrbanClass(adv_prediction),
            "new_prediction": toUrbanClass(new_prediction),
            "detected": detected
        })

        # Save true and predicted labels
        y_true.append(1)  # Since we are dealing with adversarial samples, ground truth is 1 (attack)
        y_pred.append(1 if detected else 0)  # 1 if detected, 0 if not

    # Compute detection performance metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Print results
    for res in results:
        logging.info(f"{res['adv_file']}: Adv Label = {res['adv_prediction']}, New Label = {res['new_prediction']} -> Detected: {res['detected']}")

    logging.info(f"Detection Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1-score: {f1:.4f}")

    return results, {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}
