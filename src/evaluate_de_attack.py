import torch
import torch.nn as nn
import numpy as np
import logging
from tqdm.auto import tqdm
import csv
from torch.utils.data import DataLoader, TensorDataset
import random
from datasets.datasets import get_data_loaders, get_esc50_data_loaders
from models.models import BaselineCNN, AudioCLIPWithHead, BaselineCNN2, AudioCLIPWithHead3
from attacks.de_attack import DEAttack
from utils.utils import toUrbanClass, calculate_snr, extract_mel_spectrogram, evaluate_with_predictions
import os
import soundfile as sf
from hear21passt.models.preprocess import AugmentMelSTFT
from hear21passt.base import get_basic_model, get_model_passt


def setup_logging():
    logging.basicConfig(
        filename='de_attack_Passt_AE_ESC.log',
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logging.getLogger().addHandler(logging.StreamHandler())


def load_model(config, device):
    if config['model_name'] == 'BaselineAvgPooling':
        model = BaselineCNN2(num_classes=5)
    elif config['model_name'] == 'Baseline':
        model = BaselineCNN(num_classes=5)
    elif config['model_name'] == 'AudioCLIP':
        model = AudioCLIPWithHead3(pretrained=config['pretrained_audioclip'], num_classes=5, device=device)
    elif config['model_name'] == 'Passt':
        model = get_basic_model(mode="logits")
        model.net = get_model_passt("stfthop160", input_tdim=2000)

        model.mel = AugmentMelSTFT(
            n_mels=128, sr=22050, win_length=800, hopsize=160, n_fft=1024, 
            freqm=0,  # Disable Frequency Masking
            timem=0,  # Disable Time Masking
            htk=False, fmin=0.0, fmax=None, norm=1, fmin_aug_range=10,
            fmax_aug_range=2000
        )

        # Modify the final classification head for ESC-50 (5 classes)
        num_classes = 5
        model.net.head = nn.Sequential(
            nn.LayerNorm(768),  # Keep LayerNorm
            nn.Linear(768, num_classes)  # Change from 527 → 5 classes
        )

        model.net.head_dist = None  # Alternative: nn.Identity()
        
    if config['model_path'] is not None:
        logging.info(f"Loading pre-trained model from {config['model_path']}")
        model.load_state_dict(torch.load(config['model_path'], map_location=device))
        model.eval()

    return model


def create_balanced_subset(test_loader, model_name, model, device, sample_size=50):
    model.eval()
    selected_data = []
    selected_labels = []
    selected_file_paths = []
    if model_name == "Baseline" or model_name == "BaselineAvgPooling":
        correct_indices_per_class = {label: [] for label in range(model.fc2.out_features)}
    elif model_name == "AudioCLIP":
        correct_indices_per_class = {label: [] for label in range(model.classification_head[-1].out_features)}
    elif model_name == "Passt":
        correct_indices_per_class = {label: [] for label in range(model.net.head[-1].out_features)}

    with torch.no_grad():
        for waveforms, labels, file_paths in test_loader:
            batch_spectrograms = torch.stack([extract_mel_spectrogram(waveform.numpy()) for waveform in waveforms])

            if model_name == "Baseline" or model_name == 'BaselineAvgPooling':
                outputs = model(batch_spectrograms.to(device))
            elif model_name == "AudioCLIP":
                outputs = model(waveforms.to(device))
            elif model_name == "Passt":
                data = model.mel(waveforms.to(device))
                data = data.unsqueeze(1)
                outputs = model.net(data)[0]

            _, predicted = outputs.max(1)

            for i in range(len(waveforms)):
                if predicted[i].item() == labels[i].item():
                    correct_indices_per_class[labels[i].item()].append((waveforms[i], labels[i], file_paths[i]))

    for label, samples in correct_indices_per_class.items():
        if len(samples) >= sample_size:
            selected_samples = random.sample(samples, sample_size)
        else:
            selected_samples = samples

        for waveform, label, file_path in selected_samples:
            selected_data.append(waveform)
            selected_labels.append(label)
            selected_file_paths.append(file_path)

    logging.info(f"Selected {len(selected_data)} samples for balanced evaluation with correctly classified samples.")

    dataset = TensorDataset(torch.stack(selected_data), torch.stack(selected_labels))
    return DataLoader(dataset, batch_size=32, shuffle=False), selected_file_paths


def evaluate_de_attack_on_folds(config):
    setup_logging()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    train_folds = list(range(1, 9))
    val_fold = 9
    test_folds = [9, 10]

    logging.info(f"Training with folds {train_folds}, validating with fold {val_fold}, testing with folds {test_folds}")

    # data_csv = f"{config['data_dir']}/UrbanSound8K.csv"
    # _, _, test_loader = get_data_loaders(
    #     data_csv, config['data_dir'], train_folds, [val_fold], test_folds, batch_size=config['batch_size'], mode='attack'
    # )
    
    data_csv = "/data/ESC-50-master/meta/esc50.csv" 
    audio_root_dir = "/data/ESC-50-master/Sorted"  
    
    train_loader, val_loader, test_loader = get_esc50_data_loaders(
        data_csv, audio_root_dir, [1,2,3], [4], [1,2,3,4,5], batch_size=config['batch_size'], mode='attack'
    )

    model = load_model(config, device)
    model.to(device)
    
    sample_size = 1329

    balanced_test_loader, file_paths = create_balanced_subset(test_loader, config['model_name'], model, device, sample_size=sample_size)

    all_data = []
    all_labels = []

    for data, labels in balanced_test_loader:
        all_data.append(data)
        all_labels.append(labels)

    all_data = torch.cat(all_data)
    all_labels = torch.cat(all_labels)

    indices = random.sample(range(len(all_data)), sample_size)
    selected_data = all_data[indices]
    selected_labels = all_labels[indices]
    selected_file_paths = [file_paths[i] for i in indices]

    logging.info(f"Selected {sample_size} random examples from the balanced test set.")

    logging.info("Evaluating on selected clean test data...")
    selected_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(selected_data, selected_labels),
        batch_size=config['batch_size'],
        shuffle=False
    )
    test_loss, test_acc = evaluate_with_predictions(config['model_name'], model, selected_loader, nn.CrossEntropyLoss(), device)
    logging.info(f"Selected Test Accuracy: {test_acc:.4f}")

    for target_snr in config['target_snr']:

        de_attack = DEAttack(
            model=model,
            model_name=config['model_name'],
            max_iter=config['max_iter'],
            population_size=config['pop_size'],
            epsilon=config['epsilon'],
            l2_weight=config['lambda_reg'],
            device=device,
            target_snr=target_snr
        )

        logging.info(f"Generating adversarial examples using DAE attack on selected examples with fixed SNR: {target_snr}")
        attack_metrics = []

        save_folder = config['save_folder']
        os.makedirs(save_folder, exist_ok=True)
        num_saved = 0

        for i in tqdm(range(len(selected_data))):
            original_audio = selected_data[i].cpu().numpy().squeeze()
            current_label = selected_labels[i].item()
            file_path = selected_file_paths[i]

            adv_example, iter_count, final_confidence, clean_audio = de_attack.attack(original_audio, current_label)
            original_audio = clean_audio
            if adv_example is not None:
                if config['model_name'] == 'Baseline' or config['model_name'] == 'BaselineAvgPooling':
                    adv_audio_tensor = extract_mel_spectrogram(adv_example, device=device).unsqueeze(0)
                else:
                    adv_audio_tensor = torch.tensor(adv_example, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    final_outputs = model(adv_audio_tensor)
                    final_prediction = final_outputs.argmax(dim=1).item()
                    final_confidence = torch.softmax(final_outputs, dim=1).max().item()
                success_status = "Success" if final_prediction != current_label else "Failure"
                snr = calculate_snr(original_audio, adv_example - original_audio)
            else:
                success_status = "Failure"
                final_confidence = 0.0
                iter_count = 0
                snr = float('inf')
                final_prediction = current_label

            attack_metrics.append([
                success_status,
                file_path,
                toUrbanClass(current_label),
                toUrbanClass(final_prediction),
                iter_count,
                snr,
                final_confidence
            ])

            if success_status == "Success" and num_saved < config['save_sample']:
                adv_save_path = os.path.join(save_folder, f"adv_{num_saved}_{toUrbanClass(final_prediction)}.wav")
                orig_save_path = os.path.join(save_folder, f"orig_{num_saved}_{toUrbanClass(current_label)}.wav")
                sf.write(orig_save_path, original_audio, samplerate=22050)
                sf.write(adv_save_path, adv_example, samplerate=22050)
                num_saved += 1

        csv_file = f"/home/ilias/projects/adversarial_thesis/results/50_results_{config['epsilon']}.csv"
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "Success/Failure",
                "File Path",
                "Starting Class",
                "Final Class",
                "Iterations",
                "SNR",
                "Final Confidence"
            ])
            writer.writerows(attack_metrics)

        logging.info(f"Attack metrics saved to {csv_file}.")

        # Calculate statistics
        total_attacks = len(attack_metrics)
        successful_attacks = sum(1 for metric in attack_metrics if metric[0] == "Success")
        success_rate = (successful_attacks / total_attacks) * 100 if total_attacks > 0 else 0

        successful_iterations = [metric[4] for metric in attack_metrics if metric[0] == "Success"]
        avg_iterations = np.mean(successful_iterations) if successful_iterations else 0

        successful_snrs = [metric[5] for metric in attack_metrics if metric[0] == "Success" and metric[5] != float('inf')]
        avg_snr = np.mean(successful_snrs) if successful_snrs else float('nan')
        snr_std_dev = np.std(successful_snrs) if successful_snrs else float('nan')

        logging.info(f"Success Rate: {success_rate:.2f}% out of {total_attacks} attacks")
        logging.info(f"Average Iterations of Successful Attacks: {avg_iterations:.2f}")
        logging.info(f"Average SNR of Successful Attacks: {avg_snr:.4f} dB (± {snr_std_dev:.4f})")

