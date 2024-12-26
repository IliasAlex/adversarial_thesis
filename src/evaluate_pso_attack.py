import torch
import torch.nn as nn
import numpy as np
import logging
from tqdm.auto import tqdm
import csv
from torch.utils.data import DataLoader, TensorDataset
import random
from datasets.datasets import get_data_loaders
from models.models import BaselineCNN, AudioCLIPWithHead
from loops.trainer import train
from attacks.pso_attack import PSOAttack
from utils.utils import toUrbanClass, calculate_snr, extract_mel_spectrogram
import os
import soundfile as sf


def setup_logging():
    logging.basicConfig(
        filename='pso_attack_balanced.log',
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logging.getLogger().addHandler(logging.StreamHandler())


def load_model(config, device):
    if config['model_name'] == 'Baseline':
        model = BaselineCNN(num_classes=10)
    elif config['model_name'] == 'AudioCLIP':
        model = AudioCLIPWithHead(pretrained=config['pretrained_audioclip'], num_classes=10, device=device)

    if config['model_path'] is not None:
        logging.info(f"Loading pre-trained model from {config['model_path']}")
        model.load_state_dict(torch.load(config['model_path'], map_location=device))
        model.eval()

    return model


def create_balanced_subset(test_loader, model_name, model, device, sample_size=50):
    """
    Create a balanced subset with correctly classified samples and include file paths.
    This version expects waveform inputs and transforms them to mel-spectrograms before feeding to the model.
    """
    model.eval()
    selected_data = []
    selected_labels = []
    selected_file_paths = []
    if model_name == "Baseline":
        correct_indices_per_class = {label: [] for label in range(model.fc2.out_features)}
    elif model_name == "AudioCLIP":
        correct_indices_per_class = {label: [] for label in range(model.classification_head[-1].out_features)}

    with torch.no_grad():
        for waveforms, labels, file_paths in test_loader:
            batch_spectrograms = torch.stack([extract_mel_spectrogram(waveform.numpy()) for waveform in waveforms])

            if model_name == "Baseline":
                outputs = model(batch_spectrograms.to(device))
            elif model_name == "AudioCLIP":        
                outputs = model(waveforms.to(device))

            _, predicted = outputs.max(1)

            # Store only correctly classified samples
            for i in range(len(waveforms)):
                if predicted[i].item() == labels[i].item():
                    correct_indices_per_class[labels[i].item()].append((waveforms[i], labels[i], file_paths[i]))

    # Select a balanced subset of correctly classified samples
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

    # Return waveforms and labels directly
    dataset = TensorDataset(torch.stack(selected_data), torch.stack(selected_labels))
    return DataLoader(dataset, batch_size=32, shuffle=False), selected_file_paths


def evaluate_attack_on_folds(config):
    setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Train on folds 1-8, validate on fold 9, and test on folds 9 and 10
    train_folds = list(range(1, 9))
    val_fold = 9
    test_folds = [9, 10]

    logging.info(f"Training with folds {train_folds}, validating with fold {val_fold}, testing with folds {test_folds}")

    # Load data loaders
    data_csv = f"{config['data_dir']}/UrbanSound8K.csv"
    _, _, test_loader = get_data_loaders(
        data_csv, config['data_dir'], train_folds, [val_fold], test_folds, batch_size=config['batch_size'], mode='attack'
    )

    # Load or train the model
    model = load_model(config, device)
    model.to(device)

    # Experiment with different epsilon values
    epsilon = config["epsilon"]
    results = {}

    logging.info(f"Experimenting with epsilon = {epsilon}")

    # Create a balanced subset for testing
    balanced_test_loader, file_paths = create_balanced_subset(test_loader, config['model_name'],model, device, sample_size=50)

    # Use 50 random examples from the balanced test loader
    all_data = []
    all_labels = []

    for data, labels in balanced_test_loader:
        all_data.append(data)
        all_labels.append(labels)

    all_data = torch.cat(all_data)
    all_labels = torch.cat(all_labels)

    # Randomly select 50 examples
    indices = random.sample(range(len(all_data)), 50)
    selected_data = all_data[indices]
    selected_labels = all_labels[indices]
    selected_file_paths = [file_paths[i] for i in indices]

    logging.info(f"Selected 50 random examples from the balanced test set.")

    # Evaluate the model on the selected clean test data
    logging.info("Evaluating on selected clean test data...")
    selected_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(selected_data, selected_labels),
        batch_size=config['batch_size'],
        shuffle=False
    )
    test_loss, test_acc = evaluate_with_predictions(config['model_name'], model, selected_loader, nn.CrossEntropyLoss(), device)
    logging.info(f"Selected Test Accuracy: {test_acc:.4f}")

    # Initialize PSO attack
    pso_attack = PSOAttack(
        model=model,
        model_name = config['model_name'],
        max_iter=config['max_iter'],
        swarm_size=config['swarm_size'],
        epsilon=epsilon,  # Set epsilon value for this experiment
        c1=config['c1'],
        c2=config['c2'],
        w_max=config['w_max'],
        w_min=config['w_min'],
        l2_weight= config['l2_weight'],
        device=device
    )

    logging.info("Generating adversarial examples using PSO attack on selected examples...")
    adversarial_examples = []
    # Initialize a list to store attack metrics
    attack_metrics = []

    # Setup saving for adversarial examples
    save_folder = config['save_folder']
    os.makedirs(save_folder, exist_ok=True)
    num_saved = 0

    for i in tqdm(range(len(selected_data))):
        original_audio = selected_data[i].cpu().numpy().squeeze()
        current_label = selected_labels[i].item()
        file_path = selected_file_paths[i]

        # Starting confidence and class
        starting_confidence = pso_attack.fitness_score(original_audio, original_audio, current_label)

        # Perform PSO attack
        adv_example, iterations, final_confidence = pso_attack.attack(original_audio, current_label)

        # Determine success or failure
        success = adv_example is not None
        if success:
            # Determine final class prediction of adversarial example
            adv_audio_tensor = extract_mel_spectrogram(adv_example, device=device)
            with torch.no_grad():
                final_outputs = model(adv_audio_tensor)
                final_prediction = final_outputs.argmax(dim=1).item()
            final_class = final_prediction
            success_status = "Success" if final_class != current_label else "Failure"

            # Compute SNR
            noise = adv_example - original_audio
            snr = calculate_snr(original_audio, noise)
        else:
            success_status = "Failure"
            final_class = current_label
            snr = float('inf')  # Infinite SNR if no perturbation is made

        # Store metrics
        queries = iterations * pso_attack.swarm_size
        attack_metrics.append([
            success_status,
            file_path,
            starting_confidence,
            final_confidence,
            iterations,
            snr,
            toUrbanClass(current_label),
            toUrbanClass(final_class),
            queries
        ])

        # Save adversarial and original samples if success and below the save limit
        if success and num_saved < config['save_sample']:
            adv_save_path = os.path.join(save_folder, f"adv_{num_saved}_{toUrbanClass(final_class)}.wav")
            orig_save_path = os.path.join(save_folder, f"orig_{num_saved}_{toUrbanClass(current_label)}.wav")
            sf.write(orig_save_path, original_audio, samplerate=22050)
            sf.write(adv_save_path, adv_example, samplerate=22050)
            num_saved += 1

    # Save metrics to CSV
    csv_file = f"/home/ilias/projects/adversarial_thesis/results/50_results_{epsilon}.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Success/Failure",
            "File Path",
            "Starting Confidence",
            "Final Confidence",
            "Iterations",
            "SNR",
            "Starting Class",
            "Final Class",
            "Queries"
        ])
        writer.writerows(attack_metrics)

    logging.info(f"Attack metrics saved to {csv_file}.")
    
    # Calculate statistics
    total_attacks = len(attack_metrics)
    successful_attacks = sum(1 for metric in attack_metrics if metric[0] == "Success")
    success_rate = (successful_attacks / total_attacks) * 100 if total_attacks > 0 else 0

    # Average iterations for successful attacks
    successful_iterations = [metric[4] for metric in attack_metrics if metric[0] == "Success"]
    avg_iterations = np.mean(successful_iterations) if successful_iterations else 0

    # Average SNR for successful attacks
    successful_snrs = [metric[5] for metric in attack_metrics if metric[0] == "Success" and metric[5] != float('inf')]
    avg_snr = np.mean(successful_snrs) if successful_snrs else float('nan')
    snr_std_dev = np.std(successful_snrs) if successful_snrs else float('nan')

    # Print statistics
    logging.info(f"Success Rate: {success_rate:.2f}% out of {total_attacks} attacks")
    logging.info(f"Average Iterations of Successful Attacks: {avg_iterations:.2f}")
    logging.info(f"Average SNR of Successful Attacks: {avg_snr:.4f} dB (± {snr_std_dev:.4f})")

    return results


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
            if model_name == 'Baseline':
                outputs = model(spectrograms)
            elif model_name == "AudioCLIP":
                outputs = model(waveforms)
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
