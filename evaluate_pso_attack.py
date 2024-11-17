import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from tqdm.auto import tqdm
import csv
from sklearn.metrics import classification_report
from collections import Counter
from torch.utils.data import Subset, DataLoader, TensorDataset
import random
from datasets.datasets import get_data_loaders
from models.models import BaselineCNN
from loops.trainer import train
from attacks.pso_attack import PSOAttack

def setup_logging():
    logging.basicConfig(
        filename='pso_attack_balanced.log',
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logging.getLogger().addHandler(logging.StreamHandler())

def load_or_train_model(config, train_loader, val_loader, device):
    model = BaselineCNN(num_classes=config['num_classes']).to(device)

    if config['model_path'] is not None:
        logging.info(f"Loading pre-trained model from {config['model_path']}")
        model.load_state_dict(torch.load(config['model_path'], map_location=device))
        model.eval()
    else:
        logging.info("No pre-trained model specified. Training a new model.")
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

        # Train the model
        model = train(model, train_loader, val_loader, criterion, optimizer, config['num_epochs'], device)
        
        # Save the trained model
        save_path = "trained_baseline_cnn.pth"
        torch.save(model.state_dict(), save_path)
        logging.info(f"Model trained and saved to {save_path}")

    return model

def create_balanced_subset(test_loader, model, device, sample_size=50):
    """
    Create a balanced subset with correctly classified samples and include file paths.
    """
    model.eval()
    selected_data = []
    selected_labels = []
    selected_file_paths = []
    correct_indices_per_class = {label: [] for label in range(model.fc2.out_features)}

    with torch.no_grad():
        for data, labels, file_paths in test_loader:
            data, labels = data.to(device), labels.to(device)
            if data.dim() == 3:
                data = data.unsqueeze(1)

            outputs = model(data)
            _, predicted = outputs.max(1)

            # Store only correctly classified samples
            for i in range(len(data)):
                if predicted[i].item() == labels[i].item():
                    correct_indices_per_class[labels[i].item()].append((data[i], labels[i], file_paths[i]))

    # Select a balanced subset of correctly classified samples
    for label, samples in correct_indices_per_class.items():
        if len(samples) >= sample_size:
            selected_samples = random.sample(samples, sample_size)
        else:
            selected_samples = samples

        for data, label, file_path in selected_samples:
            selected_data.append(data)
            selected_labels.append(label)
            selected_file_paths.append(file_path)

    logging.info(f"Selected {len(selected_data)} samples for balanced evaluation with correctly classified samples.")

    # Create a TensorDataset and DataLoader
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
    train_loader, val_loader, test_loader = get_data_loaders(
        data_csv, config['data_dir'], train_folds, [val_fold], test_folds, batch_size=config['batch_size']
    )

    # Load or train the model
    model = load_or_train_model(config, train_loader, val_loader, device)

    # Create a balanced subset for testing
    balanced_test_loader, file_paths = create_balanced_subset(test_loader, model, device, sample_size=50)

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
    test_loss, test_acc, y_true, y_pred = evaluate_with_predictions(model, selected_loader, nn.CrossEntropyLoss(), device)
    logging.info(f"Selected Test Accuracy: {test_acc:.4f}")

    # Initialize PSO attack
    max_iter = 20  
    swarm_size = 10  
    epsilon = 0.9
    c1 = 0.7
    c2 = 0.7
    w_max = 0.9
    w_min = 0.1

    pso_attack = PSOAttack(
        model=model,
        max_iter=max_iter,
        swarm_size=swarm_size,
        epsilon=epsilon,
        c1=c1,
        c2=c2,
        w_max=w_max,
        w_min=w_min,
        device=device
    )

    # Detailed logging of the PSO attack hyperparameters
    logging.info(
        f"Initialized PSO attack with hyperparameters:\n"
        f"\tmax_iter = {max_iter}\n"
        f"\tswarm_size = {swarm_size}\n"
        f"\tepsilon = {epsilon}\n"
        f"\tc1 = {c1}\n"
        f"\tc2 = {c2}\n"
        f"\tw_max = {w_max}\n"
        f"\tw_min = {w_min}"
    )

    # Initialize a list to store attack metrics
    attack_metrics = []

    # Apply PSO attack on the selected examples
    logging.info("Generating adversarial examples using PSO attack on selected examples...")
    adversarial_examples = []
    original_labels = []

    for i in tqdm(range(len(selected_data))):
        original_audio = selected_data[i].cpu().numpy().squeeze()
        current_label = selected_labels[i].item()
        file_path = selected_file_paths[i]

        # Starting confidence and class
        starting_confidence = pso_attack.fitness_score(original_audio, current_label)
        starting_class = current_label

        # Perform PSO attack without specifying a target label (non-targeted)
        adv_example, iterations, final_confidence = pso_attack.attack(original_audio, current_label)

        # Determine success or failure
        success = adv_example is not None
        if success:
            # Check the final prediction of the adversarial example
            adv_audio_tensor = torch.tensor(adv_example, dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                final_outputs = model(adv_audio_tensor)
                final_prediction = final_outputs.argmax(dim=1).item()
            final_class = final_prediction
        else:
            final_class = current_label

        # Determine if the attack was successful
        attack_success = success and (final_class != current_label)

        # Compute SNR (Signal-to-Noise Ratio)
        if attack_success:
            noise = adv_example - original_audio
            snr = 10 * np.log10(np.sum(original_audio ** 2) / np.sum(noise ** 2))
        else:
            snr = float('inf')  # Infinite SNR if no perturbation is made

        # Store metrics, including the file path
        attack_metrics.append([
            "Success" if attack_success else "Failure",
            file_path,
            starting_confidence,
            final_confidence,
            iterations,
            snr,
            starting_class,
            final_class,
            iterations * pso_attack.swarm_size  # Queries = iterations * swarm size
        ])

        # Store the adversarial example if the attack was successful
        if attack_success:
            adversarial_examples.append(adv_example)
            original_labels.append(current_label)

    logging.info(f"Generated a total of {len(adversarial_examples)} adversarial examples.")

    # Write attack metrics to CSV
    csv_file = f"50_results_{epsilon}.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Success/Failure", "File Path", "Starting Confidence", "Final Confidence",
                         "Iterations", "SNR", "Starting Class", "Final Class", "Queries"])
        writer.writerows(attack_metrics)

    logging.info(f"Attack metrics saved to {csv_file}.")

    total_attacks = len(attack_metrics)
    successful_attacks = sum(1 for m in attack_metrics if m[0] == "Success")
    avg_snr = np.mean([m[5] for m in attack_metrics if m[0] == "Success"])
    avg_iterations = np.mean([m[4] for m in attack_metrics if m[0] == "Success"])

    success_rate = (successful_attacks / total_attacks) * 100 if total_attacks > 0 else 0

    logging.info(f"Total Attacks: {total_attacks}")
    logging.info(f"Successful Attacks: {successful_attacks}")
    logging.info(f"Success Rate: {success_rate:.2f}%")
    logging.info(f"Average SNR of Successful Attacks: {avg_snr:.4f} dB")
    logging.info(f"Average Iterations of Successful Attacks: {avg_iterations:.2f}")

def evaluate_with_predictions(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)

            if data.dim() == 3:
                data = data.unsqueeze(1)

            outputs = model(data)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * data.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    loss = running_loss / total
    accuracy = correct / total
    return loss, accuracy, y_true, y_pred