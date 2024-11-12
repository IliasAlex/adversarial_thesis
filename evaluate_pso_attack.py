import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import yaml
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

def create_balanced_subset(data_loader, sample_size=60):
    """
    Create a balanced subset with exactly `sample_size` samples per class.
    Assumes that the dataset provides at least `sample_size` samples for each class.
    """
    all_data = []
    all_labels = []
    for data, labels in data_loader:
        all_data.append(data)
        all_labels.append(labels)

    all_data = torch.cat(all_data)
    all_labels = torch.cat(all_labels)

    # Count the samples per class
    class_indices = {label.item(): (all_labels == label).nonzero(as_tuple=True)[0].tolist() for label in torch.unique(all_labels)}
    balanced_indices = []

    for label, indices in class_indices.items():
        # Select exactly `sample_size` samples per class
        selected_indices = random.sample(indices, sample_size)
        balanced_indices.extend(selected_indices)

    logging.info(f"Selected {len(balanced_indices)} samples for balanced evaluation.")
    dataset = TensorDataset(all_data, all_labels)
    return DataLoader(Subset(dataset, balanced_indices), batch_size=32, shuffle=False)

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

    # Check class distribution
    all_labels = []
    for _, labels in test_loader:
        all_labels.extend(labels.cpu().numpy())
    label_counts = Counter(all_labels)
    logging.info(f"Class distribution in test folds: {label_counts}")

    # Load or train the model
    model = load_or_train_model(config, train_loader, val_loader, device)

    # Create a balanced subset for testing
    balanced_test_loader = create_balanced_subset(test_loader, sample_size=60)
    # Verify class distribution in the balanced subset
    balanced_labels = []
    for _, labels in balanced_test_loader:
        balanced_labels.extend(labels.cpu().numpy())
    balanced_counts = Counter(balanced_labels)
    logging.info(f"Class distribution in balanced test subset: {balanced_counts}")


    # Evaluate the model on the balanced test data
    logging.info("Evaluating on balanced clean test data...")
    test_loss, test_acc, y_true, y_pred = evaluate_with_predictions(model, balanced_test_loader, nn.CrossEntropyLoss(), device)
    logging.info(f"Balanced Test Accuracy: {test_acc:.4f}")

    # Initialize PSO attack
    pso_attack = PSOAttack(
        model,
        max_iter=500,
        swarm_size=150,
        epsilon=0.03,  
        c1=2.0,
        c2=2.0,
        w_max=1.0,
        w_min=0.3,
        patience=50,
        mutation_rate=0.2,
        device='cuda'
    )


    # Initialize counters for the number of adversarial examples per class
    adversarial_counts = {label: 0 for label in range(config['num_classes'])}
    max_samples_per_class = 60

    # Apply PSO attack to the specified test folds
    logging.info(f"Generating adversarial examples using PSO attack on Folds {test_folds}...")
    adversarial_examples = []
    original_labels = []

    for data, labels in balanced_test_loader:
        data, labels = data.to(device), labels.to(device)
        if data.dim() == 3:
            data = data.unsqueeze(1)

        for i in range(len(data)):
            original_audio = data[i].cpu().numpy().squeeze()
            current_label = labels[i].item()

            # Skip this example if we already have enough adversarial samples for this class
            if adversarial_counts[current_label] >= max_samples_per_class:
                continue

            # Select a valid target label different from the current label
            available_classes = list(label_counts.keys())
            available_classes.remove(current_label)
            target_label = random.choice(available_classes)

            adv_example = pso_attack.attack(original_audio, target_label)

            if adv_example is not None:
                adversarial_examples.append(adv_example)
                original_labels.append(current_label)
                adversarial_counts[current_label] += 1

            # Stop generating examples if we have reached the limit for all classes
            if all(count >= max_samples_per_class for count in adversarial_counts.values()):
                break

        if all(count >= max_samples_per_class for count in adversarial_counts.values()):
            break

    logging.info(f"Generated a total of {len(adversarial_examples)} adversarial examples.")


    # Evaluate the model on adversarial examples
    logging.info("Evaluating on adversarial examples...")
    adversarial_loader = create_adversarial_loader(adversarial_examples, original_labels, config['batch_size'], device)
    adv_loss, adv_acc, y_true_adv, y_pred_adv = evaluate_with_predictions(model, adversarial_loader, nn.CrossEntropyLoss(), device)
    logging.info(f"Adversarial Test Accuracy: {adv_acc:.4f}")

    # Generate classification report
    report = classification_report(y_true_adv, y_pred_adv, target_names=[str(i) for i in range(config['num_classes'])], zero_division=0)
    logging.info(f"Adversarial Classification Report:\n{report}")

def create_adversarial_loader(adversarial_examples, original_labels, batch_size, device):
    adversarial_data = torch.tensor(adversarial_examples, dtype=torch.float32).unsqueeze(1).to(device)
    adversarial_labels = torch.tensor(original_labels, dtype=torch.long).to(device)
    dataset = TensorDataset(adversarial_data, adversarial_labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

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