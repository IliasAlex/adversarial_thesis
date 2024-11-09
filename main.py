import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import argparse
import yaml
import logging
from sklearn.metrics import classification_report
from datasets.datasets import get_data_loaders
from models.models import BaselineCNN
from loops.trainer import train, evaluate
from utils.utils import load_config, set_seed, setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description='Sound classification with cross-validation')
    parser.add_argument('-c', '--config', type=str, default='config.yaml', help='Path to the configuration file')
    args = parser.parse_args()
    return args

def cross_validate(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    folds = list(range(1, 11))  # Hardcoded for UrbanSound8K (10 folds) 
    # TO DO get list of fold from CONFIG file.
    
    overall_accuracy = []
    all_reports = []

    for test_fold in folds:
        val_fold = (test_fold % 10) + 1
        train_folds = [f for f in folds if f != test_fold and f != val_fold]

        logging.info(f"Training with folds {train_folds}, validating with fold {val_fold}, testing with fold {test_fold}")

        data_csv = f"{config['data_dir']}/UrbanSound8K.csv"
        train_loader, val_loader, test_loader = get_data_loaders(
            data_csv, config['data_dir'], train_folds, [val_fold], [test_fold], batch_size=config['batch_size']
        )

        # Initialize model, criterion, and optimizer
        model = BaselineCNN(num_classes=config['num_classes']).to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

        # Train the model
        model = train(model, train_loader, val_loader, criterion, optimizer, config['num_epochs'], device)

        # Evaluate on the test set
        test_loss, test_acc, y_true, y_pred = evaluate_with_predictions(model, test_loader, criterion, device)
        overall_accuracy.append(test_acc)

        report = classification_report(y_true, y_pred, target_names=[str(i) for i in range(config['num_classes'])])
        logging.info(f"Fold {test_fold} Classification Report:\n{report}")
        all_reports.append(report)

        logging.info(f"Fold {test_fold} Test Accuracy: {test_acc:.4f}")

    avg_accuracy = np.mean(overall_accuracy)
    logging.info(f"Average Accuracy across all folds: {avg_accuracy:.4f}")

    with open("classification_reports.txt", "w") as f:
        for i, report in enumerate(all_reports):
            f.write(f"Fold {i + 1} Classification Report:\n{report}\n\n")

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
                data = data.unsqueeze(1).to(device)

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

def main():
    args = parse_args()
    config = load_config(args.config)

    setup_logging()
    logging.info("Starting cross-validation")
    set_seed(config['manualSeed'])

    cross_validate(config)
    logging.info("Cross-validation completed")

if __name__ == "__main__":
    main()