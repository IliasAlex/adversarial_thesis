import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import logging
from sklearn.metrics import classification_report
from datasets.datasets import get_data_loaders
from models.models import BaselineCNN, AudioCLIPWithHead
from loops.trainer import train, evaluate
from utils.utils import load_config, set_seed, setup_logging
from attacks.pso_attack import PSOAttack
from evaluate_pso_attack import evaluate_attack_on_folds
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Adversarial attack thesis on sound event detection model')
    parser.add_argument('-m', '--mode', type=str, help='Current mode types: train/attack/evaluate')
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
        
        if config['model_name'] == "AudioCLIP":
            train_loader, val_loader, test_loader = get_data_loaders(
                data_csv, config['data_dir'], train_folds, [val_fold], [test_fold], batch_size=config['batch_size'], mode='attack'
            )
        else:
            train_loader, val_loader, test_loader = get_data_loaders(
                data_csv, config['data_dir'], train_folds, [val_fold], [test_fold], batch_size=config['batch_size'], mode='train'
            )
            
        # Initialize model, criterion, and optimizer
        model = BaselineCNN(num_classes=config['num_classes']).to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

        # Train the model
        model = train(model, train_loader, val_loader, criterion, optimizer, config['num_epochs'], device)

        # Evaluate on the test set
        test_acc, y_true, y_pred = evaluate_with_predictions(model, test_loader, device)
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

def evaluate_with_predictions(model, data_loader, device, model_name='Baseline'):
    model.eval()
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data, labels, _ in tqdm(data_loader, desc="Evaluating"):
            data, labels = data.to(device), labels.to(device)
            
            if model_name == "AudioCLIP":
                outputs = model(audio=data)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
            elif model_name == "Baseline":
                outputs = model(data)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                
    accuracy = correct / total
    return accuracy, y_true, y_pred

def evaluate_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    data_csv = f"{config['data_dir']}/UrbanSound8K.csv"
    
    if config['model_name'] == 'Baseline':
        _, _, test_loader = get_data_loaders(
            data_csv, config['data_dir'], [1,2,3,4,5,6,7,8], [9], [10], batch_size=config['batch_size'], mode='evaluate'
        )
    elif config['model_name'] == 'AudioCLIP':
        _, _, test_loader = get_data_loaders(
            data_csv, config['data_dir'], [1,2,3,4,5,6,7,8], [9], [10], batch_size=config['batch_size'], mode='AudioCLIP'
        )
    else:
        raise "Invdalid model_name"
    
    if config['model_name'] == 'Baseline':
        model = BaselineCNN(num_classes=10)
    elif config['model_name'] == 'AudioCLIP':
        model = AudioCLIPWithHead(pretrained=config['pretrained_audioclip'], num_classes=10, device=device)

    model.load_state_dict(torch.load(config['model_path'], map_location=device))
    model.to(device)
    print(model)    

    test_acc, y_true, y_pred = evaluate_with_predictions(model, test_loader, device, config['model_name'])

    report = classification_report(y_true, y_pred, target_names=[str(i) for i in range(config['num_classes'])])
    logging.info(f"Evaluation Classification Report:\n{report}")
    logging.info(f"Test Accuracy: {test_acc:.4f}")


def main():
    args = parse_args()
    config = load_config(args.config)
    mode = args.mode
    set_seed(config['manualSeed'])
    
    if mode == "train":
        setup_logging()
        logging.info("Starting cross-validation")
        cross_validate(config)
        logging.info("Cross-validation completed")
    elif mode == "evaluate":
        setup_logging(filename='evaluating.log')
        logging.info("Starting evaluation")
        
        evaluate_model(config)
        logging.info("Evaluation completed")
    elif mode == "attack":
         evaluate_attack_on_folds(config)

if __name__ == "__main__":
    main()