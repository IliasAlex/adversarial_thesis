import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import logging
from sklearn.metrics import classification_report, mean_squared_error
from datasets.datasets import get_data_loaders
from models.models import BaselineCNN, AudioCLIPWithHead, Autoencoder, UNet
from loops.trainer import train, evaluate
from utils.utils import load_config, set_seed, setup_logging
from attacks.pso_attack import PSOAttack
from evaluate_pso_attack import evaluate_attack_on_folds
from evaluate_dea_attack import evaluate_dae_attack_on_folds
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Adversarial attack thesis on sound event detection model')
    parser.add_argument(
        '-m', '--mode', 
        type=str, 
        required=True, 
        choices=['train', 'attack', 'evaluate'], 
        help='Current mode types: train/attack/evaluate'
    )
    parser.add_argument('attack', type=str, help='Type associated with the mode, e.g., pso for attack.')
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

def evaluate_with_predictions(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    y_true = []
    y_pred = []
       
    with torch.no_grad():
        for data, labels, _ in tqdm(data_loader, desc="Evaluating"):
            data, labels = data.to(device), labels.to(device)
            
            # autoencoder = Autoencoder()
            # autoencoder.load_state_dict(torch.load('/home/ilias/projects/adversarial_thesis/src/models/best_autoencoder_model.pth'))
            # autoencoder.to("cuda")
            # data = autoencoder(data.unsqueeze(0))
            
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

    test_acc, y_true, y_pred = evaluate_with_predictions(model, test_loader, device)

    report = classification_report(y_true, y_pred, target_names=[str(i) for i in range(config['num_classes'])])
    logging.info(f"Evaluation Classification Report:\n{report}")
    logging.info(f"Test Accuracy: {test_acc:.4f}")

def train_autoencoder(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    folds = list(range(1, 11))  # Hardcoded for UrbanSound8K (10 folds)

    overall_test_mse = []
    all_fold_losses = []

    val_fold = 9
    train_folds = [1, 2, 3, 4, 5, 6, 7, 8]
    test_fold = 10

    logging.info(f"Training with folds {train_folds}, validating with fold {val_fold}, testing with fold {test_fold}")

    data_csv = f"{config['data_dir']}/UrbanSound8K.csv"

    # Load data (assuming spectrograms are used)
    train_loader, val_loader, test_loader = get_data_loaders(
        data_csv, config['data_dir'], train_folds, [val_fold], [test_fold],
        batch_size=config['batch_size'], mode='train'
    )

    # Initialize Autoencoder model, criterion, and optimizer
    #model = Autoencoder().to(device)
    model = Autoencoder().to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Track the best model
    best_val_loss = float('inf')  # Initialize with infinity
    best_model_path = "models/best_autoencoder_model.pth"

    # Training loop
    for epoch in range(config['num_epochs']):
        model.train()
        running_loss = 0.0
        for data, _, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['num_epochs']}"):
            if data.dim() == 3:  # Ensure `unsqueeze(1)` is only applied if there is no channel dimension
                data = data.unsqueeze(1)
            
            data = data.to(device)

            # Forward pass
            reconstructed = model(data)
            loss = criterion(reconstructed, data)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        logging.info(f"Epoch {epoch + 1}/{config['num_epochs']}, Training Loss: {avg_train_loss:.4f}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, _, _ in val_loader:
                if data.dim() == 3:  # Ensure `unsqueeze(1)` is only applied if there is no channel dimension
                    data = data.unsqueeze(1)
                
                data = data.to(device)
                reconstructed = model(data)
                loss = criterion(reconstructed, data)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        logging.info(f"Validation Loss for fold {test_fold}: {avg_val_loss:.4f}")

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"New best model saved with validation loss: {best_val_loss:.4f}")

    # Testing loop
    model.load_state_dict(torch.load(best_model_path))  # Load the best model
    test_loss = 0.0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for spectrograms, _, _ in test_loader:
            if spectrograms.dim() == 3:  # Ensure `unsqueeze(1)` is only applied if there is no channel dimension
                spectrograms = spectrograms.unsqueeze(1)
            
            spectrograms = spectrograms.to(device)
            reconstructed = model(spectrograms)
            loss = criterion(reconstructed, spectrograms)
            test_loss += loss.item()

            # Collect true and predicted spectrograms for evaluation
            y_true.append(spectrograms.cpu().numpy())
            y_pred.append(reconstructed.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    overall_test_mse.append(avg_test_loss)

    logging.info(f"Test MSE for fold {test_fold}: {avg_test_loss:.4f}")

    # Average Test MSE across folds
    avg_mse = np.mean(overall_test_mse)
    logging.info(f"Average Test MSE across all folds: {avg_mse:.4f}")

    # Save per-fold losses
    with open("reconstruction_losses.txt", "w") as f:
        for i, mse in enumerate(overall_test_mse):
            f.write(f"Fold {i + 1} Test MSE: {mse:.4f}\n")
        f.write(f"\nAverage Test MSE: {avg_mse:.4f}\n")


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
        if args.attack == "pso":
            evaluate_attack_on_folds(config)
        elif args.attack == "de":
            evaluate_dae_attack_on_folds(config)
if __name__ == "__main__":
    main()