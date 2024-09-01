import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import argparse
import yaml
from datasets.datasets import get_data_loaders
from models.models import BaselineCNN
from loops.trainer import train, test


def parse_args():
    parser = argparse.ArgumentParser(description='Sound classification baseline')
    parser.add_argument('-c','--config', type=str, default='config.yaml', help='Path to the configuration file')
    args = parser.parse_args()
    return args

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    
    # Load configuration from YAML file
    config = load_config(args.config)
    
    # Set manual seed
    set_seed(config['manualSeed'])

    # Prepare data loaders
    train_fold = list(map(int, config['train_fold'].split(',')))
    val_fold = list(map(int, config['val_fold'].split(',')))
    test_fold = list(map(int, config['test_fold'].split(',')))
    
    data_csv = f"{config['data_dir']}/UrbanSound8K.csv"
    
    train_loader, val_loader, test_loader = get_data_loaders(
        data_csv, config['data_dir'], train_fold, val_fold, test_fold, batch_size=config['batch_size'])

    # Initialize model, criterion, and optimizer
    model = BaselineCNN(num_classes = config['num_classes'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Train the model
    model = train(model, train_loader, val_loader, criterion, optimizer, config['num_epochs'], device)

    # Test the model
    test_loss, test_acc = test(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()

    #model.to(device)
    #torch.save(model.state_dict(), os.path.join("/home/ilias/projects/adversarial_thesis/models", f'baseline_cnn.pth'))