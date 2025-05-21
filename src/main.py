import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import argparse
import logging
import os
from sklearn.metrics import classification_report, mean_squared_error
from datasets.datasets import get_data_loaders, get_esc50_data_loaders
from models.models import BaselineCNN, AudioCLIPWithHead, Autoencoder, AudioCLIP, Autoencoder_AudioCLIP, BaselineCNN2, Autoencoder_AudioCLIP_default, AudioCLIPWithHead3, PasstAutoencoder
from loops.trainer import train, evaluate
from utils.utils import load_config, set_seed, setup_logging
from attacks.pso_attack import PSOAttack
from evaluate_pso_attack import evaluate_attack_on_folds
from evaluate_de_attack import evaluate_de_attack_on_folds
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from piq import SSIMLoss
from hear21passt.models.preprocess import AugmentMelSTFT
from hear21passt.base import get_basic_model, get_model_passt
import matplotlib.pyplot as plt
from utils.utils import evaluate_with_predictions
from detection import run_detection_experiment

def parse_args():
    parser = argparse.ArgumentParser(description='Adversarial attack thesis on sound event detection model')
    parser.add_argument(
        '-m', '--mode', 
        type=str, 
        required=True, 
        choices=['train', 'attack', 'evaluate', 'detection'], 
        help='Current mode types: train/attack/evaluate'
    )
    parser.add_argument('attack', type=str, help='Type associated with the mode, e.g., pso for attack.')
    parser.add_argument('-c', '--config', type=str, default='config.yaml', help='Path to the configuration file')

    args = parser.parse_args()
    return args

def cross_validate(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    folds = list(range(1, 6))  # Hardcoded for UrbanSound8K (10 folds) 
    # TO DO get list of fold from CONFIG file.
    
    overall_accuracy = []
    all_reports = []

    for test_fold in folds:
        val_fold = (test_fold % 10) + 1
        if val_fold > 5:
            val_fold = test_fold - 1        
        train_folds = [f for f in folds if f != test_fold and f != val_fold]
        val_fold = 4
        test_fold = 5
        train_folds=[1,2,3]
        logging.info(f"Training with folds {train_folds}, validating with fold {val_fold}, testing with fold {test_fold}")

        #data_csv = f"{config['data_dir']}/UrbanSound8K.csv"
        data_csv = "/data/ESC-50-master/meta/esc50.csv" 
        audio_root_dir = "/data/ESC-50-master/audio"  
        
        # if config['model_name'] == "AudioCLIP":
        #     train_loader, val_loader, test_loader = get_data_loaders(
        #         data_csv, config['data_dir'], train_folds, [val_fold], [test_fold], batch_size=config['batch_size'], mode='attack'
        #     )
        #     model = AudioCLIPWithHead(pretrained=config['pretrained_audioclip'], num_classes=10, device=device)
        # else:
        #     train_loader, val_loader, test_loader = get_data_loaders(
        #         data_csv, config['data_dir'], train_folds, [val_fold], [test_fold], batch_size=config['batch_size'], mode='train'
        #     )
        #     model = BaselineCNN2(num_classes=config['num_classes']).to(device)
        
        if config['model_name'] == "AudioCLIP":
            train_loader, val_loader, test_loader = get_esc50_data_loaders(
                data_csv, audio_root_dir, train_folds, [val_fold], [test_fold], batch_size=config['batch_size'], mode='attack'
            )
            model = AudioCLIPWithHead(pretrained=config['pretrained_audioclip'], num_classes=10, device=device)
        else:
            train_loader, val_loader, test_loader = get_esc50_data_loaders(
                data_csv, audio_root_dir, train_folds, [val_fold], [test_fold], batch_size=config['batch_size'], mode='train'
            )
            pretrained_model = BaselineCNN(num_classes=10)
            pretrained_model.load_state_dict(torch.load('/home/ilias/projects/adversarial_thesis/src/models/baseline_cnn.pth', map_location=device))
            
            model = BaselineCNN(num_classes=5).to(device)
            
            model.conv1.load_state_dict(pretrained_model.conv1.state_dict())
            model.conv2.load_state_dict(pretrained_model.conv2.state_dict())
            model.conv3.load_state_dict(pretrained_model.conv3.state_dict())
            
            # Optionally freeze convolutional layers to prevent training them
            for param in model.conv1.parameters():
                param.requires_grad = False
            for param in model.conv2.parameters():
                param.requires_grad = False
            for param in model.conv3.parameters():
                param.requires_grad = False

            
        # Initialize criterion, and optimizer
        criterion = nn.CrossEntropyLoss().to(device)
        
        # optimizer = optim.Adam(model.parameters(), lr=config['learning_rate']) Urbansound8k First train
        
        optimizer = optim.Adam([
            {"params": model.fc1.parameters()},  
            {"params": model.fc2.parameters()}
        ], lr=config['learning_rate'])

        # Train the model
        model = train(model, train_loader, val_loader, criterion, optimizer, config['num_epochs'], device)

        # Evaluate on the test set
        test_acc, y_true, y_pred = evaluate_with_predictions(model, test_loader, device)
        overall_accuracy.append(test_acc)
        
        if test_fold == 5:
            torch.save(model.state_dict(), 'models/baseline_cnn_ESC502.pth')

        report = classification_report(y_true, y_pred, target_names=[str(i) for i in range(config['num_classes'])])
        logging.info(f"Fold {test_fold} Classification Report:\n{report}")
        all_reports.append(report)

        logging.info(f"Fold {test_fold} Test Accuracy: {test_acc:.4f}")

    avg_accuracy = np.mean(overall_accuracy)
    logging.info(f"Average Accuracy across all folds: {avg_accuracy:.4f}")

    with open("classification_reports.txt", "w") as f:
        for i, report in enumerate(all_reports):
            f.write(f"Fold {i + 1} Classification Report:\n{report}\n\n")



# autoencoder = PasstAutoencoder()
# autoencoder.load_state_dict(torch.load('/home/ilias/projects/adversarial_thesis/src/models/Passt_autoencoder.pth'))
# autoencoder.to('cuda:1')

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
            
            # autoencoder = Autoencoder_AudioCLIP()
            # autoencoder.load_state_dict(torch.load('/home/ilias/projects/adversarial_thesis/src/models/best_audioclipautoencoder_model83.pth'))
            # autoencoder.to('cuda')
            
            # autoencoder = Autoencoder_AudioCLIP_default()
            # autoencoder.load_state_dict(torch.load('/home/ilias/projects/adversarial_thesis/src/models/best_audioclipautoencoder_model_default.pth'))
            # autoencoder.to(device)

            if repr(model) == 'AudioCLIP':
                # x = model.audioclip.audio._forward_pre_processing(data)
                # x = 2 * (x - x.min()) / (x.max() - x.min() + 1e-6) - 1  # Normalize to [-1,1]
                # #x = autoencoder(x)
                # x = model.audioclip.audio._forward_features(x)
                # x = model.audioclip.audio._forward_reduction(x)
                # emb = model.audioclip.audio._forward_classifier(x)
                # outputs = model.classification_head(emb)
                outputs = model(data)
            elif model.__class__.__name__ == 'PasstBasicWrapper':
                data = model.mel(data)
                data = data.unsqueeze(1)
                #visualize_and_save_reconstruction(autoencoder, data)
                #data = autoencoder(data)
                outputs = model.net(data)[0]
            else:
                outputs = model(data)
            
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
                
    accuracy = correct / total
    return accuracy, y_true, y_pred

def visualize_and_save_reconstruction(autoencoder, sample_input, sample_idx=0, count = 0,save_dir="/home/ilias/projects/adversarial_thesis/src/saved_spectrograms"):
    os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists

    with torch.no_grad():
        reconstructed = autoencoder(sample_input)  # Get reconstructed batch
        reconstructed = reconstructed.squeeze(1).cpu().numpy()  # Remove channel dim → [batch, height, width]
    
    original = sample_input.squeeze(1).cpu().numpy()  # Remove channel dim → [batch, height, width]

    # Select a single spectrogram from the batch
    original_sample = original[sample_idx]  # Shape: [height, width]
    reconstructed_sample = reconstructed[sample_idx]  # Shape: [height, width]

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original Spectrogram")
    plt.imshow(original_sample, aspect="auto", cmap="magma")
    plt.axis("off")  # Hide axes for a cleaner image

    plt.subplot(1, 2, 2)
    plt.title("Reconstructed Spectrogram")
    plt.imshow(reconstructed_sample, aspect="auto", cmap="magma")
    plt.axis("off")

    # Save the figure
    save_path = os.path.join(save_dir, f"spectrogram_{count}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()  # Close the plot to avoid memory issues

    print(f"Saved spectrogram comparison at: {save_path}")

def evaluate_model(config):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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
    elif config['model_name'] == 'Passt':
        _, _, test_loader = get_data_loaders(
            data_csv, config['data_dir'], [1,2,3,4,5,6,7,8], [9], [10], batch_size=config['batch_size'], mode='AudioCLIP'
        )
    else:
        raise "Invdalid model_name"
    
    data_csv = "/data/ESC-50-master/meta/esc50.csv" 
    audio_root_dir = "/data/ESC-50-master/Sorted"    
    
    train_loader, val_loader, test_loader = get_esc50_data_loaders(
        data_csv, audio_root_dir, [1,2,3], [4], [5], batch_size=config['batch_size'], mode='AudioCLIP'
    )
    
    if config['model_name'] == 'Baseline':
        model = BaselineCNN2(num_classes=5)
        model.load_state_dict(torch.load(config['model_path']))
    elif config['model_name'] == 'AudioCLIP':
        model = AudioCLIPWithHead3(pretrained=config['pretrained_audioclip'], num_classes=5, device=device)
        model.load_state_dict(torch.load(config['model_path']))
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
        num_classes = 5
        model.net.head = nn.Sequential(
            nn.LayerNorm(768),  # Keep LayerNorm
            nn.Linear(768, num_classes)  # Change from 527 → 5 classes
        )
        model.net.head_dist = None  # Alternative: nn.Identity()
        model.load_state_dict(torch.load(config['model_path']))
        model.to(device)
        print(model)    

    model.to(device)
    test_acc, y_true, y_pred = evaluate_with_predictions(model, test_loader, device)

    report = classification_report(y_true, y_pred, target_names=[str(i) for i in range(config['num_classes'])])
    logging.info(f"Evaluation Classification Report:\n{report}")
    logging.info(f"Test Accuracy: {test_acc:.4f}")

def train_autoencoder(config):
    setup_logging(filename='autoencoder.log')
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    folds = list(range(1, 11))  # Hardcoded for UrbanSound8K (10 folds)

    overall_test_mse = []
    all_fold_losses = []

    val_fold = 9
    train_folds = [1, 2, 3, 4, 5, 6, 7, 8]
    test_fold = 10

    logging.info(f"Training with folds {train_folds}, validating with fold {val_fold}, testing with fold {test_fold}")

    data_csv = f"{config['data_dir']}/UrbanSound8K.csv"

    if config['model_name'] == "AudioCLIP":
         # Load data (assuming spectrograms are used)
        train_loader, val_loader, test_loader = get_data_loaders(
            data_csv, config['data_dir'], train_folds, [val_fold], [test_fold],
            batch_size=config['batch_size'], mode='AudioCLIP'
        )       
        model = Autoencoder_AudioCLIP_default().to(device)
    elif config['model_name'] == "Passt":
        # Load data (assuming spectrograms are used)
        train_loader, val_loader, test_loader = get_data_loaders(
            data_csv, config['data_dir'], train_folds, [val_fold], [test_fold],
            batch_size=config['batch_size'], mode='AudioCLIP'
        )     
        model = PasstAutoencoder().to(device)
    else:
        # Load data (assuming spectrograms are used)
        train_loader, val_loader, test_loader = get_data_loaders(
            data_csv, config['data_dir'], train_folds, [val_fold], [test_fold],
            batch_size=config['batch_size'], mode='train'
        )
        model = Autoencoder().to(device)

    # Initialize Autoencoder model, criterion, and optimizer
    #model = Autoencoder().to(device)
    class AsymmetricLoss(nn.Module):
        def forward(self, pred, target):
            loss = F.smooth_l1_loss(pred, target, reduction="none")
            return torch.mean(torch.where(target > 0, loss * 1.5, loss * 1.0))  # More weight on positive values

    #criterion = nn.MSELoss().to(device)
    criterion = AsymmetricLoss().to(device)
    #criterion = nn.L1Loss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    #scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, verbose=True)
    
    # Track the best model
    best_val_loss = float('inf')  # Initialize with infinity
    best_model_path = "models/Passt_autoencoder.pth"

    # pretrained_audioclip = '/home/ilias/projects/adversarial_thesis/src/assets/AudioCLIP-pretrained.pt'
    # audioclip_finetuned = AudioCLIPWithHead2(pretrained=pretrained_audioclip,num_classes=10, device=device)
    # audioclip_finetuned.load_state_dict(torch.load('/home/ilias/projects/AudioCLIP/best_model_znormalization_min_max11.pth',map_location=device))
    # audioclip_finetuned.to(device)
    # audioclip = audioclip_finetuned.audioclip
    
    passt = get_basic_model(mode="logits")
    passt.net = get_model_passt("stfthop160", input_tdim=2000)
    passt.mel = AugmentMelSTFT(
        n_mels=128, sr=22050, win_length=800, hopsize=160, n_fft=1024, 
        freqm=0,  # Disable Frequency Masking
        timem=0,  # Disable Time Masking
        htk=False, fmin=0.0, fmax=None, norm=1, fmin_aug_range=10,
        fmax_aug_range=2000
    )
    num_classes = 10
    passt.net.head = nn.Sequential(
        nn.LayerNorm(768),  # Keep LayerNorm
        nn.Linear(768, num_classes)  # Change from 527 → 5 classes
    )
    passt.net.head_dist = None  # Alternative: nn.Identity()
    passt.load_state_dict(torch.load('/home/ilias/projects/AudioCLIP/Passt-Urban-normalized.pth', map_location=device))
    passt.to(device)
    # Training loop
    for epoch in range(config['num_epochs']):
        model.train()
        running_loss = 0.0
        for data, _, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['num_epochs']}"):
            
            data = data.to(device)
            # **Apply Spectrogram Preprocessing**
            data = passt.mel(data)
            data = data.unsqueeze(1) 
            # **Forward pass**
            reconstructed = model(data)
            
            # **Check Reconstructed Min/Max**
            print(f"Original Spectrogram Min: {data.min().item()}, Max: {data.max().item()}")

            print(f"Reconstructed Spectrogram Min: {reconstructed.min().item()}, Max: {reconstructed.max().item()}")
            # **Compute Loss**
            print(reconstructed.shape)
            print(data.shape)
            
            #loss = criterion(reconstructed, data)
            loss = criterion(reconstructed, data)
            
            # **Backward pass**
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        logging.info(f"Epoch {epoch + 1}/{config['num_epochs']}, Training Loss: {avg_train_loss:.4f}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        counter = 0
        with torch.no_grad():
            for data, _, _ in val_loader:
                if data.dim() == 3:  # Ensure `unsqueeze(1)` is only applied if there is no channel dimension
                    data = data.unsqueeze(1)
                
                data = data.to(device)
                # **Apply Spectrogram Preprocessing**
                data = passt.mel(data)
                data = data.unsqueeze(1) 
                reconstructed = model(data)
                
                #loss = criterion(reconstructed, data)
                loss = criterion(reconstructed, data)
                
                if counter < 5:
                    counter += 1
                    visualize_and_save_reconstruction(model, data, count = counter)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        logging.info(f"Validation Loss for fold {test_fold}: {avg_val_loss:.4f}")
        
        # if epoch % 5 == 0:  
        #     scheduler.step(avg_val_loss)
            
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
            # **Apply Spectrogram Preprocessing**
            data = passt.mel(data)
            data = 2 * (data - data.min()) / (data.max() - data.min() + 1e-8) - 1
            data = data.unsqueeze(1)        
            reconstructed = model(data)
            loss = criterion(reconstructed, data)
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

    # train_autoencoder(config)
    # exit()
    
    if mode == "train":
        setup_logging(filename='baseline-avg-ESC-50.log')
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
            evaluate_de_attack_on_folds(config)
    elif mode == "detection":
        run_detection_experiment(config, "cuda")
        
if __name__ == "__main__":
    main()