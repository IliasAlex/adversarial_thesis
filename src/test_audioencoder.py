import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.models import Autoencoder, UNet
from datasets.datasets import get_data_loaders

def save_spectrogram(spectrogram, save_path):
    """
    Save a spectrogram as an image.

    Args:
        spectrogram (torch.Tensor): The spectrogram to save (2D array).
        save_path (str): The file path to save the image.
    """
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_original_and_reconstructed(model, test_loader, output_dir, device):
    """
    Perform inference with the trained autoencoder and save original and reconstructed spectrograms.

    Args:
        config (dict): Configuration dictionary.
        model (torch.nn.Module): Trained autoencoder.
        test_loader (torch.utils.data.DataLoader): Test data loader.
        output_dir (str): Directory to save the spectrograms.
        device (torch.device): Device to run inference on.
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for i, (spectrograms, _, _) in enumerate(tqdm(test_loader, desc="Saving spectrograms")):
            if spectrograms.dim() == 3:  # Ensure `unsqueeze(1)` is only applied if there is no channel dimension
                spectrograms = spectrograms.unsqueeze(1)

            spectrograms = spectrograms.to(device)

            # Reconstruct the spectrograms
            reconstructed = model(spectrograms)

            for j in range(spectrograms.size(0)):
                original = spectrograms[j].squeeze(0).cpu().numpy()  # Remove batch and channel dimensions
                reconstructed_spectrogram = reconstructed[j].squeeze(0).cpu().numpy()

                # Define save paths
                original_save_path = os.path.join(output_dir, f"original_{i * test_loader.batch_size + j}.png")
                reconstructed_save_path = os.path.join(output_dir, f"reconstructed_{i * test_loader.batch_size + j}.png")

                # Save original and reconstructed spectrograms
                save_spectrogram(original, original_save_path)
                save_spectrogram(reconstructed_spectrogram, reconstructed_save_path)


# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = "saved_spectrograms"
train_loader, val_loader, test_loader = get_data_loaders(
    "/data/urbansound8k/UrbanSound8K.csv", "/data/urbansound8k", [1,2,3,4,5,6,7], [8], [9], batch_size=32, mode='train'
)

# Load the trained model
model = UNet().to(device)
model.load_state_dict(torch.load("models/best_autoencoder_model.pth"))
save_original_and_reconstructed(model, test_loader, output_dir, device)
