import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.models import Autoencoder, UNet, Autoencoder_AudioCLIP, AudioCLIP, Autoencoder_AudioCLIP_default
from datasets.datasets import get_data_loaders


pretrained_audioclip = '/home/ilias/projects/adversarial_thesis/src/assets/AudioCLIP-pretrained.pt'
audioclip = AudioCLIP(pretrained=pretrained_audioclip, multilabel=False)
audioclip.to('cuda')


def plot_spectrogram(spec: torch.Tensor, spectrogram_path):
    """
    Plot a spectrogram from a given spectrogram tensor.

    Args:
        spec (torch.Tensor): Spectrogram tensor of shape (batch, freq_bins, time_frames, 2)
                             or (freq_bins, time_frames, 2). It contains real and imaginary parts.

    Returns:
        None
    """
    # Remove batch dimension if present
    if spec.dim() == 4:
        spec = spec.squeeze(0)  # Remove batch dimension

    # Convert complex (real, imag) to magnitude
    spec = torch.sqrt(spec[..., 0] ** 2 + spec[..., 1] ** 2)  # Compute magnitude

    # Convert to decibel scale
    spec = 20 * torch.log10(spec + 1e-6)  # Avoid log(0) by adding a small constant

    # Convert to NumPy for plotting
    spec_np = spec.detach().cpu().numpy()

    # Plot the spectrogram
    plt.figure(figsize=(10, 6))
    plt.imshow(spec_np, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label="Amplitude (dB)")
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency Bins")
    plt.title("Spectrogram")
    plt.show()
    plt.savefig(spectrogram_path)
    plt.close()
    
def save_spectrogram(spectrogram, save_path, channel=0, batch_idx=0):
    """
    Save a spectrogram as an image.

    Args:
        spectrogram (torch.Tensor): The spectrogram to save. Shape: [B, C, H, W] or [C, H, W] or [H, W].
        save_path (str): The file path to save the image.
        channel (int): The channel index to visualize (if multi-channel spectrogram).
        batch_idx (int): The batch index to visualize (if batched spectrogram).
    """
    # Handle batched input [B, C, H, W]
    if spectrogram.ndim == 4:
        spectrogram = spectrogram[batch_idx, channel]
    # Handle multi-channel input [C, H, W]
    elif spectrogram.ndim == 3:
        spectrogram = spectrogram[channel]
    # Ensure the spectrogram is now 2D
    if spectrogram.ndim != 2:
        raise ValueError(f"Spectrogram must be 2D after indexing, but got shape {spectrogram.shape}")
    
    # Plot and save the spectrogram
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Amplitude (dB)')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_reconstructed_spectrograms(model, audioclip, test_loader, output_dir, device):
    """
    Save original waveforms, processed spectrograms, and reconstructed spectrograms for visualization.
    """
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():
        for i, (waveforms, _, _) in enumerate(tqdm(test_loader, desc="Saving spectrograms")):
            if waveforms.dim() == 3:  # Ensure correct shape
                waveforms = waveforms.unsqueeze(1)  # Add channel dim if missing

            waveforms = waveforms.to(device)

            # **Convert Waveforms to Spectrograms (Log10 dB scale)**
            x = audioclip.audio._forward_pre_processing(waveforms)
            processed_spectrograms = 2 * (x - x.min()) / (x.max() - x.min() + 1e-6) - 1            # **Reconstruct Spectrograms**
            reconstructed_spectrograms = model(processed_spectrograms)

            for j in range(waveforms.size(0)):  # Iterate over batch
                sample_idx = i * test_loader.batch_size + j

                # **Extract Individual Components**
                original_waveform = waveforms[j].squeeze().cpu().numpy()
                processed_spectrogram = processed_spectrograms[j].squeeze().cpu().numpy()
                reconstructed_spectrogram = reconstructed_spectrograms[j].squeeze().cpu().numpy()

                # **Define save paths**
                original_waveform_path = os.path.join(output_dir, f"original_waveform_{sample_idx}.png")
                processed_spectrogram_path = os.path.join(output_dir, f"processed_spectrogram_{sample_idx}.png")
                reconstructed_spectrogram_path = os.path.join(output_dir, f"reconstructed_spectrogram_{sample_idx}.png")

                # **Plot and Save the Original Waveform**
                plt.figure(figsize=(10, 4))
                plt.plot(original_waveform, color='b', alpha=0.7)
                plt.title(f"Original Waveform {sample_idx}")
                plt.xlabel("Time (samples)")
                plt.ylabel("Amplitude")
                plt.grid()
                plt.savefig(original_waveform_path)
                plt.close()

                #**Plot and Save the Processed Spectrogram**
                plt.figure(figsize=(10, 4))
                plt.imshow(processed_spectrogram.mean(axis=0), aspect='auto', origin='lower', cmap='inferno')
                plt.title(f"Processed Spectrogram {sample_idx}")
                plt.xlabel("Time")
                plt.ylabel("Frequency")
                plt.colorbar(label="Power (dB)")
                plt.savefig(processed_spectrogram_path)
                plt.close()

                #**Plot and Save the Reconstructed Spectrogram**
                plt.figure(figsize=(10, 4))
                plt.imshow(reconstructed_spectrogram.mean(axis=0), aspect='auto', origin='lower', cmap='inferno')
                plt.title(f"Reconstructed Spectrogram {sample_idx}")
                plt.xlabel("Time")
                plt.ylabel("Frequency")
                plt.colorbar(label="Power (dB)")
                plt.savefig(reconstructed_spectrogram_path)
                plt.close()

                # **Print Debugging Info**
                print(f"[Sample {sample_idx}]")
                print(f"  - Waveform Min: {original_waveform.min()}, Max: {original_waveform.max()}")
                print(f"  - Processed Spectrogram Min: {processed_spectrogram.min()}, Max: {processed_spectrogram.max()}")
                print(f"  - Reconstructed Spectrogram Min: {reconstructed_spectrogram.min()}, Max: {reconstructed_spectrogram.max()}")
# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = "saved_spectrograms"
train_loader, val_loader, test_loader = get_data_loaders(
    "/data/urbansound8k/UrbanSound8K.csv", "/data/urbansound8k", [1,2,3,4,5,6,7], [8], [9], batch_size=32, mode='AudioCLIP'
)

# Load the trained model
model = Autoencoder_AudioCLIP_default().to(device)
model.load_state_dict(torch.load("models/best_audioclipautoencoder_model_default.pth"))
save_reconstructed_spectrograms(model, audioclip,test_loader, output_dir, device)