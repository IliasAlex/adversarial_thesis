import torch
import torch.nn as nn
import torch.nn.functional as F
from models.audioclip import AudioCLIP

class BaselineCNN(nn.Module):
    '''
    CNN baseline model with three convolutional layers followed by max-pooling, and two fully connected layers.
    The model uses ReLU activations and dropout for regularization. The final layer outputs a prediction for each class.
    
    Parameters:
    - num_classes (int): The number of classes in the classification task. Defaults to 10 (for urbansound8k dataset).
    
    Architecture:
    - 3 Convolutional layers with increasing filter sizes (16, 32, 64)
    - Max Pooling after each convolutional layer to reduce spatial dimensions
    - 2 Fully connected layers: the first with 128 units and the second with 'num_classes' units
    - Dropout layer with a dropout rate of 0.5 after the first fully connected layer
    '''
    def __init__(self, num_classes=10):
        super(BaselineCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 16 * 10, 128) 
        self.fc2 = nn.Linear(128, num_classes)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Convolutional layers with ReLU activation and pooling
        x = self.pool(F.relu(self.conv1(x)))  # Output: [batch_size, 16, 64, 42]
        x = self.pool(F.relu(self.conv2(x)))  # Output: [batch_size, 32, 32, 21]
        x = self.pool(F.relu(self.conv3(x)))  # Output: [batch_size, 64, 16, 10]
        
        # Flatten the output from the convolutional layers
        x = x.view(-1, 64 * 16 * 10)  # Flatten to match the input size of fc1
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# AudioCLIP Model with Classification Head
class AudioCLIPWithHead(nn.Module):
    def __init__(self, pretrained, num_classes=10, device=None):
        super(AudioCLIPWithHead, self).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.audioclip = AudioCLIP(pretrained=pretrained, multilabel=False)

        # Freeze all parameters except audio-related ones
        for p in self.audioclip.parameters():
            p.requires_grad = False
        for p in self.audioclip.audio.parameters():
            p.requires_grad = True

        self.classification_head = nn.Sequential(
            nn.Linear(1024, 512),  
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )


    def forward(self, audio):
        # Extract audio features
        audio_features = self.audioclip.encode_audio(audio=audio)
        
        output = self.classification_head(audio_features)
        return output
    
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # [B, 1, H, W] -> [B, 16, H/2, W/2]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # [B, 16, H/2, W/2] -> [B, 32, H/4, W/4]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # [B, 32, H/4, W/4] -> [B, 64, H/8, W/8]
            nn.ReLU(),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # [B, 64, H/8, W/8] -> [B, 32, H/4, W/4]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # [B, 32, H/4, W/4] -> [B, 16, H/2, W/2]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 16, H/2, W/2] -> [B, 1, H, W+1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        # Explicit cropping to match the input size
        if decoded.shape[-1] > x.shape[-1]:
            decoded = decoded[:, :, :, :x.shape[-1]]
        return decoded
    
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # [B, 1, H, W] -> [B, 16, H/2, W/2]
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # [B, 16, H/2, W/2] -> [B, 32, H/4, W/4]
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # [B, 32, H/4, W/4] -> [B, 64, H/8, W/8]
            nn.ReLU()
        )
        
        # Decoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # [B, 64, H/8, W/8] -> [B, 32, H/4, W/4]
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # [B, 64, H/4, W/4] -> [B, 16, H/2, W/2]
            nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 32, H/2, W/2] -> [B, 1, H, W]
        )

    def forward(self, x):
        # Encoder
        enc1_out = self.enc1(x)  # [B, 16, H/2, W/2]
        enc2_out = self.enc2(enc1_out)  # [B, 32, H/4, W/4]
        enc3_out = self.enc3(enc2_out)  # [B, 64, H/8, W/8]

        # Decoder with skip connections
        dec3_out = self.dec3(enc3_out)  # [B, 32, H/4, W/4]

        # Crop if dimensions don't match
        if dec3_out.size(2) != enc2_out.size(2) or dec3_out.size(3) != enc2_out.size(3):
            dec3_out = dec3_out[:, :, :enc2_out.size(2), :enc2_out.size(3)]

        dec3_out = torch.cat((dec3_out, enc2_out), dim=1)  # Concatenate along channel dimension

        dec2_out = self.dec2(dec3_out)  # [B, 16, H/2, W/2]

        # Crop if dimensions don't match
        if dec2_out.size(2) != enc1_out.size(2) or dec2_out.size(3) != enc1_out.size(3):
            dec2_out = dec2_out[:, :, :enc1_out.size(2), :enc1_out.size(3)]

        dec2_out = torch.cat((dec2_out, enc1_out), dim=1)  # Concatenate along channel dimension

        dec1_out = self.dec1(dec2_out)  # [B, 1, H, W]

        # Final cropping to match input size (if necessary)
        if dec1_out.size(2) != x.size(2) or dec1_out.size(3) != x.size(3):
            dec1_out = dec1_out[:, :, :x.size(2), :x.size(3)]

        return dec1_out
