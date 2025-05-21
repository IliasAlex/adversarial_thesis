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
    
class BaselineCNN2(nn.Module):
    def __init__(self, num_classes=10):
        super(BaselineCNN2, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1) 

        # Fully connected layers
        self.fc1 = nn.Linear(64, 128)  
        self.fc2 = nn.Linear(128, num_classes)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Convolutional layers with ReLU activation and pooling
        x = self.pool(F.relu(self.conv1(x)))  
        x = self.pool(F.relu(self.conv2(x))) 
        x = self.pool(F.relu(self.conv3(x)))  
        
        # Global Average Pooling 
        x = self.global_avg_pool(x)  
        x = torch.flatten(x, 1)  
        
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
        #audio_features = self.audioclip.audio(audio)
        output = self.classification_head(audio_features)
        return output
    
    def __repr__(self):
        return 'AudioCLIP'
    
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
    
class Autoencoder_AudioCLIP(nn.Module):
    def __init__(self):
        super(Autoencoder_AudioCLIP, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
        )


        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(3)
        )

    def forward(self, x):
        # Encoder with skip connections
        e1 = self.encoder[:2](x)  # [B, 32, H/2, W/2]
        e2 = self.encoder[2:4](e1)  # [B, 64, H/4, W/4]
        e3 = self.encoder[4:6](e2)  # [B, 256, H/8, W/8]  (Bottleneck)

        # Decoder with skip connections
        d1 = self.decoder[:3](e3)  # [B, 64, H/4, W/4]
        d1 = d1 + F.interpolate(e2, size=d1.shape[-2:], mode="bilinear", align_corners=False) 
        d2 = self.decoder[3:6](d1)  # [B, 32, H/2, W/2]
        d2 = d2 + F.interpolate(e1, size=d2.shape[-2:], mode="bilinear", align_corners=False) 

        d3 = self.decoder[6:](d2)  # [B, 3, H, W]

        # Ensure final output matches input shape
        if d3.shape[-1] > x.shape[-1]:
            d3 = d3[:, :, :, :x.shape[-1]]
        if d3.shape[-2] > x.shape[-2]:
            d3 = d3[:, :, :x.shape[-2], :]

        return d3

class Autoencoder_AudioCLIP_default(nn.Module):
    def __init__(self):
        super(Autoencoder_AudioCLIP_default, self).__init__()

        # **Encoder (Processing 3-channel RGB Spectrogram)**
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
        )

        # **Bottleneck**
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=3, stride=2, padding=1),  
            nn.ReLU()
        )

        # **Decoder**
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1),  
            nn.Tanh()  # Ensure output remains in [-1,1]
        )

        # **Channel Matching Layers for Skip Connections**
        self.channel_match_x3 = nn.Conv2d(128, 128, kernel_size=1)  # Keeps the same shape
        self.channel_match_x2 = nn.Conv2d(64, 64, kernel_size=1)  # Keeps the same shape
        self.channel_match_x1 = nn.Conv2d(32, 3, kernel_size=1)  # Converts 32 → 3 channels

    def forward(self, x):
        x_input = x.squeeze(1)  # Convert (batch, 1, 3, H, W) → (batch, 3, H, W)

        # **Encoding**
        x1 = self.encoder[0:2](x_input)  
        x2 = self.encoder[2:4](x1)  
        x3 = self.encoder[4:](x2)  

        # **Bottleneck**
        x_b = self.bottleneck(x3)  

        # **Decoding with Skip Connections**
        x_d1 = self.decoder[0:2](x_b)  
        x_d1 = F.interpolate(x_d1, size=x3.shape[2:], mode="bilinear", align_corners=False) + self.channel_match_x3(x3)  # Match channels before adding

        x_d2 = self.decoder[2:4](x_d1)
        x_d2 = F.interpolate(x_d2, size=x2.shape[2:], mode="bilinear", align_corners=False) + self.channel_match_x2(x2)  

        x_d3 = self.decoder[4:](x_d2)
        x_d3 = F.interpolate(x_d3, size=x1.shape[2:], mode="bilinear", align_corners=False) + self.channel_match_x1(x1)  # Convert 32 → 3 before adding

        x_out = x_d3.unsqueeze(1)  # Restore shape (batch, 1, 3, H, W)

        return x_out
    
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

class AudioCLIPWithHead2(nn.Module):
    def __init__(self, pretrained, num_classes=10, device=None):
        super(AudioCLIPWithHead2, self).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.audioclip = AudioCLIP(pretrained=pretrained, multilabel=False)

        # Freeze all parameters except audio-related ones
        for p in self.audioclip.parameters():
            p.requires_grad = False
        for p in self.audioclip.audio.parameters():
            p.requires_grad = True

        self.classification_head = nn.Sequential(
            nn.Linear(1024, 512),  # Reduce first layer to prevent overfitting
            nn.BatchNorm1d(512),
            nn.ReLU(),  # Revert to ReLU for stability
            nn.Dropout(0.3),  # Slightly lower dropout

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),  # Lower dropout further

            nn.Linear(256, num_classes)
        )

    def forward(self, audio):
        # Extract audio features       
        x = self.audioclip.audio._forward_pre_processing(audio)
        x = 2 * (x - x.min()) / (x.max() - x.min() + 1e-6) - 1  # Normalize to [-1,1]

        x = self.audioclip.audio._forward_features(x)
        x = self.audioclip.audio._forward_reduction(x)
        emb = self.audioclip.audio._forward_classifier(x)
        
        output = self.classification_head(emb)
        return output
    
    def __repr__(self):
        return 'AudioCLIP'
    
class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        weights = self.attention(x)  # Compute attention weights
        return x * weights  # Suppress adversarial noise

class AdaptiveFeatureScaling(nn.Module):
    def __init__(self, input_dim):
        super(AdaptiveFeatureScaling, self).__init__()
        self.scale = nn.Parameter(torch.ones(input_dim))  # Learnable scale
        self.shift = nn.Parameter(torch.zeros(input_dim))  # Learnable shift

    def forward(self, x):
        return x * self.scale + self.shift

class SoftPlusSquared(nn.Module):
    def forward(self, x):
        return torch.pow(F.softplus(x, beta=10), 2)  # Squared activation

class LowRankProjection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LowRankProjection, self).__init__()
        self.projection = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        return self.projection(x)

class TemperatureScaling(nn.Module):
    def __init__(self, temperature=2.0):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, logits):
        return logits / self.temperature
    
class AudioCLIPWithHead3(nn.Module):
    def __init__(self, pretrained, num_classes=10, device=None):
        super(AudioCLIPWithHead3, self).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.audioclip = AudioCLIP(pretrained=pretrained, multilabel=False)

        # Freeze all parameters except audio-related ones
        for p in self.audioclip.parameters():
            p.requires_grad = False
        for p in self.audioclip.audio.parameters():
            p.requires_grad = True

        self.classification_head = nn.Sequential(
            LowRankProjection(2048, 1024),  # Adjusted for 2048 input
            AdaptiveFeatureScaling(1024),  # Adjust scaling layer
            SelfAttention(1024),  # Adjust attention layer
            
            nn.Linear(1024, 512),  # Reduce gradually from 1024 → 512
            nn.BatchNorm1d(512),
            SoftPlusSquared(),  # Quadratic non-linearity
            nn.Dropout(0.3),

            nn.Linear(512, 256),  # Reduce further
            nn.BatchNorm1d(256),
            SoftPlusSquared(),
            nn.Dropout(0.2),

            nn.Linear(256, num_classes)  # Final classification layer
        )

    def forward(self, audio):
        x = self.audioclip.audio._forward_pre_processing(audio)
        x = 2 * (x - x.min()) / (x.max() - x.min() + 1e-6) - 1  # -1,1] normalization
        x = self.audioclip.audio._forward_features(x)
        x = self.audioclip.audio._forward_reduction(x)
        output = self.classification_head(x)

        return output

class AdaptiveScalingLayer(nn.Module):
    def forward(self, x):
        return torch.where(x > 0, x * 1.5, x * 1)  # Scale positive more than negative

class PasstAutoencoder(nn.Module):
    def __init__(self, latent_dim=1024, input_shape=(1, 128, 552)):
        super(PasstAutoencoder, self).__init__()

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.encoder5 = nn.Sequential(
            nn.Conv2d(512, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # Compute encoder output shape
        self.encoder_output_shape = self.compute_encoder_output_shape(input_shape)
        self.flattened_size = self.encoder_output_shape[1] * self.encoder_output_shape[2] * self.encoder_output_shape[3]

        # Fully connected latent bottleneck
        self.latent_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, self.flattened_size),
            nn.ReLU(),
            nn.Unflatten(1, (self.encoder_output_shape[1], self.encoder_output_shape[2], self.encoder_output_shape[3]))
        )

        # Decoder (with Skip Connections)
        self.decoder5 = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            #nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2)
        )

        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # Input doubled due to skip connection
            #nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2)
        )

        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            #nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2)
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            #nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2)
        )

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(128, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

    def compute_encoder_output_shape(self, input_shape=(1, 128, 552)):
        x = torch.randn(1, *input_shape)
        x = self.encoder5(self.encoder4(self.encoder3(self.encoder2(self.encoder1(x)))))
        return x.shape

    def forward(self, x):
        # Encoder with skip connections
        e1 = self.encoder1(x)   # (64, H/2, W/2)
        e2 = self.encoder2(e1)  # (128, H/4, W/4)
        e3 = self.encoder3(e2)  # (256, H/8, W/8)
        e4 = self.encoder4(e3)  # (512, H/16, W/16)
        e5 = self.encoder5(e4)  # (1024, H/32, W/32)

        latent = self.latent_fc(e5)  # Fully connected latent space

        # Decoder with skip connections (Fixing Shape Mismatch)
        d5 = self.decoder5(latent)  # (512, H/16, W/16)
        
        e4_resized = F.interpolate(e4, size=d5.shape[-2:], mode="bilinear", align_corners=False)
        d4 = self.decoder4(torch.cat([d5, e4_resized], dim=1))  # (256, H/8, W/8)
        
        e3_resized = F.interpolate(e3, size=d4.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.decoder3(torch.cat([d4, e3_resized], dim=1))  # (128, H/4, W/4)
        
        e2_resized = F.interpolate(e2, size=d3.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.decoder2(torch.cat([d3, e2_resized], dim=1))  # (64, H/2, W/2)
        
        e1_resized = F.interpolate(e1, size=d2.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.decoder1(torch.cat([d2, e1_resized], dim=1))  # (1, H, W)

        # Ensure final output shape matches input shape
        if d1.shape[-1] != x.shape[-1]:  
            d1 = F.interpolate(d1, size=(128, x.shape[-1]), mode="bilinear", align_corners=False)

        return d1
