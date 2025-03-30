import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(MNISTEncoder, self).__init__()
        # MNIST: 1x28x28
        self.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)  # 32x14x14
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)  # 64x7x7
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)  # 128x4x4
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 4 * 4, latent_dim)
        
    def forward(self, x):
        # Handle both MNIST and batched inputs
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension if missing
            
        # Ensure proper channel dimension for MNIST
        if x.size(1) == 3:  # If RGB input
            x = x.mean(dim=1, keepdim=True)  # Convert to grayscale
            
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.fc(x)
        return x

class MNISTDecoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(MNISTDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)  # 64x8x8
        self.deconv2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)  # 32x16x16
        self.deconv3 = nn.ConvTranspose2d(32, 1, 3, stride=2, padding=2, output_padding=1)  # 1x28x28
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 128, 4, 4)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.tanh(self.deconv3(x))  # Output in range [-1, 1] to match normalization
        if x.size(2) != 28 or x.size(3) != 28:
            x = F.interpolate(x, size=(28, 28), mode='bilinear', align_corners=False)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class CIFAR10Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(CIFAR10Encoder, self).__init__()
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Downsampling blocks with residual connections
        # Block 1: 32x32 -> 16x16
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.skip_conv2 = nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0)
        
        # Block 2: 16x16 -> 8x8
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.skip_conv3 = nn.Conv2d(128, 256, kernel_size=1, stride=2, padding=0)
        
        # Block 3: 8x8 -> 4x4
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.skip_conv4 = nn.Conv2d(256, 512, kernel_size=1, stride=2, padding=0)
        
        # Global pooling and projection to latent space
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, latent_dim)
    
    def forward(self, x):
        # Initial convolution
        x = F.gelu(self.bn1(self.conv1(x)))
        
        # Block 1 with residual connection
        identity = self.skip_conv2(x)
        x = F.gelu(self.bn2(self.conv2(x)))
        x = x + identity  # Residual connection
        
        # Block 2 with residual connection
        identity = self.skip_conv3(x)
        x = F.gelu(self.bn3(self.conv3(x)))
        x = x + identity  # Residual connection
        
        # Block 3 with residual connection
        identity = self.skip_conv4(x)
        x = F.gelu(self.bn4(self.conv4(x)))
        x = x + identity  # Residual connection
        
        # Global pooling and projection
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

class CIFAR10Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(CIFAR10Decoder, self).__init__()
        
        # Project from latent space to initial features
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)
        self.bn_fc = nn.BatchNorm1d(512 * 4 * 4)
        
        # Upsampling blocks with residual connections
        # Block 1: 4x4 -> 8x8
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.skip_upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=1, stride=2, output_padding=1)
        
        # Block 2: 8x8 -> 16x16
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.skip_upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=1, stride=2, output_padding=1)
        
        # Block 3: 16x16 -> 32x32
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.skip_upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=1, stride=2, output_padding=1)
        
        # Final reconstruction layer
        self.conv_out = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Project and reshape from latent vector to initial feature maps
        x = F.gelu(self.bn_fc(self.fc(x)))
        x = x.view(-1, 512, 4, 4)
        
        # Block 1 with residual connection
        identity = self.skip_upconv1(x)
        x = F.gelu(self.bn1(self.upconv1(x)))
        x = x + identity  # Residual connection
        
        # Block 2 with residual connection
        identity = self.skip_upconv2(x)
        x = F.gelu(self.bn2(self.upconv2(x)))
        x = x + identity  # Residual connection
        
        # Block 3 with residual connection
        identity = self.skip_upconv3(x)
        x = F.gelu(self.bn3(self.upconv3(x)))
        x = x + identity  
        
        # Final output layer with tanh for [-1, 1] range
        x = torch.tanh(self.conv_out(x))
        
        return x

class Classifier(nn.Module):
    def __init__(self, latent_dim=128, num_classes=10):
        super(Classifier, self).__init__()
        
        # First layer
        self.fc1 = nn.Linear(latent_dim, 512)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.15)
        
        # Residual block for better feature learning
        self.residual1 = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512)
        )
        
        # Second layer
        self.fc2 = nn.Linear(512, 256)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.15)
        
        # Output layer
        self.fc3 = nn.Linear(256, num_classes)
        nn.init.xavier_uniform_(self.fc3.weight)
        
    def forward(self, x):
        # Apply L2 normalization to input features
        x = F.normalize(x, p=2, dim=1)
        
        # First layer
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        
        # Residual connection
        identity = x
        x = self.residual1(x)
        x = x + identity  # Add residual connection
        x = F.gelu(x)
        
        # Second layer
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        
        # Output layer
        x = self.fc3(x)
        return x

# Full autoencoder models
class MNISTAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(MNISTAutoencoder, self).__init__()
        self.encoder = MNISTEncoder(latent_dim)
        self.decoder = MNISTDecoder(latent_dim)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class CIFAR10Autoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(CIFAR10Autoencoder, self).__init__()
        self.encoder = CIFAR10Encoder(latent_dim)
        self.decoder = CIFAR10Decoder(latent_dim)
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction 