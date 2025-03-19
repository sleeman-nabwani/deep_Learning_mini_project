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

class CIFAR10Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(CIFAR10Encoder, self).__init__()
        # CIFAR10: 3x32x32
        # Deeper architecture with more filters and batch normalization
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)  # 64x32x32
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)  # 64x16x16
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)  # 128x16x16
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)  # 128x8x8
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)  # 256x8x8
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=2, padding=1)  # 256x4x4
        self.bn6 = nn.BatchNorm2d(256)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256 * 4 * 4, latent_dim)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class CIFAR10Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(CIFAR10Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.bn1 = nn.BatchNorm2d(256)
        self.deconv1 = nn.ConvTranspose2d(256, 256, 3, stride=1, padding=1)  # 256x4x4
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)  # 128x8x8
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1)  # 128x8x8
        self.bn4 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)  # 64x16x16
        self.bn5 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1)  # 64x16x16
        self.bn6 = nn.BatchNorm2d(64)
        self.deconv6 = nn.ConvTranspose2d(64, 3, 3, stride=2, padding=1, output_padding=1)  # 3x32x32
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 256, 4, 4)
        x = F.relu(self.bn1(x))
        x = F.relu(self.bn2(self.deconv1(x)))
        x = F.relu(self.bn3(self.deconv2(x)))
        x = F.relu(self.bn4(self.deconv3(x)))
        x = F.relu(self.bn5(self.deconv4(x)))
        x = F.relu(self.bn6(self.deconv5(x)))
        x = torch.tanh(self.deconv6(x))  # Output in range [-1, 1] to match normalization
        return x

class Classifier(nn.Module):
    def __init__(self, latent_dim=128, num_classes=10):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
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
        x = self.encoder(x)
        x = self.decoder(x)
        return x 