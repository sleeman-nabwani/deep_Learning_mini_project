import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(MNISTEncoder, self).__init__()
        # MNIST: 1x28x28
        self.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)  
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)  
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)  
        self.bn3 = nn.BatchNorm2d(128)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 4 * 4, latent_dim)
        
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
            
        if x.size(1) == 3: 
            x = x.mean(dim=1, keepdim=True) 
            
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x

class MNISTDecoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(MNISTDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.bn_fc = nn.BatchNorm1d(128 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)  
        self.bn1 = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)  
        self.bn2 = nn.BatchNorm2d(32)
        self.deconv3 = nn.ConvTranspose2d(32, 1, 3, stride=2, padding=2, output_padding=1)  
        
    def forward(self, x):
        x = self.fc(x)
        x = F.gelu(self.bn_fc(x))
        x = x.view(-1, 128, 4, 4)
        x = F.gelu(self.bn1(self.deconv1(x)))
        x = F.gelu(self.bn2(self.deconv2(x)))
        x = torch.tanh(self.deconv3(x))
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # Proper weight initialization from He et al. (2015)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out')

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.leaky_relu(out)
        return out

class ResidualUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(ResidualUpBlock, self).__init__()
        output_padding = 1 if stride == 2 else 0
        self.upconv1 = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=3, 
            stride=stride, padding=1, output_padding=output_padding, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size=1, 
                    stride=stride, output_padding=output_padding, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        
        # Proper initialization
        nn.init.kaiming_normal_(self.upconv1.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out')

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.leaky_relu(self.bn1(self.upconv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.leaky_relu(out)
        return out

class CIFAR10Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(CIFAR10Encoder, self).__init__()
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual blocks - standard ResNet practice
        self.res_block1 = ResidualBlock(64, 128, stride=2)    # 16x16
        self.res_block2 = ResidualBlock(128, 256, stride=2)   # 8x8
        self.res_block3 = ResidualBlock(256, 512, stride=2)   # 4x4
        
        # Global average pooling - research shows this is better than flattening
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final FC layer
        self.fc = nn.Linear(512, latent_dim)
        self.bn_out = nn.BatchNorm1d(latent_dim)
        
        # Dropout for regularization - standard rate
        self.dropout = nn.Dropout(0.2)
        
        # Initialize conv1 with Kaiming init
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out')

    def forward(self, x):
        # Extract features
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        
        # Global average pooling
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Apply dropout before FC layer (research-based placement)
        x = self.dropout(x)
        
        # Final projection
        x = self.fc(x)
        x = self.bn_out(x)
        
        # No duplicate normalization or activation here
        return x

class CIFAR10Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(CIFAR10Decoder, self).__init__()
        # Initial projection
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)
        self.bn_fc = nn.BatchNorm1d(512 * 4 * 4)
        
        # Upsampling blocks
        self.res_up_block1 = ResidualUpBlock(512, 256, stride=2)  # 8x8
        self.res_up_block2 = ResidualUpBlock(256, 128, stride=2)  # 16x16
        self.res_up_block3 = ResidualUpBlock(128, 64, stride=2)   # 32x32
        
        # Output convolution
        self.conv_out = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        
        # Initialize weights
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out')
        nn.init.xavier_uniform_(self.conv_out.weight)  # Better for output layers

    def forward(self, x):
        # Initial FC
        x = self.fc(x)
        x = F.leaky_relu(self.bn_fc(x))
        x = x.view(-1, 512, 4, 4)
        
        # Upsampling
        x = self.res_up_block1(x)
        x = self.res_up_block2(x)
        x = self.res_up_block3(x)
        
        # Output layer - tanh bounds output to [-1, 1] range
        x = torch.tanh(self.conv_out(x))
        return x

class Classifier(nn.Module):
    def __init__(self, input_dim=128, num_classes=10):
        super(Classifier, self).__init__()
        # Simple MLP with single normalization
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.4)  # Higher dropout in classifier
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(128, num_classes)
        
        # Xavier initialization for classification layers
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        # L2 normalization - research shows this improves classification
        x = F.normalize(x, p=2, dim=1)
        
        # First layer
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        # Second layer
        x = F.leaky_relu(self.bn2(self.fc2(x)))
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
        output = self.decoder(latent)
        return output 