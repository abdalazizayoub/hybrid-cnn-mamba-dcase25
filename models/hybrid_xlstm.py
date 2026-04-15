import torch
import torch.nn as nn
import sys
import os

# Force the stable PyTorch backend for sLSTM to avoid CUDA kernel crashes
os.environ["SLSTM_BACKEND"] = "vanilla"

try:
    from xlstm import (
        xLSTMBlockStack, 
        xLSTMBlockStackConfig, 
        mLSTMBlockConfig, 
        mLSTMLayerConfig, 
        sLSTMBlockConfig,
        sLSTMLayerConfig
    )
except ImportError as e:
    raise ImportError(f"Could not import xLSTM. Ensure the 'xlstm' package is installed. Error: {e}")

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel-wise attention."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.act = nn.ReLU()
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        y = x.mean(dim=(2, 3), keepdim=True)
        y = self.fc1(y).relu()
        y = self.fc2(y).sigmoid()
        return x * y

class ConvBlock(nn.Module):
    """Standard CNN block with Batchnorm, GELU, and SE attention."""
    def __init__(self, in_channels, out_channels, stride=(2, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        self.se = SEBlock(out_channels) 

    def forward(self, x):
        return self.se(self.act(self.bn(self.conv(x))))

class HybridCNNxLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config['embed_dim']
        self.depth = config['depth']
        self.slstm_at = config.get('slstm_at', [1]) # List of layers that are sLSTM

        # 1. CNN Backbone: Aggressive frequency downsampling
        self.cnn = nn.Sequential(
            ConvBlock(1, 8, stride=(2, 1)),   
            ConvBlock(8, 16, stride=(2, 1)),  
            ConvBlock(16, 16, stride=(2, 1)), 
            ConvBlock(16, 32, stride=(2, 1)), 
        )
        
        # Calculate resulting dimensions for the bridge
        with torch.no_grad():
            dummy_x = torch.zeros(1, 1, config['n_mels'], config['target_length'])
            dummy_out = self.cnn(dummy_x)
            _, c_out, f_out, t_out = dummy_out.shape
            
        self.bridge_input_dim = c_out * t_out

        # 2. Bridge: Project CNN features to xLSTM embedding space
        self.bridge_proj = nn.Sequential(
            nn.Linear(self.bridge_input_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Dropout(p=0.2) # Defense against co-adaptation
        )
        
        # 3. xLSTM Brain: Configured for Edge Stability
        xlstm_config = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(conv1d_kernel_size=4) 
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(backend='vanilla') # Stable PyTorch math
            ),
            context_length=f_out,  # Sequence length is the remaining frequency bins
            num_blocks=self.depth, 
            embedding_dim=self.embed_dim,
            slstm_at=self.slstm_at 
        )
        
        self.xlstm = xLSTMBlockStack(xlstm_config)
        self.final_norm = nn.LayerNorm(self.embed_dim)
        
        # 4. Classifier: Pool frequency context and classify
        self.dropout = nn.Dropout(p=0.3) # Final defense against overfitting
        self.classifier = nn.Linear(self.embed_dim, config['n_classes'])

    def forward(self, x):
        # x shape: (B, 1, Mel, Time)
        x = self.cnn(x)  
        
        # Reshape for Frequency Scanning: (B, F_bins, Context_Vector)
        x = x.permute(0, 2, 1, 3) 
        B, F, C, T = x.shape
        x = x.reshape(B, F, C * T) 
        
        x = self.bridge_proj(x) 
        x = self.xlstm(x)
        x = self.final_norm(x)
        
        # Global Average Pooling over the frequency axis
        x = x.mean(dim=1) 
        
        x = self.dropout(x) 
        return self.classifier(x)

def get_model(n_classes, n_mels, target_length, embed_dim, depth, slstm_at=[1], **kwargs):
    config = {
        'n_classes': n_classes,
        'n_mels': n_mels,
        'target_length': target_length,
        'embed_dim': embed_dim,
        'depth': depth,
        'slstm_at': slstm_at
    }
    return HybridCNNxLSTM(config)