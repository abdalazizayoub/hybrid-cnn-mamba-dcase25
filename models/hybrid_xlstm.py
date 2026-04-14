import torch
import torch.nn as nn
import sys
import os

try:
    from xlstm import (
        xLSTMBlockStack, 
        xLSTMBlockStackConfig, 
        mLSTMBlockConfig, 
        mLSTMLayerConfig, 
        sLSTMBlockConfig
    )
except ImportError as e:
    raise ImportError(f"Could not import xLSTM. Ensure the 'xlstm' package is installed. Error: {e}")


class SEBlock(nn.Module):
    """SNTL-NTU Squeeze-and-Excitation Block to suppress acoustic noise."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.act = nn.ReLU()
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        y = x.mean(dim=(2, 3), keepdim=True)
        y = self.fc1(y)
        y = self.act(y)
        y = self.fc2(y)
        y = self.gate(y)
        return x * y


class ConvBlock(nn.Module):
    """CNN block with Squeeze-and-Excite (SNTL-NTU style)."""
    def __init__(self, in_channels, out_channels, stride=(2, 1)):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=(3, 3), 
            stride=stride,      
            padding=(1, 1), 
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        self.se = SEBlock(out_channels) 

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.se(x) 
        return x


class HybridCNNxLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.embed_dim = config['embed_dim']
        self.depth = config['depth']

        # ==========================================
        # 1. The CNN "Ears" 
        # ==========================================
        self.cnn = nn.Sequential(
            ConvBlock(1, 8, stride=(2, 1)),   
            ConvBlock(8, 16, stride=(2, 1)),  
            ConvBlock(16, 16, stride=(2, 1)), 
            ConvBlock(16, 32, stride=(2, 1)), 
        )
        
        # ==========================================
        # DYNAMIC BRIDGE CALCULATOR
        # ==========================================
        with torch.no_grad():
            dummy_x = torch.zeros(1, 1, config['n_mels'], config['target_length'])
            dummy_out = self.cnn(dummy_x)
            _, c_out, f_out, t_out = dummy_out.shape
            
        self.bridge_input_dim = c_out * t_out

        # ==========================================
        # 2. The Bridge (CNN -> xLSTM Translation)
        # ==========================================
        self.bridge_proj = nn.Sequential(
            nn.Linear(self.bridge_input_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU()
        )
        
        # ==========================================
        # 3. The xLSTM "Brain" 
        # ==========================================
        # FIXED: Using proper mLSTMLayerConfig object instead of a dict
        xlstm_config = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(conv1d_kernel_size=4) 
            ),
            slstm_block=sLSTMBlockConfig(),
            context_length=f_out,  
            num_blocks=self.depth, 
            embedding_dim=self.embed_dim,
            slstm_at=[1] if self.depth > 1 else [] 
        )
        
        self.xlstm = xLSTMBlockStack(xlstm_config)
        self.final_norm = nn.LayerNorm(self.embed_dim)
        
        # ==========================================
        # 4. The Classifier
        # ==========================================
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, config['n_classes'])
        )

    def forward(self, x):
        """
        Input x: [Batch, 1, n_mels (256), target_length (33)]
        """
        x = self.cnn(x)  
        
        # SNTL-NTU FREQUENCY SCANNING TRICK
        x = x.permute(0, 2, 1, 3) 
        B, F, C, T = x.shape
        
        x = x.reshape(B, F, C * T) 
        x = self.bridge_proj(x) 
        
        # xLSTM Sequence Modeling
        x = self.xlstm(x)
                
        x = self.final_norm(x)
        
        # Global Average Pooling
        x = x.mean(dim=1) 
        
        # Classification
        logits = self.classifier(x) 
        
        return logits


def get_model(n_classes, n_mels, target_length, embed_dim, depth, **kwargs):
    config = {
        'n_classes': n_classes,
        'n_mels': n_mels,
        'target_length': target_length,
        'embed_dim': embed_dim,
        'depth': depth,
    }
    return HybridCNNxLSTM(config)