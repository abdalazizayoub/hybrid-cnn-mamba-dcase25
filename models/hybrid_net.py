import torch
import torch.nn as nn
import sys
import os

# --- BULLETPROOF IMPORTS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))

# FIX: Insert Mamba at the very FRONT of the system path (index 0).
# This prevents Python from confusing Mamba's internal code with your local 'models' folder!
mamba_ssm_path = os.path.join(root_dir, 'AUM', 'vim-mamba_ssm')
sys.path.insert(0, mamba_ssm_path)

try:
    from mamba_ssm import Mamba
except ImportError as e:
    raise ImportError(f"Could not import Mamba. Ensure AUM/vim-mamba_ssm is accessible. Error: {e}")


class SEBlock(nn.Module):
    """SNTL-NTU Squeeze-and-Excitation Block to suppress acoustic noise."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.act = nn.ReLU()
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        # Squeeze (Global Average Pool)
        y = x.mean(dim=(2, 3), keepdim=True)
        # Excite
        y = self.fc1(y)
        y = self.act(y)
        y = self.fc2(y)
        y = self.gate(y)
        # Scale
        return x * y


class ConvBlock(nn.Module):
    """CNN block with Squeeze-and-Excite (SNTL-NTU style)."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=(3, 3), 
            stride=(2, 1),      
            padding=(1, 1), 
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        self.se = SEBlock(out_channels) # NEW: Attention mechanism

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.se(x)
        return x


class HybridCNNMamba(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.embed_dim = config['embed_dim']
        self.depth = config['depth']
        self.d_state = config.get('d_state', 16)

        # ==========================================
        # 1. The CNN "Ears" (Texture Extractor)
        # ==========================================
        self.cnn = nn.Sequential(
            ConvBlock(1, 8),   
            ConvBlock(8, 16),  
            ConvBlock(16, 16), 
            ConvBlock(16, 32), 
        )
        
        # ==========================================
        # DYNAMIC BRIDGE CALCULATOR
        # ==========================================
        # Push a dummy tensor through to mathematically perfectly calculate the shapes
        with torch.no_grad():
            dummy_x = torch.zeros(1, 1, config['n_mels'], config['target_length'])
            dummy_out = self.cnn(dummy_x)
            _, c_out, f_out, t_out = dummy_out.shape
            
        # SNTL-NTU TRICK: Sequence Length is Frequency! 
        # feature size Mamba receives at each step is (Channels * Time)
        self.bridge_input_dim = c_out * t_out

        # ==========================================
        # 2. The Bridge (CNN -> Mamba Translation)
        # ==========================================
        self.bridge_proj = nn.Sequential(
            nn.Linear(self.bridge_input_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU()
        )
        
        # ==========================================
        # 3. The Mamba "Brain" (Temporal Processor)
        # ==========================================
        self.mamba_blocks = nn.ModuleList([
            Mamba(
                d_model=self.embed_dim, 
                d_state=self.d_state, 
                d_conv=4, 
                expand=2  
            ) for _ in range(self.depth)
        ])
        
        self.mamba_norms = nn.ModuleList([
            nn.LayerNorm(self.embed_dim) for _ in range(self.depth)
        ])
        
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
        # 1. CNN Feature Extraction
        x = self.cnn(x)  # Shape: [B, C, F, T]
        
        # 2. SNTL-NTU FREQUENCY SCANNING TRICK
        # We permute so Frequency (F) becomes the Sequence Length!
        x = x.permute(0, 2, 1, 3) # Shape becomes: [B, F, C, T]
        B, F, C, T = x.shape
        
        # Flatten Channels and Time into a single feature vector per frequency band
        x = x.reshape(B, F, C * T) # Shape: [B, F, Features]
        
        # Bridge to Mamba Embed Dimension
        x = self.bridge_proj(x) 
        
        # 3. Mamba Sequence Modeling (Scanning bottom-to-top frequencies)
        for mamba, norm in zip(self.mamba_blocks, self.mamba_norms):
            x = x + mamba(norm(x))
            
        x = self.final_norm(x)
        
        # 4. Global Average Pooling over the Frequency Sequence
        x = x.mean(dim=1) 
        
        # 5. Classification
        logits = self.classifier(x) 
        
        return logits


def get_model(n_classes, n_mels, target_length, embed_dim, depth, patch_size, d_state=16):
    config = {
        'n_classes': n_classes,
        'n_mels': n_mels,
        'target_length': target_length,
        'embed_dim': embed_dim,
        'depth': depth,
        'd_state': d_state, 
    }
    return HybridCNNMamba(config)