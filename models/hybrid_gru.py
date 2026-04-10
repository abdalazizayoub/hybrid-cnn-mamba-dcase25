import torch
import torch.nn as nn
import sys
import os

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


class HybridCNNRNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.embed_dim = config['embed_dim']
        self.depth = config['depth']

        # ==========================================
        # 1. The CNN "Ears" (Apples-to-Apples Baseline)
        # All layers use stride (2,1) to match the original Mamba baseline.
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
        # 2. The Bridge (CNN -> RNN Translation)
        # ==========================================
        self.bridge_proj = nn.Sequential(
            nn.Linear(self.bridge_input_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU()
        )
        
        # ==========================================
        # 3. The GRU "Brain" (Replaces Mamba)
        # ==========================================
        # batch_first=True ensures it expects [Batch, Sequence, Features]
        # bidirectional=False keeps the parameter count low for Edge AI comparison
        self.gru = nn.GRU(
            input_size=self.embed_dim,
            hidden_size=self.embed_dim,
            num_layers=self.depth,
            batch_first=True,
            bidirectional=False 
        )
        
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
        x = self.cnn(x)  
        
        # 2. SNTL-NTU FREQUENCY SCANNING TRICK
        x = x.permute(0, 2, 1, 3) 
        B, F, C, T = x.shape
        
        x = x.reshape(B, F, C * T) 
        x = self.bridge_proj(x) 
        
        # ==========================================
        # 3. GRU Sequence Modeling
        # ==========================================
        # The GRU returns a tuple: (output_sequence, final_hidden_state)
        # We only want the output_sequence `x` to pass into our pooling layer.
        x, _ = self.gru(x)
                
        x = self.final_norm(x)
        
        # 4. Global Average Pooling over the Frequency Sequence
        x = x.mean(dim=1) 
        
        # 5. Classification
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
    return HybridCNNRNN(config)