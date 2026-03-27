import torch
import torch.nn as nn
import sys
import os
import copy
from typing import Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
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
        y = x.mean(dim=(2, 3), keepdim=True)
        y = self.fc1(y)
        y = self.act(y)
        y = self.fc2(y)
        y = self.gate(y)
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
        self.se = SEBlock(out_channels)

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
        self.d_state = config.get('d_state', 32)

        # ==========================================
        # 1. The CNN "Ears" (Texture Extractor)
        # ==========================================
        self.cnn = nn.Sequential(
            ConvBlock(2, 8),   # <-- COORDCONV UPDATE: Expects 2 channels (Audio + GPS Map)
            ConvBlock(8, 16),  
            ConvBlock(16, 16), 
            ConvBlock(16, 32), 
        )
        
        # ==========================================
        # DYNAMIC BRIDGE CALCULATOR
        # ==========================================
        with torch.no_grad():
            # <-- COORDCONV UPDATE: Dummy tensor must also have 2 channels now
            dummy_x = torch.zeros(1, 2, config['n_mels'], config['target_length'])
            dummy_out = self.cnn(dummy_x)
            _, c_out, f_out, t_out = dummy_out.shape
            
        # SNTL-NTU TRICK: Sequence Length is Frequency! 
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
        B, C, F, T = x.shape
        
        # ==========================================
        # ZERO-COST TRICK: THE COORD-CONV "GPS" MAP
        # ==========================================
        # 1. Create a 1D gradient from -1.0 (bottom) to 1.0 (top)
        y_coords = torch.linspace(-1.0, 1.0, steps=F, dtype=x.dtype, device=x.device)
        
        # 2. Expand it to match the exact shape of your audio batch
        y_grid = y_coords.view(1, 1, F, 1).expand(B, 1, F, T)
        
        # 3. Glue the GPS grid onto the audio as Channel 2
        x = torch.cat([x, y_grid], dim=1)  # Shape becomes [Batch, 2, 256, 33]
        # ==========================================

        # 1. CNN Feature Extraction
        x = self.cnn(x)  # Shape: [B, C, F, T]
        
        # 2. SNTL-NTU FREQUENCY SCANNING TRICK
        x = x.permute(0, 2, 1, 3) # Shape becomes: [B, F, C, T]
        B_new, F_new, C_new, T_new = x.shape
        
        x = x.reshape(B_new, F_new, C_new * T_new) # Shape: [B, F, Features]
        x = self.bridge_proj(x) 
        
        # 3. Mamba Sequence Modeling 
        for mamba, norm in zip(self.mamba_blocks, self.mamba_norms):
            x = x + mamba(norm(x))
            
        x = self.final_norm(x)
        
        # 4. Global Average Pooling over the Frequency Sequence
        x = x.mean(dim=1) 
        
        # 5. Classification
        logits = self.classifier(x) 
        
        return logits

class MultiDeviceModelContainer(nn.Module):
    def __init__(self, base_model: nn.Module, devices: list):
        super().__init__()
        self.base_model = base_model
        self.devices = devices
        self.device_models = nn.ModuleDict({
            device: copy.deepcopy(base_model) for device in devices
        })

    def forward(self, x: torch.Tensor, devices: Tuple[str] = None) -> torch.Tensor:
        if devices is None:
            return self.base_model(x)
        elif len(set(devices)) > 1:
            return self._forward_multi_device(x, devices)
        elif devices[0] in self.device_models:
            # ==========================================
            # ZERO-COST TRICK: LOGIT BLENDING
            # 70% Specialist Hardware Brain, 30% Generalized Brain
            # ==========================================
            specialist_logits = self.get_model_for_device(devices[0])(x)
            base_logits = self.base_model(x)
            return (0.7 * specialist_logits) + (0.3 * base_logits)
        else:
            return self.base_model(x)

    def _forward_multi_device(self, x: torch.Tensor, devices: Tuple[str]) -> torch.Tensor:
        outputs = []
        for i, device in enumerate(devices):
            if device in self.device_models:
                spec_logits = self.device_models[device](x[i].unsqueeze(0))
                base_logits = self.base_model(x[i].unsqueeze(0))
                outputs.append((0.7 * spec_logits) + (0.3 * base_logits))
            else:
                outputs.append(self.base_model(x[i].unsqueeze(0)))
        return torch.cat(outputs)

    def get_model_for_device(self, device_name: str) -> nn.Module:
        if device_name in self.device_models:
            return self.device_models[device_name]
        else:
            return self.base_model

def get_model(**kwargs):
    config = {
        'n_classes': kwargs.get('n_classes', 10),
        'n_mels': kwargs.get('n_mels', 256),
        'target_length': kwargs.get('target_length', 33),
        'embed_dim': kwargs.get('embed_dim', 28),
        'depth': kwargs.get('depth', 2),
        'patch_size': kwargs.get('patch_size', 4),
        'd_state': kwargs.get('d_state', 32),
    }
    
    base_model = HybridCNNMamba(config)
    
    train_devices = kwargs.get('train_devices', None)
    if train_devices is not None:
        return MultiDeviceModelContainer(base_model, train_devices)
        
    return base_model