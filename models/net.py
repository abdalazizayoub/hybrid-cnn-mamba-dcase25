import torch
import torch.nn as nn
import sys
import os
import warnings

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import AUM.src.models.mamba_models as mamba_models  # Import Audio Mamba models

class AudioMambaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Audio Mamba configuration
        self.spectrogram_size = (config['n_mels'], config['target_length']) 
        
        # 1. RECTANGULAR PATCHING & OVERLAPPING STRIDES
        if isinstance(config['patch_size'], int):
            self.patch_size = (16, config['patch_size']) 
        elif isinstance(config['patch_size'], str):
            patch_values = [int(v.strip()) for v in config['patch_size'].split(',')]
            self.patch_size = tuple(patch_values) if len(patch_values) > 1 else (16, patch_values[0])
        else:
            self.patch_size = tuple(config['patch_size'])
            
        # 50% Overlap! (This increases MACs but keeps parameters identical)
        self.strides = (self.patch_size[0] // 2, self.patch_size[1] // 2) 
        
        self.embed_dim = config['embed_dim']
        self.depth = config['depth']
        
        # 2. INCREASED WORKING MEMORY
        d_state = config.get('d_state', 24) # Bumped default from 16 to 24 to use free KB!
        ssm_config = {'d_state': d_state}
        
        bimamba_type = 'v2' 
        
        # Initialize Audio Mamba model
        self.aum_model = mamba_models.AudioMamba(
            spectrogram_size=self.spectrogram_size,
            patch_size=self.patch_size,
            strides=self.strides,         # <--- Now overlapping!
            embed_dim=self.embed_dim,
            num_classes=config['n_classes'],
            ssm_cfg=ssm_config,  
            depth=self.depth, 
            imagenet_pretrain=False,
            imagenet_pretrain_path=None,
            aum_pretrain=False,  
            aum_pretrain_path=None, 
            bimamba_type=bimamba_type,
            if_bidirectional=True,
            if_abs_pos_embed=False,
        )
        
        # Remove the final classification head
        if hasattr(self.aum_model, 'head'):
            self.aum_model.head = nn.Identity()
        
        # Add a new classification head for DCASE25
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, config['n_classes'])
        )
        
        # Initialize classifier weights
        self._init_weights(self.classifier)
    
    def _init_weights(self, module):
        """Initialize weights for linear layers"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        features = self.aum_model.forward_features(x)
        logits = self.classifier(features)
        return logits

def get_model(n_classes, n_mels, target_length, embed_dim, depth, patch_size, d_state=24):
    """
    Get Audio Mamba model for DCASE25 Task 1
    """
    config = {
        'n_classes': n_classes,
        'n_mels': n_mels,
        'target_length': target_length,
        'embed_dim': embed_dim,
        'depth': depth,
        'patch_size': patch_size,
        'd_state': d_state, 
    }
    return AudioMambaModel(config)