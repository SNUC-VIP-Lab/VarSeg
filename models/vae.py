from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .basic_vae import Decoder, Encoder

class VAE(nn.Module):
    def __init__(
        self, z_channels=32, ch=128, dropout=0.0,
        test_mode=True,
    ):
        super().__init__()
        self.test_mode = test_mode
        self.Cvae = z_channels
        
        ddconfig = dict(
            dropout=dropout, ch=ch, z_channels=z_channels,
            in_channels=3, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2,
            using_sa=True, using_mid_sa=True,
        )
        
        self.encoder = Encoder(double_z=False, **ddconfig)
        self.decoder = Decoder(**ddconfig)
        
        self.latent_conv = torch.nn.Conv2d(self.Cvae, self.Cvae, 3, stride=1, padding=1)
        
        if self.test_mode:
            self.eval()
            [p.requires_grad_(False) for p in self.parameters()]
    
    def forward(self, inp):
        latent = self.latent_conv(self.encoder(inp))
        return self.decoder(latent)
    
    def img_to_latent(self, inp_img_no_grad: torch.Tensor):
        return self.latent_conv(self.encoder(inp_img_no_grad))
    
    def latent_to_img(self, latent: torch.Tensor):
        return self.decoder(latent).clamp_(-1, 1)
    
    def load_state_dict(self, state_dict: Dict[str, Any], strict=True, assign=False):
        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)
