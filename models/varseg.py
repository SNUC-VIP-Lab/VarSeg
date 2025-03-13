import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

import dist
from models.basic_var import AdaLNBeforeHead, AdaLNSelfAttn
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from models.basic_vae import Encoder, Decoder


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)   # B16C


class VARSEG(nn.Module):
    def __init__(self, depth=16, embed_dim=1024, num_heads=16):
        super().__init__()
        
        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        
        ddconfig = dict(
            dropout=0, ch=128, z_channels=32,
            in_channels=3, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2,
            using_sa=True, using_mid_sa=True,
        )
        
        self.encoder = Encoder(double_z=False, **ddconfig)
        self.decoder = Decoder(**ddconfig)
        
        self.latent_conv = torch.nn.Conv2d(self.Cvae, self.Cvae, 3, stride=1, padding=1)

