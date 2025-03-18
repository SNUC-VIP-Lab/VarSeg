################## 1. Download checkpoints and build models
import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed

from models.vqvae import VQVAE
from models.var2 import VAR

MODEL_DEPTH = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# build vae
vae_ckpt = '/home/viplab/SuperRes/VAR/checkpoints/vqvae8192_best.pth'
vae = VQVAE(in_channels=1, vocab_size=8192, z_channels=64, test_mode=False).to(device)
vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)

#build var
var = VAR(vae_local=vae, depth = 16, patch_nums=(1,2,3,4,5,6,8,10,13,16))
out = var.autoregressive_infer_cfg(B = 3)
print(out.shape)

print(f'prepare finished.')