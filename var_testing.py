################## 1. Download checkpoints and build models
import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed

from models import build_vae_var, quant

MODEL_DEPTH = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# build vae
vae_ckpt = '/home/viplab/SuperRes/VAR/checkpoints/vqvae8192_best.pth'
vae, var = build_vae_var(device, depth=MODEL_DEPTH, patch_nums=(1,2,3,4,5,6,8,10,13,16), V=8192, Cvae=64, ch=128, share_quant_resi=4)
vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)

# out = var.autoregressive_infer_cfg(torch.randn(5,3,256,256).to(device))
# print(out.shape)
ov = vae.img_to_idxBl(torch.randn(3, 1, 256, 256).to(device))
oov = vae.quantize.idxBl_to_var_input(ov)
print(oov.shape)
# print(oov[0,:5,:5])

out = var(oov, torch.randn(3, 3, 256, 256).to(device))
print(out.shape)
# print(out[0,:5,:5])

print(f'prepare finished.')