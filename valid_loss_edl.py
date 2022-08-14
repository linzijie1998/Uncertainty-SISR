# -*- coding:utf-8 -*-
import os
from tqdm import *
import torch
import numpy as np
from torch.utils.data import DataLoader

from models.swinir import SwinIR
from dataset.div2k import DIV2K
from criterions.deep_evidential_regression_loss import EvidentialLossSumOfSquares

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

valid_ds = DIV2K(r"./data/DIV2K", factor=3, train=False)
valid_dl = DataLoader(valid_ds, batch_size=5, shuffle=True)
criterion = EvidentialLossSumOfSquares(n_task=3, coefficient=0.01)

for epoch in range(81, 97):
    model = SwinIR(upscale=3, in_chans=3, img_size=80, window_size=8,
                   img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                   mlp_ratio=2, upsampler="pixelshuffle", resi_connection="1conv", nig=True)
    model = model.to(device)
    pretrained_model = torch.load(rf"./checkpoints/swinir/edl/epoch{str(epoch)}_DIV2K_s80w8_x3.pth")
    model.load_state_dict(pretrained_model)
    model.eval()

    loop = tqdm(enumerate(valid_dl), total=len(valid_dl))

    for step, (lrs, hrs) in loop:
        lrs = lrs.to(device)
        hrs = hrs.to(device)
        srs = model(lrs)
        loss = criterion(srs, hrs)

        loop.set_description(f'Epoch [{epoch}/80]')
        loop.set_postfix(nll_loss=round(criterion.nll_loss[-1], 4),
                         reg_loss=round(criterion.reg_loss[-1], 4))

    np.save(rf'./output/valid_loss/epoch_{str(epoch + 1)}_nll_loss.npy', np.array(criterion.nll_loss))
    np.save(rf'./output/valid_loss/epoch_{str(epoch + 1)}_reg_loss.npy', np.array(criterion.reg_loss))
    criterion.nll_loss.clear()
    criterion.reg_loss.clear()
