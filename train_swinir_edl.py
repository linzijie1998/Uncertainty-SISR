# -*- coding: utf-8 -*-
import os
import argparse

import cv2
import numpy as np
from tqdm import *

import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from models.swinir import SwinIR
from dataset.div2k import DIV2K
from criterions.deep_evidential_regression_loss import EvidentialLossSumOfSquares

from utils import image, metric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True)
    parser.add_argument("--gpu_device", default="0")
    parser.add_argument("--seed", default=8888)

    parser.add_argument("--upscale", default=3)
    parser.add_argument("--training_patch_size", default=80)
    parser.add_argument("--window_size", default=8)
    parser.add_argument("--depths", default=[6, 6, 6, 6])
    parser.add_argument("--embed_dim", default=60)
    parser.add_argument("--num_heads", default=[6, 6, 6, 6])
    parser.add_argument("--upsampler", default="pixelshuffle")
    parser.add_argument("--resi_connection", default="1conv")

    parser.add_argument("--coefficient", default=0.01)
    parser.add_argument("--epochs", default=100)
    parser.add_argument("--lr", default=2.e-4)
    parser.add_argument("--step_size", default=10)
    parser.add_argument("--gamma", default=0.8)

    parser.add_argument("--train_dataset", default=r"./data/DIV2K")
    parser.add_argument("--valid_dataset", default=r"./data/Set5")
    parser.add_argument("--batch_size", default=10)

    parser.add_argument("--checkpoint", default=r"./checkpoints/swinir/edl")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)

    # classical image super-resolution model
    model = SwinIR(upscale=args.upscale, in_chans=3, img_size=args.training_patch_size, window_size=args.window_size,
                   img_range=1., depths=args.depths, embed_dim=args.embed_dim, num_heads=args.num_heads,
                   mlp_ratio=2, upsampler=args.upsampler, resi_connection=args.resi_connection, nig=True)

    criterion = EvidentialLossSumOfSquares(n_task=3, coefficient=args.coefficient)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    train_ds = DIV2K(path=args.train_dataset, factor=args.upscale)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model = model.to(device)

    for epoch in range(args.epochs):
        print(f"[{epoch + 1}/{args.epochs}] training model with DIV2K dataset...")

        model.train()

        total_loss = 0.
        total_count = 0.

        loop = tqdm(enumerate(train_dl), total=len(train_dl))

        # for step, (lrs, hrs) in loop:
        for lrs, hrs in train_dl:
            optimizer.zero_grad()
            lrs = lrs.to(device)
            hrs = hrs.to(device)
            srs = model(lrs)
            loss = criterion(srs, hrs)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * lrs.size(0)
            total_count += lrs.size(0)

            # loop.set_description(f'Epoch [{epoch}/{args.epochs}]')
            # loop.set_postfix(nll_loss=round(criterion.nll_loss[-1], 4),
            #                  reg_loss=round(criterion.reg_loss[-1], 4),
            #                  avg_loss=round(total_loss / total_count, 4),
            #                  type=type(criterion.nll_loss[-1]))

            print(f"nll: {criterion.nll_loss[-1]}, reg: {criterion.reg_loss[-1]}")

        scheduler.step()
        np.save(f'epoch_{str(epoch + 1)}_nll_loss.npy', np.array(criterion.nll_loss))
        np.save(f'epoch_{str(epoch + 1)}_reg_loss.npy', np.array(criterion.reg_loss))
        criterion.nll_loss.clear()
        criterion.reg_loss.clear()


        print(f"[{epoch + 1}/{args.epochs}] average Edl loss: {round(total_loss / total_count, 6)}")

        checkpoint_path = os.path.join(
            args.checkpoint,
            f'epoch{epoch + 1}_DIV2K_s{args.training_patch_size}w{args.window_size}_x{args.upscale}.pth')
        torch.save(model.state_dict(), checkpoint_path)


        model.eval()

        print(f"[{epoch + 1}/{args.epochs}] valid model with Set5 dataset...")
        lrs, hrs = image.get_image_pairs(args.valid_dataset, scale=args.upscale)
        total_psnr = 0.
        total_ssim = 0.

        for i in range(len(lrs)):
            lr_tensor = image.png2tensor(lrs[i])

            with torch.no_grad():
                lr_tensor, h, w = image.image_padding(lr_tensor, args.window_size)
                lr_tensor = lr_tensor.to(device)
                sr_tensor = model(lr_tensor)
                sr_tensor = sr_tensor[..., :h * args.upscale, :w * args.upscale]
            gamma, _, _, _ = torch.split(sr_tensor, 3, dim=1)
            sr_img = image.tensor2png(gamma)
            gt_img = cv2.imread(hrs[i], cv2.IMREAD_COLOR).astype(np.uint8)

            psnr = metric.calculate_psnr(sr_img, gt_img, crop_border=args.window_size, test_y_channel=True)
            ssim = metric.calculate_ssim(sr_img, gt_img, crop_border=args.window_size, test_y_channel=True)

            total_psnr += psnr
            total_ssim += ssim

        print(f"[{epoch + 1}/{args.epochs}] "
              f"average psnr: {round(total_psnr / len(lrs), 2)}, average ssim: {round(total_ssim / len(lrs), 4)}")


if __name__ == '__main__':
    main()
