# -*- coding:utf-8 -*-
import argparse
import os

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset.div2k import DIV2K
from models.swinir import SwinIR
from utils import image, metric


def main():
    parser = argparse.ArgumentParser(description="train swinir")
    parser.add_argument("--cuda", default=True)
    parser.add_argument("--gpu_device", default="0")

    parser.add_argument("--upscale", default=3)
    parser.add_argument("--img_size", default=80)
    parser.add_argument("--window_size", default=8)
    parser.add_argument("--depths", default=[6, 6, 6, 6])
    parser.add_argument("--embed_dim", default=60)
    parser.add_argument("--num_heads", default=[6, 6, 6, 6])
    parser.add_argument("--upsampler", default='pixelshuffle')
    parser.add_argument("--resi_connection", default='1conv')

    parser.add_argument("--epochs", default=50)
    parser.add_argument("--lr", default=2.e-4)
    parser.add_argument("--step_size", default=10)
    parser.add_argument("--gamma", default=0.5)

    parser.add_argument("--train_dataset", default=r"./data/DIV2K")
    parser.add_argument("--valid_dataset", default=r"./data/Set5")
    parser.add_argument("--batch_size", default=10)

    parser.add_argument("--checkpoint", default=r"./checkpoints/swinir/l1")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_device
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    model = SwinIR(upscale=args.upscale, in_chans=3, img_size=args.img_size, window_size=args.window_size,
                   img_range=1., depths=args.depths, embed_dim=args.embed_dim, num_heads=args.num_heads,
                   mlp_ratio=2, upsampler=args.upsampler, resi_connection=args.resi_connection)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_ds = DIV2K(path=args.train_dataset, factor=args.upscale)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model = model.to(device)

    for epoch in range(args.epochs):
        print(f"[{epoch + 1}/{args.epochs}] training model with DIV2K dataset...")

        model.train()

        total_loss = 0.
        total_count = 0.

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

        print(f"[{epoch + 1}/{args.epochs}] average L1 loss: {round(total_loss / total_count, 6)}")

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

            sr_img = image.tensor2png(sr_tensor)
            gt_img = cv2.imread(hrs[i], cv2.IMREAD_COLOR).astype(np.uint8)

            psnr = metric.calculate_psnr(sr_img, gt_img, crop_border=args.window_size, test_y_channel=True)
            ssim = metric.calculate_ssim(sr_img, gt_img, crop_border=args.window_size, test_y_channel=True)

            total_psnr += psnr
            total_ssim += ssim

        print(f"[{epoch + 1}/{args.epochs}] "
              f"average psnr: {round(total_psnr / len(lrs), 2)}, average ssim: {round(total_ssim / len(lrs), 4)}")

        checkpoint_path = os.path.join(
            args.checkpoint, f'epoch{epoch+1}_DIV2K_s{args.img_size}w{args.window_size}_x{args.upscale}.pth')
        torch.save(model.state_dict(), checkpoint_path)


if __name__ == '__main__':
    main()
