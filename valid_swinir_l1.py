# -*- coding:utf-8 -*-

import os
import cv2
import argparse
import numpy as np
import pandas as pd

import torch

from models.swinir import SwinIR
from utils import image, metric


def main():
    parser = argparse.ArgumentParser(description="valid swinir")
    parser.add_argument("--upscale", default=3)
    parser.add_argument("--img_size", default=80)
    parser.add_argument("--window_size", default=8)
    parser.add_argument("--depths", default=[6, 6, 6, 6])
    parser.add_argument("--embed_dim", default=60)
    parser.add_argument("--num_heads", default=[6, 6, 6, 6])
    parser.add_argument("--upsampler", default='pixelshuffle')
    parser.add_argument("--resi_connection", default='1conv')

    parser.add_argument("--model_path", default=r"./checkpoints/swinir/l1/epoch50_DIV2K_s80w8_x3.pth")
    parser.add_argument("--gpu_device", default="2")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SwinIR(upscale=args.upscale, in_chans=3, img_size=args.img_size, window_size=args.window_size,
                   img_range=1., depths=args.depths, embed_dim=args.embed_dim, num_heads=args.num_heads,
                   mlp_ratio=2, upsampler=args.upsampler, resi_connection=args.resi_connection)
    model = model.to(device)

    assert os.path.exists(args.model_path)
    pretrained_model = torch.load(args.model_path)
    model.load_state_dict(pretrained_model)
    model.eval()

    metric_table = []

    for ds_name in ['Set5', 'Set14', 'BSD100', 'Urban100', 'DIV2K']:

        lrs, hrs = image.get_image_pairs(os.path.join(r"./data", ds_name), scale=args.upscale)

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

        avg_psnr = round(total_psnr / len(lrs), 2)
        avg_ssim = round(total_ssim / len(lrs), 4)

        print(f"[{ds_name}/x{args.upscale}] average psnr: {avg_psnr}, average ssim: {avg_ssim}")
        metric_table.append([ds_name, avg_psnr, avg_ssim])

    result_file = f"./output/swinir/metric_swinir_l1_x{str(args.upscale)}.csv"
    pd.DataFrame(metric_table, columns=['Dataset', 'PSNR', 'SSIM'], index=None).to_csv(result_file)
    print(f"result save to {result_file}")


if __name__ == '__main__':
    main()
