# -*- coding:utf-8 -*-
import os
import cv2
import glob
import torch
import numpy as np

from PIL import Image


def load_image(path):
    _, ext = os.path.splitext(os.path.basename(path))
    if ext == ".npy":
        image = np.load(path).astype(np.float32) / 255
    else:
        image = Image.open(path)
    return image


def png2tensor(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.  # HWC-BGR
    img = np.transpose(img if img.shape[-1] == 1 else img[:, :, [2, 1, 0]], (2, 0, 1))  # CHW-RGB
    img = torch.from_numpy(img).float().unsqueeze(0)  # BCHW-RGB
    return img


def tensor2png(img_tensor):
    img = img_tensor.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if img.ndim == 3:
        img = np.transpose(img[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
    img = (img * 255.0).round().astype(np.uint8)  # float32 to uint8
    return img


def image_padding(img_tensor, scale):
    _, _, h_old, w_old = img_tensor.size()
    h_pad = (h_old // scale + 1) * scale - h_old
    w_pad = (w_old // scale + 1) * scale - w_old
    img_tensor = torch.cat([img_tensor, torch.flip(img_tensor, [2])], 2)[:, :, : h_old + h_pad, :]
    img_tensor = torch.cat([img_tensor, torch.flip(img_tensor, [3])], 3)[:, :, :, : w_old + w_pad]
    return img_tensor, h_old, w_old


def get_image_pairs(path, scale):
    if "DIV2K" in path:
        lrs = sorted(glob.glob(os.path.join(path, "DIV2K_valid_LR_bicubic", f"X{str(scale)}", "*.png")))
        hrs = sorted(glob.glob(os.path.join(path, "DIV2K_valid_HR", "*.png")))
    else:
        root = os.path.join(path, f"image_SRF_{str(scale)}")
        lrs = sorted(glob.glob(os.path.join(root, "*_LR.png")))
        hrs = sorted(glob.glob(os.path.join(root, "*_HR.png")))
    assert len(lrs) == len(hrs)
    return lrs, hrs
