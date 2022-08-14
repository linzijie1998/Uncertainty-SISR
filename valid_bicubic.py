# -*- coding:utf-8 -*-
import os
import cv2
import numpy as np
import pandas as pd

from utils import image, metric

def main():
    psnr_table = []
    ssim_table = []

    for ds_name in ['Set5', 'Set14', 'BSD100', 'Urban100', 'DIV2K']:
        psnr_data = [ds_name]
        ssim_data = [ds_name]

        for scale in [2, 3, 4]:
            lrs, hrs = image.get_image_pairs(os.path.join(r"./data", ds_name), scale=scale)

            total_psnr = 0
            total_ssim = 0

            for i in range(len(lrs)):
                lr = cv2.imread(lrs[i], cv2.IMREAD_COLOR).astype(np.uint8)
                hr = cv2.imread(hrs[i], cv2.IMREAD_COLOR).astype(np.uint8)

                h, w, _ = lr.shape
                bicubic_img = cv2.resize(lr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

                psnr = metric.calculate_psnr(hr, bicubic_img, crop_border=4, test_y_channel=True)
                ssim = metric.calculate_ssim(hr, bicubic_img, crop_border=4, test_y_channel=True)

                total_psnr += psnr
                total_ssim += ssim

            print(f"[{ds_name}/x{scale}] average psnr: {total_psnr / len(lrs)}, average ssim: {total_ssim / len(lrs)}")

            psnr_data.append(round(total_psnr / len(lrs), 4))
            ssim_data.append(round(total_ssim / len(lrs), 4))

        psnr_table.append(psnr_data)
        ssim_table.append(ssim_data)

    pd.DataFrame(psnr_table, columns=["Dataset", "X2", "X3", "X4"]).to_csv("./metric_psnr_bicubic.csv", index=False)
    pd.DataFrame(ssim_table, columns=["Dataset", "X2", "X3", "X4"]).to_csv("./metric_ssim_bicubic.csv", index=False)

if __name__ == '__main__':
    main()
