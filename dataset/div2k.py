# -*- coding: utf-8 -*-
"""
DIV2K Dataset Class
"""

import os
import numpy as np

from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.image import load_image


class DIV2K(Dataset):
    def __init__(self, path, trans=transforms.ToTensor(), train=True, factor=4, ext='npy'):
        """
        DIV2K dataset class init function
        Args:
            path: local path of DIV2K dataset.
            trans: torchvision transforms.
            train: get training dataset if this arg is True, otherwise get valid dataset.
            factor: downscale factor.
            ext: file's extension.
        """
        super(DIV2K, self).__init__()

        assert ext in ['png', 'npy']
        assert factor in [2, 3, 4]
        assert os.path.exists(path)

        self.path = path
        self.trans = trans
        self.train = train
        self.factor = factor
        self.ext = ext

        self.lr_images, self.hr_images = self.scan_files()

    def scan_files(self):
        """
        Obtain file path information through "glob" method.
        Returns:
            HR & LR file path information
        """
        lr_folder = os.path.join(
            self.path,
            f'DIV2K_{"train" if self.train else "valid"}_LR_bicubic{"_decode" if self.ext == "npy" else ""}',
            f'X{str(self.factor)}')
        hr_folder = os.path.join(
            self.path,
            f'DIV2K_{"train" if self.train else "valid"}_HR{"_decode" if self.ext == "npy" else ""}'
        )

        assert os.path.exists(lr_folder) and os.path.exists(hr_folder)
        lr_images = sorted(glob.glob(os.path.join(lr_folder, f'*.{self.ext}')))
        hr_images = sorted(glob.glob(os.path.join(hr_folder, f'*.{self.ext}')))

        return lr_images, hr_images

    def load_file(self, idx):
        """
        Load file by extension.
        Args:
            idx: file's index

        Returns:
            LR & HR Image dataset of ndarray typy
        """
        if self.ext == 'npy':
            lr_image = np.load(self.lr_images[idx]).astype(np.float32) / 255.
            hr_image = np.load(self.hr_images[idx]).astype(np.float32) / 255.
        else:
            # lr_image = cv2.imread(self.lr_images[idx])
            # hr_image = cv2.imread(self.hr_images[idx])
            lr_image = Image.open(self.lr_images[idx])
            hr_image = Image.open(self.hr_images[idx])
        return lr_image, hr_image

    def __getitem__(self, item):
        lr_data, hr_data = self.load_file(item)
        lr_tensor = self.trans(lr_data)
        hr_tensor = self.trans(hr_data)

        return lr_tensor, hr_tensor

    def __len__(self):
        return len(self.lr_images)


class DIV2KMultiScales(Dataset):
    def __init__(self, path: str, upscale: list, trans=transforms.ToTensor()):
        super(DIV2KMultiScales, self).__init__()
        assert os.path.exists(path), f"{path} not EXISTS!"
        self.path = path
        self.upscale = upscale
        self.trans = trans
        self.lr_images, self.hr_images = self.scan()

    def scan(self):
        hr_images = []
        lr_images = []
        for scale in self.upscale:
            if scale == 1:
                lr_images += sorted(glob(os.path.join(self.path, "DIV2K_train_HR_decode", "*")))
            else:
                lr_images += sorted(
                    glob(os.path.join(self.path, "DIV2K_train_LR_bicubic_decode", f"X{str(scale)}", "*"))
                )
            hr_images += sorted(glob(os.path.join(self.path, "DIV2K_train_HR_decode", "*")))
        assert len(lr_images) == len(hr_images)
        return lr_images, hr_images

    def __getitem__(self, idx):
        lr_image = load_image(self.lr_images[idx])
        hr_image = load_image(self.hr_images[idx])

        lr_tensor = self.trans(lr_image)
        hr_tensor = self.trans(hr_image)

        return lr_tensor, hr_tensor

    def __len__(self):
        return len(self.lr_images)


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from torch.utils.data import DataLoader
    ds = DIV2KMultiScales(r"/data2/guesthome/wenbop/linzijie/workspace/dataset/DIV2K", upscale=[2, 3, 4])
    dl = DataLoader(ds, batch_size=8, shuffle=True)
    lrs, hrs = next(iter(dl))
    print(lrs.shape, hrs.shape)
    lr = lrs[0].permute(1, 2, 0).numpy()
    hr = hrs[0].permute(1, 2, 0).numpy()
    plt.subplot(1, 2, 1)
    plt.imshow(lr)
    plt.subplot(1, 2, 2)
    plt.imshow(hr)
    plt.show()
