# -*- coding: utf-8 -*-
"""
DIV2K Dataset Class
"""

import os
import glob

import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms



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


class Div2kRandom(Dataset):
    def __init__(self, path, num_of_images, trans=transforms.ToTensor(), seed=None):
        super(Div2kRandom, self).__init__()
        assert os.path.exists(path)

        self.path = path
        self.num_of_images = num_of_images
        self.trans = trans
        self.seed = seed

        self.lr_paths, self.hr_paths = self.scanf_files()

    def scanf_files(self):
        hr_all_paths = sorted(glob.glob(os.path.join(self.path, 'DIV2K_train_HR_decode', '*.npy')))
        lr_all_paths_2 = sorted(glob.glob(os.path.join(self.path, 'DIV2K_train_LR_bicubic_decode', f'X2', '*.npy')))
        lr_all_paths_3 = sorted(glob.glob(os.path.join(self.path, 'DIV2K_train_LR_bicubic_decode', f'X3', '*.npy')))
        lr_all_paths_4 = sorted(glob.glob(os.path.join(self.path, 'DIV2K_train_LR_bicubic_decode', f'X4', '*.npy')))

        index = [d for d in range(len(hr_all_paths))]
        if self.seed is not None:
            np.random.seed(self.seed)

        # factor 1
        # np.random.shuffle(index)
        # hr_paths = [hr_all_paths[d] for d in index[:self.num_of_images]]
        # lr_paths = [hr_all_paths[d] for d in index[:self.num_of_images]]

        # factor 2
        np.random.shuffle(index)
        hr_paths = [hr_all_paths[d] for d in index[:self.num_of_images]]
        lr_paths = [lr_all_paths_2[d] for d in index[:self.num_of_images]]

        # factor 3
        np.random.shuffle(index)
        hr_paths += [hr_all_paths[d] for d in index[:self.num_of_images]]
        lr_paths += [lr_all_paths_3[d] for d in index[:self.num_of_images]]

        # factor 4
        np.random.shuffle(index)
        hr_paths += [hr_all_paths[d] for d in index[:self.num_of_images]]
        lr_paths += [lr_all_paths_4[d] for d in index[:self.num_of_images]]

        # print(len(lr_paths), len(hr_paths))

        return lr_paths, hr_paths

    def __getitem__(self, item):
        lr = np.load(self.lr_paths[item])
        hr = np.load(self.hr_paths[item])
        lr_tensor = self.trans(lr)
        hr_tensor = self.trans(hr)
        return lr_tensor, hr_tensor

    def __len__(self):
        return len(self.lr_paths)


class Div2kBenchmark(Dataset):
    def __init__(self, path, factor, trans=transforms.ToTensor()):
        super(Div2kBenchmark, self).__init__()
        assert os.path.exists(path)
        assert factor in [2, 3, 4, 6, 8]
        self.path = path
        self.factor = factor
        self.trans = trans
        self.lr_paths, self.hr_paths = self.scan_files()


    def scan_files(self):
        hr_paths = sorted(glob.glob(
            os.path.join(self.path, 'DIV2K_valid_HR', '*.png')))
        lr_paths = sorted(glob.glob(
            os.path.join(self.path, 'DIV2K_valid_LR_bicubic', f'X{str(self.factor)}', '*.png')))
        return lr_paths, hr_paths

    def __getitem__(self, idx):
        lr_img = Image.open(self.lr_paths[idx])
        hr_img = Image.open(self.hr_paths[idx])
        lr_tensor = self.trans(lr_img)
        hr_tensor = self.trans(hr_img)
        return lr_tensor, hr_tensor

    def __len__(self):
        return len(self.lr_paths)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    ds = DIV2K(path=r'../data/DIV2K', factor=3)
    dl = DataLoader(ds, batch_size=18, shuffle=True)

    lrs, hrs = next(iter(dl))
    print(lrs.shape, hrs.shape)
    lr = lrs[0].permute(1, 2, 0).numpy()
    hr = hrs[0].permute(1, 2, 0).numpy()

    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(lr)
    plt.subplot(1, 2, 2)
    plt.imshow(hr)
    plt.savefig('test.png')
    # plt.show()
