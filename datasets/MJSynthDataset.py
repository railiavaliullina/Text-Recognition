import torch
import os
from torchvision import transforms
from PIL import Image
import numpy as np
from torchvision.transforms import functional as f
from skimage.util import random_noise
import matplotlib.pyplot as plt


class MJSynthDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, dataset_type='train'):
        self.cfg = cfg
        self.dataset_type = dataset_type
        self.vertical_crop_h = 32

        self.paths, self.labels, self.ids = [], [], []

        # read dataset
        print(f'Loading {dataset_type} data...')
        with open(self.cfg.annotation_files_path[self.dataset_type], 'r') as f:
            annotation = f.readlines()  # TODO  [:100]

        images_num = len(annotation)
        for idx, ant in enumerate(annotation):
            if idx % 1e6 == 0:
                print(f'loaded {idx}/{images_num}')
            ant = ant.replace('\n', '')
            path = os.path.join(self.cfg.dataset_path, ant.split()[0].replace('./', ''))
            self.paths.append(path)
            self.labels.append(ant.split('_')[1])
        self.ids = np.arange(self.__len__())

    @staticmethod
    def show_image(image, title=''):
        im_to_show = image
        if len(image.shape) > 2:  # rgb
            im_to_show = image.transpose((1, 2, 0))
        plt.imshow(im_to_show)
        plt.title(title)
        plt.show()

    def apply_augmentation(self, img):
        # self.show_image(np.asarray(img).transpose((2, 0, 1)))
        w_unpadded, h = img.size
        # 1. vertical resize
        vertical_crop_w = int(round((self.cfg.vertical_crop_h * w_unpadded) / h))
        img = transforms.Resize((self.cfg.vertical_crop_h, vertical_crop_w))(img)

        if self.dataset_type == 'train':
            # 2a. random horizontal crop
            w, h = img.size
            crop_part = np.random.choice(np.arange(0.9, 1.0, 0.01), 1)[0]
            new_w = int(round(w * crop_part))
            margin = (w - new_w) // 2
            img = f.crop(img, 0, margin, h, new_w)
            # 2b. random horizontal resize
            w, h = img.size
            resize_ratio = np.random.choice(np.arange(0.8, 1.2, 0.1), 1)[0]  # uniform
            new_w = np.min([int(round(resize_ratio * w)), 500])
            img = transforms.Resize((h, new_w))(img)
            # self.show_image((np.asarray(img).transpose((2, 0, 1))))
            # 2c. random gaussian noise
            sigma = np.random.choice(np.arange(0, 0.02, 0.005))
            img = np.asarray(img).astype(np.float).transpose((2, 0, 1)) #/ 255
            # img = np.array([random_noise(img[c, :, :], mode='gaussian', mean=0, var=sigma, clip=True)
            #                 for c in range(3)]) * 255
            # self.show_image(img)
        else:
            # 2. horizontal resize
            if vertical_crop_w > self.cfg.horizontal_size_thr:
                img = transforms.Resize((self.cfg.vertical_crop_h, self.cfg.horizontal_size_thr))(img)
            img = np.asarray(img).astype(np.float).transpose((2, 0, 1))

        # 3. image whitening
        img = (img - self.cfg.mean) / 255
        return img.astype(np.float32), w_unpadded

    def __len__(self):
        return len(self.labels)

    def encode_labels(self, label):
        label = [self.cfg.alphabet[s] for s in label]
        return label

    def __getitem__(self, idx):
        assert os.path.exists(self.paths[idx])
        img = Image.open(self.paths[idx])
        img, w_unpadded = self.apply_augmentation(img)
        label = self.encode_labels(self.labels[idx])
        return img, label
