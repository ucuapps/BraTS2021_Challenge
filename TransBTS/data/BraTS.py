import os
import torch
from torch.utils.data import Dataset
import random
import numpy as np
from torchvision.transforms import transforms
import pickle
from scipy import ndimage


def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


class MaxMinNormalization(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        Max = np.max(image)
        Min = np.min(image)
        image = (image - Min) / (Max - Min)

        return {'image': image, 'label': label}


class Random_Flip(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        if random.random() < 0.5:
            image = np.flip(image, 0)
            label = np.flip(label, 0)
        if random.random() < 0.5:
            image = np.flip(image, 1)
            label = np.flip(label, 1)
        if random.random() < 0.5:
            image = np.flip(image, 2)
            label = np.flip(label, 2)

        return {'image': image, 'label': label}


class Random_Crop(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        H = random.randint(0, 240 - 128)
        W = random.randint(0, 240 - 128)
        D = random.randint(0, 160 - 128)

        image = image[H: H + 128, W: W + 128, D: D + 128, ...]
        label = label[..., H: H + 128, W: W + 128, D: D + 128]

        return {'image': image, 'label': label}


class Random_intencity_shift(object):
    def __call__(self, sample, factor=0.1):
        image = sample['image']
        label = sample['label']

        scale_factor = np.random.uniform(1.0-factor, 1.0+factor, size=[1, image.shape[1], 1, image.shape[-1]])
        shift_factor = np.random.uniform(-factor, factor, size=[1, image.shape[1], 1, image.shape[-1]])

        image = image*scale_factor+shift_factor

        return {'image': image, 'label': label}


class Random_rotate(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        angle = round(np.random.uniform(-10, 10), 2)
        image = ndimage.rotate(image, angle, axes=(0, 1), reshape=False)
        label = ndimage.rotate(label, angle, axes=(0, 1), order=0, reshape=False)

        return {'image': image, 'label': label}


class Pad(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        image = np.pad(image, ((0, 0), (0, 0), (0, 5), (0, 0)), mode='constant')
        label = np.pad(label, ((0, 0), (0, 0), (0, 5)), mode='constant')
        return {'image': image, 'label': label}
    #(240,240,155)>(240,240,160)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image']
        image = np.ascontiguousarray(image.transpose(3, 0, 1, 2))
        label = sample['label']
        label = np.ascontiguousarray(label)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()

        return {'image': image, 'label': label}


def transform(sample):
    trans = transforms.Compose([
        Pad(),
        # Random_rotate(),  # time-consuming
        Random_Crop(),
        Random_Flip(),
        Random_intencity_shift(),
        ToTensor()
    ])

    return trans(sample)

# 2D code -----------------------------------------------------------------------

class Pad2D(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        image = np.pad(image, ((0, 0), (0, 0), (0, 0)), mode='constant')
        label = np.pad(label, ((0, 0), (0, 0)), mode='constant')
        return {'image': image, 'label': label}


class Random_Crop2D(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        H = random.randint(0, 240 - 128)
        W = random.randint(0, 240 - 128)

        image = image[H: H + 128, W: W + 128, ...]
        label = label[..., H: H + 128, W: W + 128]

        return {'image': image, 'label': label}


class Random_Flip2D(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        if random.random() < 0.5:
            image = np.flip(image, 0)
            label = np.flip(label, 0)
        if random.random() < 0.5:
            image = np.flip(image, 1)
            label = np.flip(label, 1)

        return {'image': image, 'label': label}


class ToTensor2D(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image']
        if image.shape[0] == 1:
            image = np.ascontiguousarray(image.squeeze(0).transpose(2, 0, 1))
        else:
            image = np.ascontiguousarray(image.transpose(2, 0, 1))
        label = sample['label']
        label = np.ascontiguousarray(label)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()

        return {'image': image, 'label': label}


def transform2d(sample):
    trans = transforms.Compose([
        Pad2D(),
        Random_Crop2D(),
        Random_Flip2D(),
        Random_intencity_shift(),
        ToTensor2D()
    ])

    return trans(sample)


def transform_valid2D(sample):
    trans = transforms.Compose([
        Pad2D(),
        ToTensor2D()
    ])

    return trans(sample)


def transform_valid(sample):
    trans = transforms.Compose([
        Pad(),
        ToTensor()
    ])

    return trans(sample)


class BraTS(Dataset):
    def __init__(self, list_file, root='', mode='train', return_names=False):
        self.lines = []
        paths, names = [], []
        with open(list_file) as f:
            for line in f:
                line = line.strip()
                name = line.split('/')[-1]
                names.append(name)
                path = os.path.join(root, line, name + '_')
                paths.append(path)
                self.lines.append(line)
        self.mode = mode
        self.names = names
        self.paths = paths

        self.return_names = return_names

    def __getitem__(self, item):
        path = self.paths[item]
        if self.mode == 'train':
            image, label = pkload(path + 'data_f32b0.pkl')
            sample = {'image': image, 'label': label}
            sample = transform(sample)
            result = [sample['image'], sample['label']]
        elif self.mode == 'valid':
            image, label = pkload(path + 'data_f32b0.pkl')
            sample = {'image': image, 'label': label}
            sample = transform_valid(sample)
            result = [sample['image'], sample['label']]
        else:
            image = pkload(path + 'data_f32b0.pkl')
            image = np.pad(image, ((0, 0), (0, 0), (0, 5), (0, 0)), mode='constant')
            image = np.ascontiguousarray(image.transpose(3, 0, 1, 2))
            image = torch.from_numpy(image).float()
            result = [image]

        if self.return_names:
            result.append(self.names[item])

        return result

    def __len__(self):
        return len(self.names)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]


class BraTS2D(Dataset):
    def __init__(self, mode='train', return_names=False):
        self.data_path = '/home/dobko/datasets/BraTS2021/{}_2d'.format(mode)
        self.imgs_paths = os.listdir(os.path.join(self.data_path, 'imgs'))
        self.return_names = return_names
        self.mode = mode

    def __getitem__(self, item):
        im_name = self.imgs_paths[item].split('/')[-1]
        image = np.load(os.path.join(self.data_path, 'imgs', im_name))
        label = np.load(os.path.join(self.data_path, 'lbls', im_name))
        sample = {'image': image, 'label': label}

        if self.mode == 'train':
            sample = transform2d(sample)
        elif self.mode == 'val':
            sample = transform_valid2D(sample)
        result = [sample['image'], sample['label']]

        if self.return_names:
            result.append(im_name)

        return result

    def __len__(self):
        return len(self.imgs_paths)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]


