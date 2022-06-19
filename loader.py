from cProfile import label
from pathlib import Path
from itertools import chain
from PIL import Image
import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

def listdir(dname):
    # read repeat file name
    # fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                           #  for ext in ['png', 'jpg', 'jpeg', 'JPG']]))

    fnames = [_ for _ in os.listdir(dname) if _.endswith('jpg')]
    return fnames


def DefaultDataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples = listdir(root)
        self.samples.sort()
        self.transform = transform
        self.targets = None

    def __getitem__(self, index):
        fname = self.samples[index]
        img = Image.open(fname).convert('RGB').copy()
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)


def ReferenceDataset(Dataset):
    def __init__(self, root, transform-None):
        self.sample, self.targets = self._make_dataset(root)
        self.transform = transform

    def _make_dataset(self, root):
        domains = os.listdir(root)
        fnames, fnames2, labels = [], [], []

        for idx, domain in enumerate(sorted(domains)):
            class_dir = os.path.join(root, domain)
            cls_fnames = listdir(class_dir)
            fnames += cls_fnames
            fnames2 += random.sample(cls_fnames, len(cls_fnames))
            labels += [idx] * len(cls_fnames)

        return list(zip(fnames, fnames2)), labels

    def __getitem__(self, index):
        fname, fname2 = self.samples[index]
        label = self.targets[index]
        img = Image.open(fname).convert('RGB').copy()
        img2 = Image.open(fname2).convert('RGB').copy()

        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)
        return img, img2, label

    def __len__(self):
        return len(self.targets)


def _make_balanced_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]

    # WeightedRandomSampler can solve unbalance sample number
    return WeightedRandomSampler(weights, len(weights))

def get_train_loader(root, which='source', img_size=256, batch_size=8, prob=0.5, num_workers=4):
    crop = transforms.RandimResizedCrop(
        img_size, scale = [0.8, 1.0], 
        ratio = [0.9,1.1]
    )
    rand_crop = transforms.Compose(
        lambda x: crop(x) if random.random() < prob else x
    )

    transform = transforms.Compose([
        rand_crop, 
        transforms.Resize([img_size, img_size]),
        transforms.RamdomHorizontalFlip(),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])

    if which == 'source':
        dataset = ImageFolder(root, transform)
    elif which == 'reference':
        dataset = ReferenceDataset(root, transform)
    else:
        raise NotImplementedError

    sampler = _make_balanced_sampler(dataset.targets)

    return DataLoader(dataset=dataset, 
                      batch_size=batch_size,
                      sampler=sampler,
                      num_workers=num_workers,
                      pin_memory=True,
                      drop_last=True)


def get_eval_loader(root, img_size=256, batch_size=32,
                    imagenet_normalize=True, shuffle=True,
                    num_workers=4, drop_last=False):

    if imagenet_normalize:
        height, width = 299, 299
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        height, width = img_size, img_size
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.Resize([height, width]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    dataset = DefaultDataset(root, transform=transform)
    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=True,
                      drop_last=drop_last)


def get_test_loader(root, img_size=256, batch_size=32,
                    shuffle=True, num_workers=4):
    print('Preparing DataLoader for the generation phase...')
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    dataset = ImageFolder(root, transform)
    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=True)

class InputFetcher:
    def __init__(self, loader, loader_ref=None) :
        



    try:
        x, y = next(self.iter)