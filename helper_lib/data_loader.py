"""Shared preprocessing prevents training/inference mismatch."""
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
from pathlib import Path
import numpy as np
from PIL import Image
import torch

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']


def image_transform():
    return transforms.Compose([
        transforms.Resize((64, 64), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def get_data_loaders(data_dir='./data', batch_size=128):
    training = get_dataset(data_dir, train=True, transform=image_transform())
    train, validation = random_split(training, [45000, 5000],
                                    generator=torch.Generator().manual_seed(42))
    test = get_dataset(data_dir, train=False, transform=image_transform())
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(validation, batch_size=batch_size, shuffle=False, num_workers=0),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0),
    )


class MirrorCIFAR10(Dataset):
    classes = CLASSES

    def __init__(self, path, transform=None):
        with np.load(path, allow_pickle=False) as data:
            self.images = data['images']
            self.labels = data['labels']
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = Image.fromarray(self.images[index])
        if self.transform:
            image = self.transform(image)
        return image, int(self.labels[index])


def get_dataset(data_dir='./data', train=True, transform=None):
    mirror = Path(data_dir) / ('train.npz' if train else 'test.npz')
    if mirror.exists():
        return MirrorCIFAR10(mirror, transform)
    return datasets.CIFAR10(data_dir, train=train, download=True, transform=transform)
