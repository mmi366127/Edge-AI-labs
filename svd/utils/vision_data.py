import os
import math
import time
import json
from tqdm import tqdm


from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader
import numpy as np
import torch




def data_loader_to_list(data_loader, num_batches):

    new_data_loader = []

    for i, data in enumerate(data_loader):
        if i >= num_batches:
            break
        new_data_loader.append(data)

    return new_data_loader


def build_dataset_CIFAR100(is_train, data_path):
    transform = build_transform(is_train)
    dataset = datasets.CIFAR100(data_path, train=is_train, transform=transform, download=True)
    nb_classes = 100
    return dataset, nb_classes


def build_transform(is_train):
    input_size = 224
    eval_crop_ratio = 1.0

    resize_im = input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=input_size,
            is_training=True,
            color_jitter=0.3,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            re_prob=0.0,
            re_mode='pixel',
            re_count=1,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int(input_size / eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def prepare_data(batch_size):
    train_set, nb_classes = build_dataset_CIFAR100(is_train=True, data_path='./data')
    test_set, _ = build_dataset_CIFAR100(is_train=False, data_path='./data')

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, test_loader, nb_classes


def getMiniTestDataset():
    # Create a test_loader with batch size = 1
    _, test_loader, _ = prepare_data(batch_size=1)

    # Prepare to collect 10 images per class
    class_images = [[] for _ in range(100)]

    # Iterate through the data
    for (image, label) in test_loader:
        if len(class_images[label]) < 5:
            class_images[label].append((image, label))
        if all(len(images) == 5 for images in class_images):
            break  # Stop once we have 10 images per class

    # flatten class_images
    mini_test_dataset = []
    for images in class_images:
        mini_test_dataset.extend(images)
    return mini_test_dataset


def get_calib_data(name, nsamples, batch_size=32, seed=42):
    from . import set_seed
    set_seed(seed)
    if name == "cifar100":
        train_set, nb_classes = build_dataset_CIFAR100(is_train=True, data_path='./data')
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        return data_loader_to_list(train_loader, nsamples // batch_size)

    else:
        raise NotImplementedError