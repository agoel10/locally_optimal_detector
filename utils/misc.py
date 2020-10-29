'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import errno
import os
import sys
import time
import math
import pickle
import torch
import numpy as np
import pdb
import PIL
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, roc_curve

import six
import lmdb

import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import pyarrow as pa
import os.path as osp
from PIL import Image

__all__ = ['get_mean_and_std', 'init_params', 'mkdir_p', 'approx', 'AverageMeter', 'computeROC', 'data_luminance', 'reverse_preprocessing', 'get_random_samples', 'get_patches', 'combine_patches', 'load_gmm_model', 'get_train_valid_loader', 'get_test_loader']


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)


class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = loads_pyarrow(txn.get(b'__len__'))
            self.keys = loads_pyarrow(txn.get(b'__keys__'))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])

        unpacked = loads_pyarrow(byteflow)

        # load img
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        im2arr = np.array(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # return img, target
        return im2arr, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


def dumps_pyarrow(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()


def approx(X, value):
    return (np.argmin(np.abs(X - value)))


def get_patches(x, image_size, patch_size):
    '''
     Accepts input x of shape (B,H,W) and returns output of shape (B*N, P) 
     where N is number of patches and P is the number of features in each patch
     B*N*P = B*H*W
    '''
    num_y = image_size[0] / patch_size[0]
    num_x = image_size[1] / patch_size[1]
    num_patches = num_y * num_x
    # break the input into patches
    x = x.unfold(1, patch_size[0], patch_size[0]).unfold(2, patch_size[1], patch_size[1])
    x = x.reshape(-1, patch_size[0] * patch_size[1])
    return x


def computeROC(testStatistics, method='given'):
    T0, T1 = testStatistics[0], testStatistics[1]
    if(method == 'max'):
        T1 = np.max(T1, axis=1)
        T0 = np.max(T0, axis=1)
        overall = np.ravel(np.concatenate((T0, T1)))
        label = np.ravel(np.concatenate((np.zeros(T0.shape), np.ones(T1.shape))))
        return roc_curve(label, overall, drop_intermediate=False)

    if(method == 'sum'):
        T1 = np.sum(T1, axis=1)
        T0 = np.sum(T0, axis=1)
        overall = np.ravel(np.concatenate((T0, T1)))
        label = np.ravel(np.concatenate((np.zeros(T0.shape), np.ones(T1.shape))))
        return roc_curve(label, overall, drop_intermediate=False)

    if(method == 'given'):
        overall = np.ravel(np.concatenate((T0, T1)))
        label = np.ravel(np.concatenate((np.zeros(T0.shape), np.ones(T1.shape))))
        return roc_curve(label, overall, drop_intermediate=False)


def combine_patches(x, image_size, patch_size):
    '''
     Converts the input x of shape (B*N, P) to shape (B, H, W). 
    '''
    num_y = int(image_size[0] / patch_size[0])
    num_x = int(image_size[1] / patch_size[1])
    num_patches = num_y * num_x
    x = x.reshape(-1, num_y, num_x, patch_size[0], patch_size[1])
    x = x.permute(0, 1, 3, 2, 4)
    x = torch.reshape(x, (-1, image_size[0], image_size[1]))
    return x


def load_gmm_model(location):
    #location = '../generative_models/dataset_%s_patch_%s_components_%s_luminance.p'%(dataset,patch_size,n_components)
    gmm_model = pickle.load(open(location, "rb"))
    p_centers = torch.tensor(gmm_model.weights_, dtype=torch.float32, device=torch.device("cuda"), requires_grad=False)
    mu = list(map(lambda x: torch.tensor(x, dtype=torch.float32, device=torch.device("cuda"), requires_grad=False), list(gmm_model.means_)))
    Sigma_inv = list(map(lambda x: torch.tensor(x, dtype=torch.float32, device=torch.device("cuda"), requires_grad=False), list(gmm_model.precisions_)))
    n_components = len(p_centers)
    return (mu, Sigma_inv, p_centers)


def reverse_preprocessing(x):
    mean, std = ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    y = torch.zeros(x.shape, device='cuda', dtype=torch.float32)
    if(len(x.shape) == 4):
        for ch in range(3):
            y[:, ch, :, :] = (x[:, ch, :, :] * std[ch] + mean[ch]) * 255
        return y
    else:
        for ch in range(3):
            y[:, :, ch, :, :] = (x[:, :, ch, :, :] * std[ch] + mean[ch]) * 255
        return y


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def data_luminance(data_X, type='input'):
    # Hard coded values for CIFAR100
    mean = 124.34
    std = 64.2
    if type == 'input':
        if(len(data_X.shape) == 4):
            data_X = 255 * data_X
            luminance = 0.2126 * data_X[:, 0, :, :] + 0.7152 * data_X[:, 1, :, :] + 0.0722 * data_X[:, 2, :, :]
            return torch.squeeze((luminance - mean) / std)

        if(len(data_X.shape) == 5):
            luminance = 0.2126 * data_X[:, :, 0, :, :] + 0.7152 * data_X[:, :, 1, :, :] + 0.0722 * data_X[:, :, 2, :, :]
            return torch.squeeze((luminance - mean) / std)

        if len(data_X.shape) <= 3:
            return data_X

    if type == 'perturbation':
        if(len(data_X.shape) == 4):
            data_X = 255 * data_X
            luminance = 0.2126 * data_X[:, 0, :, :] + 0.7152 * data_X[:, 1, :, :] + 0.0722 * data_X[:, 2, :, :]
            return torch.squeeze((luminance) / std)

        if(len(data_X.shape) == 5):
            data_X = 255 * data_X
            luminance = 0.2126 * data_X[:, :, 0, :, :] + 0.7152 * data_X[:, :, 1, :, :] + 0.0722 * data_X[:, :, 2, :, :]
            return torch.squeeze((luminance) / std)
        if len(data_X.shape) <= 3:
            return data_X


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_train_valid_loader(dataset,
                           data_dir,
                           batch_size,
                           augment=True,
                           random_seed=1234,
                           valid_size=0.2,
                           shuffle=True,
                           show_sample=False,
                           num_workers=4,
                           pin_memory=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    valid_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # load the dataset
    if dataset == 'CIFAR10':
        if augment:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        train_dataset = datasets.CIFAR10(
            root=data_dir, train=True,
            download=True, transform=train_transform,
        )
        valid_dataset = datasets.CIFAR10(
            root=data_dir, train=True,
            download=True, transform=valid_transform,
        )

    if dataset == 'CIFAR100':
        if augment:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
            ])

        train_dataset = datasets.CIFAR100(
            root=data_dir, train=True,
            download=True, transform=train_transform,
        )
        valid_dataset = datasets.CIFAR100(
            root=data_dir, train=True,
            download=True, transform=valid_transform,
        )

    if dataset == 'ImageNet':
        valid_size = 0.2
        shuffle = False
        train_dataset = ImageFolderLMDB(data_dir, transforms.Compose([
            transforms.Resize((224, 224), interpolation=PIL.Image.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]))
        valid_dataset = ImageFolderLMDB(data_dir, transforms.Compose([
            transforms.Resize((224, 224), interpolation=PIL.Image.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]))

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    # visualize some images
    if show_sample:
        sample_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=9, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        data_iter = iter(sample_loader)
        images, labels = data_iter.next()
        X = images.numpy().transpose([0, 2, 3, 1])
        plot_images(X, labels)

    return (train_loader, valid_loader)


def get_test_loader(dataset,
                    data_dir,
                    batch_size,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=False):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transform

    if dataset == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        dataset = datasets.CIFAR10(
            root=data_dir, train=False,
            download=True, transform=transform,
        )

    if dataset == 'CIFAR100':
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        dataset = datasets.CIFAR100(
            root=data_dir, train=False,
            download=True, transform=transform,
        )

    if dataset == 'ImageNet-100':
        dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transforms.Compose([
            transforms.Resize((224, 224), interpolation=PIL.Image.BILINEAR),
            transforms.ToTensor(),
        ]))

    if dataset == 'ImageNet':
        valid_size = 0.2
        shuffle = False
        dataset = ImageFolderLMDB(data_dir, transforms.Compose([
            transforms.Resize((224, 224), interpolation=PIL.Image.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]))
        num_train = len(dataset)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))

        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        valid_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=valid_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        return valid_loader

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader
