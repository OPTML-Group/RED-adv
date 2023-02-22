from typing import List

import torch
import torchvision
from torchvision import transforms

from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter

import os


def CIFAR10(dir='/tmp', ffcv_dir='/tmp', batch_size=512):
    datasets = {
        'train': torchvision.datasets.CIFAR10(dir, train=True, download=True),
        'test': torchvision.datasets.CIFAR10(dir, train=False, download=True)
    }

    os.makedirs(ffcv_dir, exist_ok=True)
    for (name, ds) in datasets.items():
        writer = DatasetWriter(f'{ffcv_dir}/CIFAR10_{name}.beton', {
            'image': RGBImageField(),
            'label': IntField()
        })
        writer.from_indexed_dataset(ds)

    # Note that statistics are wrt to uin8 range, [0,255].
    CIFAR_MEAN = [125.307, 122.961, 113.8575]
    CIFAR_STD = [51.5865, 50.847, 51.255]
    BATCH_SIZE = batch_size

    loaders = {}
    for name in ['train', 'test']:
        label_pipeline: List[Operation] = [
            IntDecoder(), ToTensor(), ToDevice('cuda:0'), Squeeze()]
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

        # Add image transforms and normalization
        if name == 'train':
            image_pipeline.extend([
                RandomHorizontalFlip(),
                RandomTranslate(padding=2),
                # Note Cutout is done before normalization.
                Cutout(8, tuple(map(int, CIFAR_MEAN))),
            ])
        image_pipeline.extend([
            ToTensor(),
            ToDevice('cuda:0', non_blocking=True),
            ToTorchImage(),
            Convert(torch.float16),
            torchvision.transforms.Normalize(
                CIFAR_MEAN, CIFAR_STD),  # type: ignore
        ])

        # Create loaders
        loaders[name] = Loader(f'{ffcv_dir}/CIFAR10_{name}.beton',
                               os_cache=True,
                               batch_size=BATCH_SIZE,
                               num_workers=8,
                               order=OrderOption.RANDOM,
                               drop_last=(name == 'train'),
                               pipelines={'image': image_pipeline,
                                          'label': label_pipeline})
    return loaders


def CIFAR100(dir='/tmp', ffcv_dir='/tmp', batch_size=512):
    datasets = {
        'train': torchvision.datasets.CIFAR100(dir, train=True, download=True),
        'test': torchvision.datasets.CIFAR100(dir, train=False, download=True)
    }

    os.makedirs(ffcv_dir, exist_ok=True)
    for (name, ds) in datasets.items():
        writer = DatasetWriter(f'{ffcv_dir}/CIFAR100_{name}.beton', {
            'image': RGBImageField(),
            'label': IntField()
        })
        writer.from_indexed_dataset(ds)

    # Note that statistics are wrt to uin8 range, [0,255].
    CIFAR_MEAN = [125.307, 122.961, 113.8575]
    CIFAR_STD = [51.5865, 50.847, 51.255]
    BATCH_SIZE = batch_size

    loaders = {}
    for name in ['train', 'test']:
        label_pipeline: List[Operation] = [
            IntDecoder(), ToTensor(), ToDevice('cuda:0'), Squeeze()]
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

        # Add image transforms and normalization
        if name == 'train':
            image_pipeline.extend([
                RandomHorizontalFlip(),
                RandomTranslate(padding=2),
                # Note Cutout is done before normalization.
                Cutout(8, tuple(map(int, CIFAR_MEAN))),
            ])
        image_pipeline.extend([
            ToTensor(),
            ToDevice('cuda:0', non_blocking=True),
            ToTorchImage(),
            Convert(torch.float16),
            torchvision.transforms.Normalize(
                CIFAR_MEAN, CIFAR_STD),  # type: ignore
        ])

        # Create loaders
        loaders[name] = Loader(f'{ffcv_dir}/CIFAR100_{name}.beton',
                               os_cache=True,
                               batch_size=BATCH_SIZE,
                               num_workers=8,
                               order=OrderOption.RANDOM,
                               drop_last=(name == 'train'),
                               pipelines={'image': image_pipeline,
                                          'label': label_pipeline})
    return loaders


def TinyImageNet(dir='/tmp', ffcv_dir='/tmp', batch_size=512):
    # before running this function, prepare TinyImageNet in "dir"
    # see TinyImageNet prepartion in ./TinyImageNet
    # please bash run.sh in ./TinyImageNet before loading TinyImageNet
    datasets = {x: torchvision.datasets.ImageFolder(os.path.join(dir, x))
                for x in ['train', 'test']}

    os.makedirs(ffcv_dir, exist_ok=True)
    for (name, ds) in datasets.items():
        writer = DatasetWriter(f'{ffcv_dir}/TinyImageNet_{name}.beton', {
            'image': RGBImageField(),
            'label': IntField()
        })
        writer.from_indexed_dataset(ds)

    # Note that statistics are wrt to uin8 range, [0,255].
    TinyImageNet_MEAN = [x * 255 for x in [0.4802, 0.4481, 0.3975]]
    TinyImageNet_STD = [x * 255 for x in [0.2302, 0.2265, 0.2262]]
    BATCH_SIZE = batch_size

    loaders = {}
    for name in ['train', 'test']:
        label_pipeline: List[Operation] = [
            IntDecoder(), ToTensor(), ToDevice('cuda:0'), Squeeze()]
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

        # Add image transforms and normalization
        if name == 'train':
            image_pipeline.extend([
                RandomHorizontalFlip(),
                RandomTranslate(padding=2),
                # Note Cutout is done before normalization.
                Cutout(8, tuple(map(int, TinyImageNet_MEAN))),
            ])
        image_pipeline.extend([
            ToTensor(),
            ToDevice('cuda:0', non_blocking=True),
            ToTorchImage(),
            Convert(torch.float16),
            torchvision.transforms.Normalize(
                TinyImageNet_MEAN, TinyImageNet_STD),  # type: ignore
        ])

        # Create loaders
        loaders[name] = Loader(f'{ffcv_dir}/TinyImageNet_{name}.beton',
                               os_cache=True,
                               batch_size=BATCH_SIZE,
                               num_workers=8,
                               order=OrderOption.RANDOM,
                               drop_last=(name == 'train'),
                               pipelines={'image': image_pipeline,
                                          'label': label_pipeline})
    return loaders

def MNIST(dir='/tmp', batch_size=512):
    datasets = {
        'train': torchvision.datasets.MNIST(dir, train=True, download=True, transform=transforms.ToTensor()),
        'test': torchvision.datasets.MNIST(dir, train=False, download=True, transform=transforms.ToTensor())
    }
    
    loaders = {}
    for name in ['train', 'test']:
        # Create loaders
        loaders[name] = torch.utils.data.DataLoader(datasets[name], batch_size = batch_size)
    return loaders
