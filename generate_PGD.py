from sys import _current_frames
from typing import List

import torch as ch
import torchvision
from torch.cuda.amp import GradScaler, autocast  # type: ignore
from torch.nn import CrossEntropyLoss

from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter

import argparse
import time
import os
import numpy as np

from attacks.PGD import PGD, PGD_l2

start = time.time()

parser = argparse.ArgumentParser(description='CIFAR-10 training')
# parser.add_argument('--num_resnet_layer', type=int, default=20)
parser.add_argument('--seed', type=int,default=0)
parser.add_argument('--activation_function', type=str,choices= ["relu", "tanh", "elu"])
parser.add_argument('--kernel_size', type=int, choices= [3, 5, 7])
parser.add_argument('--pruning_ratio', type=float, default=0, choices = [0.0, 0.375, 0.625])
parser.add_argument('--rewind_epoch', type=int, default=2)
parser.add_argument('--steps', type=int, default=10)
parser.add_argument('--eps', type=float, default=8/255)
parser.add_argument('--dataset', type=str, default="cifar10")
parser.add_argument('--times', type=int, default=1)
parser.add_argument('--attack', type=str, default='PGD', choices = ['PGD', 'PGD_l2'])
# parser.add_argument('--structured_pruning', default=True, action=argparse.BooleanOptionalAction)
args = parser.parse_args()

seed = args.seed
ch.manual_seed(seed)
ch.cuda.manual_seed(seed)
ch.cuda.manual_seed_all(seed)
ch.backends.cudnn.benchmark = False
ch.backends.cudnn.deterministic = True
np.random.seed(seed)


datasets = {
    # 'train': torchvision.datasets.CIFAR10('/tmp', train=True, download=True),
    'test': torchvision.datasets.CIFAR10('/tmp', train=False, download=True)
}

for (name, ds) in datasets.items():
    writer = DatasetWriter(f'/tmp/cifar_{name}.beton', {
        'image': RGBImageField(),
        'label': IntField()
    })
    writer.from_indexed_dataset(ds)

# Note that statistics are wrt to uin8 range, [0,255].
CIFAR_MEAN = [125.307, 122.961, 113.8575]
CIFAR_STD = [51.5865, 50.847, 51.255]

BATCH_SIZE = 2000

loaders = {}
for name in ['test']:
    label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice('cuda:0'), Squeeze()]
    image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

    # Add image transforms and normalization
    image_pipeline.extend([
        ToTensor(),
        ToDevice('cuda:0', non_blocking=True),
        ToTorchImage(),
        Convert(ch.float16),
        torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),  # type: ignore
    ])

    # Create loaders
    loaders[name] = Loader(f'/tmp/cifar_{name}.beton',
                            os_cache=True,
                            batch_size=BATCH_SIZE,
                            num_workers=8,
                            order=OrderOption.RANDOM,
                            drop_last=(name == 'train'),
                            pipelines={'image': image_pipeline,
                                       'label': label_pipeline})

class Mul(ch.nn.Module):
    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight
    def forward(self, x): return x * self.weight

class Flatten(ch.nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

class Residual(ch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module
    def forward(self, x): return x + self.module(x)



NUM_CLASSES = 10


scaler = GradScaler()
loss_fn = CrossEntropyLoss()


total_correct, total_num = 0., 0.
if not os.path.exists("pgd_pt"):
    os.makedirs("pgd_pt")
if not os.path.exists("labs"):
    os.makedirs("labs")

for ks in [3,5,7]:
    for act in ['elu','relu','tanh']:
        for pr in ['0.0',  '0.375_structFalse', '0.375_structTrue', '0.625_structFalse', '0.625_structTrue']:
            print(f"{ks},{act},{pr}")
            args.kernel_size = ks
            args.activation_function = act
            def conv_bn(channels_in, channels_out, kernel_size=args.kernel_size, stride=1, padding=int((args.kernel_size-1)/2), groups=1, activation_function=args.activation_function):
                assert activation_function in ["relu", "tanh", "elu"]
                
                if activation_function == "relu":
                    mod = ch.nn.ReLU(inplace=True)
                elif activation_function == 'tanh':
                    mod = ch.nn.Tanh()
                elif activation_function == 'elu':
                    mod = ch.nn.ELU(inplace=True)
                
                return ch.nn.Sequential(
                        ch.nn.Conv2d(channels_in, channels_out,
                                    kernel_size=kernel_size, stride=stride, padding=padding,
                                    groups=groups, bias=False),
                        ch.nn.BatchNorm2d(channels_out),
                        mod
                )
            model = ch.nn.Sequential(
                conv_bn(3, 64, kernel_size=args.kernel_size, stride=1, padding=int((args.kernel_size-1)/2)), # if ks = 3 then padding = 1; if ks = 1 then padding = 0; if ks = 5 then padding = 2
                conv_bn(64, 128, kernel_size=args.kernel_size, stride=2, padding=int((args.kernel_size+1)/2)), # if ks = 3 then padding = 2; if 
                Residual(ch.nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
                conv_bn(128, 256, kernel_size=args.kernel_size, stride=1, padding=int((args.kernel_size-1)/2)),
                ch.nn.MaxPool2d(2),
                Residual(ch.nn.Sequential(conv_bn(256, 256), conv_bn(256, 256))),
                conv_bn(256, 128, kernel_size=args.kernel_size, stride=1, padding=max(0,int((args.kernel_size-3)/2))),
                ch.nn.AdaptiveMaxPool2d((1, 1)),
                Flatten(),
                ch.nn.Linear(128, NUM_CLASSES, bias=False),
                Mul(0.2)
            )
            

            model = model.to(memory_format=ch.channels_last).cuda()  # type: ignore

            args.seed = 0
            for i in [0]:
                model.load_state_dict(ch.load("model_pt_{}/resnet9_seed{}_ks{}_act{}_prune{}.pt".format(args.dataset, i, ks, act, pr)))
                model.eval()
                for ims, labs in loaders['test']:
                    with autocast():
                        # we may need to change the attack type here from PGD to PGD_l2
                        if args.attack == 'PGD':
                            ims_adv = PGD(ims, labs, model, loss_fn, scaler, args.steps, args.eps)
                        elif args.attack == 'PGD_l2':
                            ims_adv = PGD_l2(ims, labs, model, loss_fn, scaler, args.steps, args.eps)
                        
                        ori_out = model(ims)
                        adv_out = model(ims_adv)  # Test-time augmentation

                        idx = ori_out.argmax(1).eq(labs) * adv_out.argmax(1).ne(labs)
                        print(sum(idx))
                        ims_adv_save = (ims_adv-ims)[idx]*args.times
                        labs = labs[idx]
                        
                        ch.save(ims_adv_save,"pgd_pt/{}_ims_adv_resnet9_seed{}_ks{}_act{}_prune{}.pt".format(args.attack, args.seed, ks, act, pr))
                        ch.save(ims_adv[idx],"x_adv/{}_x_adv_resnet9_seed{}_ks{}_act{}_prune{}.pt".format(args.attack, args.seed, ks, act, pr))
                        ch.save(labs,"labs/{}_lab_resnet9_seed{}_ks{}_act{}_prune{}.pt".format(args.attack, args.seed, ks, act, pr))
                        args.seed += 1
            
        
