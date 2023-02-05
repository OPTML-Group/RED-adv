from sys import _current_frames
from typing import List

import torch as ch
import torchvision
from torch.cuda.amp import GradScaler, autocast  # type: ignore
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler

from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter

import argparse
import time
from copy import deepcopy
import os
import numpy as np
from tqdm import tqdm


from prune.pruner import extract_mask, prune_model_custom, pruning_model, pruning_model_structured, remove_prune, check_sparsity
from PGD import PGD

parser = argparse.ArgumentParser(description='CIFAR-10 training')
# parser.add_argument('--num_resnet_layer', type=int, default=20)
parser.add_argument('--seed', type=int,default=0)
parser.add_argument('--activation_function', type=str,choices= ["relu", "tanh", "elu"])
parser.add_argument('--kernel_size', type=int, choices= [3, 5, 7])
parser.add_argument('--pruning_ratio', type=float, default=0, choices = [0.0, 0.375, 0.625])
parser.add_argument('--rewind_epoch', type=int, default=2)
parser.add_argument('--steps', type=int, default=10)
parser.add_argument('--eps', type=float, default=8/255)
parser.add_argument('--num_batch', type=int, default=15)
parser.add_argument('--folder_name', type=str, description= 'output folder name')
parser.add_argument('--attack', default = 'PGD', type=str)

# parser.add_argument('--structured_pruning', default=True, action=argparse.BooleanOptionalAction)
args = parser.parse_args()

seed = args.seed
ch.manual_seed(seed)
ch.cuda.manual_seed(seed)
ch.cuda.manual_seed_all(seed)
ch.backends.cudnn.benchmark = False
ch.backends.cudnn.deterministic = True
np.random.seed(seed)

if not os.path.exists(f"{args.folder_name}"):
    os.makedirs(f"{args.folder_name}")

l = []
s = []
xs = []
lbl = []
num_batch = args.num_batch
for i,ks in enumerate([3,5,7]):
    for j,act in enumerate(['elu','relu','tanh']):
        for k,pr in enumerate(['0.0', '0.375_structFalse', '0.375_structTrue', '0.625_structFalse', '0.625_structTrue']):
        # for k,pr in enumerate(['0.0']):
            for seed in range(5):
                with autocast():
                    ims_adv = ch.load("pgd_pt/{}_ims_adv_resnet9_seed{}_ks{}_act{}_prune{}.pt".format(args.attack, seed, ks, act, pr))
                    x_adv = ch.load("x_adv/{}_x_adv_resnet9_seed{}_ks{}_act{}_prune{}.pt".format(args.attack, seed, ks, act, pr))
                    # dt = ch.load("dt/{}_dt_resnet9_seed{}_ks{}_act{}_prune{}.pt".format(seed, ks, act, pr))
                    # d0 = ch.load("d0/{}_d0_resnet9_seed{}_ks{}_act{}_prune{}.pt".format(seed, ks, act, pr))
                    # score = dt + d0
                    l.append(ims_adv)
                    xs.append(x_adv)

                    # s.append(score[:num_batch*100,:].reshape([num_batch,100,3,32,32]))
                    labs = ch.Tensor([i,j,k]).repeat(len(x_adv),1)
                    lbl.append(labs)

l = ch.cat(l).detach()
xs = ch.cat(xs).detach()
lbl = ch.cat(lbl).detach()
print(l.shape)
print(xs.shape)
print(lbl.shape)
ch.save(l, f"{args.folder_name}/data.pt")
ch.save(xs, f"{args.folder_name}/xs.pt")
# ch.save(s, f"{args.folder_name}/score.pt")
ch.save(lbl, f"{args.folder_name}/label.pt")