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
import torchattacks

import utils
import models


from pruner.pruner import extract_mask, prune_model_custom, pruning_model, pruning_model_structured, remove_prune, check_sparsity
# from PGD import PGD

start = time.time()

parser = argparse.ArgumentParser(description='CIFAR-10 training')
# parser.add_argument('--num_resnet_layer', type=int, default=20)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--activation_function', type=str,
                    choices=["relu", "tanh", "elu"])
parser.add_argument('--kernel_size', type=int, choices=[3, 5, 7])
parser.add_argument('--pruning_ratio', type=float,
                    default=0, choices=[0.0, 0.375, 0.625])
parser.add_argument('--rewind_epoch', type=int, default=2)
parser.add_argument('--steps', type=int, default=10)
parser.add_argument('--eps', type=float, default=8/255)
parser.add_argument('--dataset', type=str, default="cifar10")
parser.add_argument('--times', type=int, default=1)

# parser.add_argument('--structured_pruning', default=True, action=argparse.BooleanOptionalAction)
args = parser.parse_args()

utils.set_seed(args.seed)


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
    label_pipeline: List[Operation] = [
        IntDecoder(), ToTensor(), ToDevice('cuda:0'), Squeeze()]
    image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

    # Add image transforms and normalization
    image_pipeline.extend([
        ToTensor(),
        ToDevice('cuda:0', non_blocking=True),
        ToTorchImage(),
        Convert(ch.float16),
        torchvision.transforms.Normalize(
            CIFAR_MEAN, CIFAR_STD),  # type: ignore
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




NUM_CLASSES = 10


scaler = GradScaler()
criterion = CrossEntropyLoss()


total_correct, total_num = 0., 0.
if not os.path.exists("pgd_pt"):
    os.makedirs("pgd_pt")
if not os.path.exists("labs"):
    os.makedirs("labs")

for ks in [3, 5, 7]:
    for act in ['elu', 'relu', 'tanh']:
        for pr in ['0.0',  '0.375_structFalse', '0.375_structTrue', '0.625_structFalse', '0.625_structTrue']:
            print(f"{ks},{act},{pr}")
            args.kernel_size = ks
            args.activation_function = act
            model = models.ResNet9(
                num_classes=NUM_CLASSES, kernel_size=ks, act_func=act)

            # type: ignore
            model = model.to(memory_format=ch.channels_last).cuda()

            args.seed = 0
            for i in [0]:
                model.load_state_dict(ch.load(
                    "model_pt_{}/resnet9_seed{}_ks{}_act{}_prune{}.pt".format(args.dataset, i, ks, act, pr)))
                # model.eval()
                CIFAR_MEAN_1 = [125.307/255, 122.961/255, 113.8575/255]
                CIFAR_STD_1 = [51.5865/255, 50.847/255, 51.255/255]
                atk = torchattacks.PGD(
                    model, eps=8/255, alpha=2/255, steps=10)  # PGD_linf
                atk = torchattacks.CW(model)  # CW
                # PGD_l2, view -> reshape
                atk = torchattacks.PGDL2(model, eps=0.5, alpha=0.1, steps=10)
                atk = torchattacks.AutoAttack(
                    model, norm='Linf', eps=8/255)  # AutoAttack
                atk = torchattacks.FGSM(model, eps=8/255)  # FGSM (l_inf)
                # normalization if data is already normalized
                atk.set_normalization_used(mean=CIFAR_MEAN_1, std=CIFAR_STD_1)
                for ims, labs in loaders['test']:
                    with autocast():
                        ims_adv = atk(ims, labs)
                        ori_out = model(ims)
                        adv_out = model(ims_adv)  # Test-time augmentation

                        idx = ori_out.argmax(1).eq(
                            labs) * adv_out.argmax(1).ne(labs)
                        idx_adv = adv_out.argmax(1).ne(labs)
                        print("clf correct & atk rate",
                              sum(idx)/len(idx)*100, '%')
                        print("atk rate:", sum(idx_adv)/len(idx)*100, '%')
                        # idx = adv_out.argmax(1).ne(labs)
                        ims_adv_save = (ims_adv-ims)[idx]*args.times
                        labs = labs[idx]
                        exit(0)
                        # ch.save(ims_adv_save,"pgd_pt/linf_ims_adv_resnet9_seed{}_ks{}_act{}_prune{}.pt".format(args.seed, ks, act, pr))
                        # ch.save(ims_adv[idx],"x_adv/linf_x_adv_resnet9_seed{}_ks{}_act{}_prune{}.pt".format(args.seed, ks, act, pr))
                        # ch.save(labs,"labs/linf_lab_resnet9_seed{}_ks{}_act{}_prune{}.pt".format(args.seed, ks, act, pr))
                        # ch.save(dt, "dt/dt_resnet9_seed{}_ks{}_act{}_prune{}.pt".format(args.seed, ks, act, pr))
                        # ch.save(d0, "d0/d0_resnet9_seed{}_ks{}_act{}_prune{}.pt".format(args.seed, ks, act, pr))
                        # args.seed += 1
