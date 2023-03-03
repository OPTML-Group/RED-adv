import argparse
import global_args as gargs


def general_args(parser):
    parser.add_argument('--seed', default=2, type=int, help='random seed')
    # parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    # parser.add_argument('--workers', type=int, default=4,
    #                     help='number of workers in dataloader')
    parser.add_argument('--rerun', action="store_true",
                        help="Do not load checkpoint")
    # parser.add_argument('--checkpoint', type=str,
    #                     default=None, help='checkpoint file')
    parser.add_argument('--save-dir', default="./results/", type=str,
                        help='The directory used to save the trained models')
    parser.add_argument('--tensorboard', action='store_true',
                        help='Using tensorboard during training.')
    parser.add_argument('--dataset', default="cifar10", type=str,
                        choices=gargs.VALID_DATASETS,
                        help='dataset name')
    parser.add_argument('--dataset-dir', type=str,
                        help='dataset dir')
    parser.add_argument('--ffcv-dir', type=str, default=None,
                        help='ffcv dir')


def training_args(parser):
    parser.add_argument('--batch-size', type=int,
                        default=256, help='batch size')
    parser.add_argument('--lr', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=5e-4,
                        type=float, help='weight decay')
    parser.add_argument('--epochs', default=75, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--save-freq', type=int,
                        default=15, help='num of epochs for saving checkpoint')
    parser.add_argument('--num-classes', type=int, default=10)
    parser.add_argument('--robust-train', action="store_true",
                        help="Robustive training")
    # parser.add_argument('--decreasing_lr', default='91,136',
    #                     help='decreasing strategy')


def pruning_args(parser):
    # parser.add_argument('--prune', type=str, default="omp",
    #                     help="method to prune")
    parser.add_argument('--pruning-ratio', type=float, default=0.0,
                        choices=[0.0, 0.375, 0.625],
                        help='pruning ratio')
    parser.add_argument('--structured-pruning', action="store_true",
                        help='whether using structured prune')
    parser.add_argument('--rewind-epoch', default=2,
                        type=int, help='rewind checkpoint')


def model_args(parser):
    parser.add_argument('--arch', type=str, default="resnet9",
                        choices=gargs.VALID_ARCHITECTURES)
    parser.add_argument('--act-func', type=str, default="relu",
                        choices=gargs.ACTIVATION_FUNCTIONS)
    parser.add_argument('--kernel-size', type=int,
                        default=3, choices=[1, 3, 5, 7]) # ks can be chosen from 1,3,5 for mnist and 3,5,7 for others. 
    parser.add_argument('--num_conv', type=int,
                        default=3, choices=[1, 3, 5]) # num_conv can be set for mnist
    parser.add_argument('--num_fc', type=int,
                        default=2, choices=[2, 3, 4]) # num_fc can be set for mnist


def attack_args(parser):
    parser.add_argument('--attack', type=str, default=None,
                        choices=["pgd", "pgdl2", "fgsm", "cw", "square", "autoattack", "zosignsgd", "zosgd", "nes"])
    parser.add_argument('--attack-save-dir', type=str, default="attack_img")
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--eps', type=float, default=8)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--norm', type=str, default="Linf",
                        choices=["Linf", "L2"])
    parser.add_argument('--n-queries', type=int, default=5000)
    parser.add_argument('--cw-c', type=int, default=1)
    parser.add_argument('--cw-kappa', type=float, default=0)


# def parse_args_model_parsing():
#     parser = argparse.ArgumentParser(
#         description='Model Parsing Experiments')
#     parser.add_argument('--input-type', type=str, default="delta",
#                         choices=["delta", "x_adv"], help='input type')
#     general_args(parser)
#     training_args(parser)

#     return parser.parse_args()


def parse_args_victim_training():
    parser = argparse.ArgumentParser(description='CIFAR-10 training')
    general_args(parser)
    training_args(parser)
    pruning_args(parser)
    model_args(parser)
    attack_args(parser)

    return parser.parse_args()

def parse_args_model_parsing(train):
    parser = argparse.ArgumentParser(description='train clf')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=100)

    parser.add_argument('--input_folder', type=str)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--attack', type=str, default='PGD_eps8_alpha1_steps10')
    parser.add_argument('--input-type', type=str, default='delta', choices=['x_adv', 'delta', 'denoise'])
    parser.add_argument('--save_folder', type=str)
    parser.add_argument('--attr-arch', type=str, choices=gargs.VALID_ATTR_ARCHS)

    parser.add_argument('--denoiser_pretrain_epoch', type=int, default=20)
    parser.add_argument('--denoiser_pretrain_lr', type=float, default=1e-2)
    parser.add_argument('--cotrain_epoch', type=int, default=50)
    parser.add_argument('--denoiser_cotrain_lr', type=float, default=1e-5)
    parser.add_argument('--parser_cotrain_lr', type=float, default=1e-3)
    # parser.add_argument('--lambda1', type=float, default=15,
    #                     help='the coefficient of attribution loss')
    parser.add_argument('--gamma1', type=float, default=1,
                        help='the coefficient of mae loss')
    parser.add_argument('--pretrained-denoiser-path', type=str, default='./pretrained_models/DO.pth.tar',
                        help='Path to a denoiser ')
    # test
    if not train:
        parser.add_argument('--log_dir', type=str, default=None)

    return parser.parse_args()