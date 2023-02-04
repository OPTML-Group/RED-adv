import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch Lottery Tickets Experiments')

    ##################################### General setting ############################################
    # parser.add_argument('--seed', default=2, type=int, help='random seed')
    # parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    # parser.add_argument('--workers', type=int, default=4,
    #                     help='number of workers in dataloader')
    # parser.add_argument('--resume', action="store_true",
    #                     help="resume from checkpoint")
    # parser.add_argument('--checkpoint', type=str,
    #                     default=None, help='checkpoint file')
    parser.add_argument(
        '--save_dir', help='The directory used to save the trained models', default="./", type=str)
    # parser.add_argument('--mask', type=str, default=None, help='sparse model')

    ##################################### Training setting #################################################
    parser.add_argument('--batch_size', type=int,
                        default=512, help='batch size')
    parser.add_argument('--lr', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=5e-4,
                        type=float, help='weight decay')
    parser.add_argument('--epochs', default=50, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--print_freq', default=100,
                        type=int, help='print frequency')
    parser.add_argument('--decreasing_lr', default='91,136',
                        help='decreasing strategy')
    parser.add_argument('--no-aug', action='store_true', default=False,
                        help='No augmentation in training dataset (transformation).')

    parser.add_argument('--dataset-dir', type=str,
                        default=None, help='dataset dir')
    parser.add_argument('--input-type', type=str,
                        default="delta", help='dataset dir')

    return parser.parse_args()
