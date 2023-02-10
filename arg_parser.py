import argparse


def parse_args_model_parsing():
    parser = argparse.ArgumentParser(
        description='Model Parsing Experiments')

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
        '--save_dir', help='The directory used to save the trained models', default="./results/", type=str)
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
    parser.add_argument('--tensorboard', action='store_true', default=False,
                        help='Using tensorboard during training.')

    parser.add_argument('--dataset-dir', type=str,
                        default=None, help='dataset dir')
    parser.add_argument('--input-type', type=str,
                        default="delta", help='dataset dir')

    return parser.parse_args()


def parse_args_victim_training():
    parser = argparse.ArgumentParser(description='CIFAR-10 training')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dir', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=24)
    parser.add_argument('--lr', type=float, default=0.5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--rewind_epoch', type=int, default=2)
    parser.add_argument('--structured_pruning', action="store_true")
    
    parser.add_argument('--act_func', type=str, default="relu",
                        choices=["relu", "tanh", "elu"])
    parser.add_argument('--kernel_size', type=int, default=3, choices=[3, 5, 7])
    parser.add_argument('--pruning_ratio', type=float, default=0.0,
                        choices=[0.0, 0.375, 0.625])
    
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--eps', type=float, default=8/255)
    return parser.parse_args()
