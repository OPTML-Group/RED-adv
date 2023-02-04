import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='CIFAR-10 training')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dir', type=str, default='/tmp')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--act_func', type=str,choices= ["relu", "tanh", "elu"])
    parser.add_argument('--kernel_size', type=int, choices= [3, 5, 7])
    parser.add_argument('--pruning_ratio', type=float, choices = [0.0, 0.375, 0.625])
    parser.add_argument('--epochs', type=int, default = 24)
    parser.add_argument('--lr', type=float, default = 0.5)
    parser.add_argument('--momentum', type=float, default = 0.9)
    parser.add_argument('--weight_decay', type=float, default = 5e-4)
    parser.add_argument('--rewind_epoch', type=int, default = 2)
    parser.add_argument('--structured_pruning', default=False, action=argparse.BooleanOptionalAction)
    return parser.parse_args()
