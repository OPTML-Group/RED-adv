import torch.nn as nn


def conv_block(in_channels, out_channels, kernel_size, act_func):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, 1),
        nn.BatchNorm2d(out_channels),
        act_func(),
    )


def fc_block(in_channels, out_channels, act_func):
    return nn.Sequential(
        nn.Linear(in_channels, out_channels),
        act_func(),
    )


_act_func_dict = {"relu": nn.ReLU, "tanh": nn.Tanh, "silu": nn.SiLU, "elu": nn.ELU}


class LeNet(nn.Module):
    def __init__(self, args):
        super(LeNet, self).__init__()
        self.num_conv = args.num_conv
        self.ks = args.kernel_size
        self.num_fc = args.num_fc

        self.act_func = _act_func_dict[args.act_func]
        size = 28 - (self.ks - 1) * self.num_conv
        layers = []
        last_channel = 1
        _conv_hidden_channel = 32
        _fc_hidden_channel = 64

        for _ in range(self.num_conv):
            layers.append(
                conv_block(last_channel, _conv_hidden_channel, self.ks, self.act_func)
            )
            last_channel = _conv_hidden_channel

        layers.append(Flatten())
        last_channel = last_channel * size**2

        for _ in range(self.num_fc - 1):
            layers.append(fc_block(last_channel, _fc_hidden_channel, self.act_func))
            last_channel = _fc_hidden_channel

        layers.append(fc_block(last_channel, 10, self.act_func))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


# if __name__ == "__main__":
#     class Args: pass
#     args = Args()
#     args.num_conv = 2
#     args.kernel_size = 5
#     args.num_fc = 3
#     args.act_func = "relu"
#     model = LeNet(args)
#     a = model.forward(torch.zeros([5, 1, 28, 28]))
#     print(a.shape)
