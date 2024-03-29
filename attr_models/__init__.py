from .conv2 import AttrNet, ConvNet2
from .conv4 import ConvNet4
from .dncnn import DnCNN
from .lenet import LeNet
from .mlp import MLP
from .resnset9 import ResNet9


def get_model(name, num_channel, num_class, num_output, img_size):
    if name == "mlp":
        return MLP(num_channel, num_class, num_output, img_size)
    elif name == "lenet":
        return LeNet(num_channel, num_class, num_output, img_size)
    elif name == "attrnet":
        return AttrNet(num_channel, num_class, num_output, img_size)
    elif name == "conv4":
        return ConvNet4(num_channel, num_class, num_output, img_size)
    elif name == "conv2":
        return ConvNet2(num_channel, num_class, num_output, img_size)
    elif name == "resnet9":
        return ResNet9(num_channel, num_class, num_output)
    else:
        raise NotImplementedError(name)
