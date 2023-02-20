from .convnet import ConvNet
from .resnet9 import ResNet9
from .resnet import resnet18
from .resnet_s import resnet20
from .vgg import VGG


def get_model(name, args):
    if name == "resnet9":
        return ResNet9(num_classes=args.num_classes,
                       kernel_size=args.kernel_size, act_func=args.act_func)
    elif name == "resnet18":
        return resnet18(args)
    elif name == "resnet20s":
        return resnet20(args)
    elif name == "vgg11":
        return VGG(args, 'VGG11')
    elif name == "vgg13":
        return VGG(args, 'VGG13')
    else:
        raise NotImplementedError(f"Arch {name} not Implemented!")