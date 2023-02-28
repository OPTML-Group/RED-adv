from . import attack as _atk

KERNEL_SIZES = [3, 5, 7]
ACTIVATION_FUNCTIONS = ["relu", "tanh", "elu"]
PRUNING_RATIOS = [0.0, 0.375, 0.625]


VALID_DATASETS = ["cifar10", "cifar100", "tinyimagenet", "mnist"]
VALID_ARCHITECTURES = ['resnet9', 'resnet20s', 'resnet18', 'vgg11', 'vgg13']#, 'lenet']

VALID_SETTINGS = ['origin', 'robust', 'robust_all']
VALID_ATTR_ARCHS = ["mlp", "lenet", "attrnet", "conv2", "conv4", "resnet9"]


def _get_exps():
    archs = VALID_ARCHITECTURES
    datasets = VALID_DATASETS
    settings = VALID_SETTINGS
    default_arch = archs[0]
    default_data = datasets[0]
    default_setting = settings[0]

    # default
    exp_combs = [dict(
        arch=default_arch,
        data=default_data,
        setting=default_setting,
        attacks=_atk.ALL_ATTACKS
    )]
    # arch ablation
    for arch in archs[1:]:
        if arch == 'lenet':
            continue
        s = dict(
            arch=arch,
            data=default_data,
            setting=default_setting,
            attacks=_atk.WHITEBOX_ATTACKS
        )
        exp_combs.append(s)
    # robust ablation
    for setting in settings[1:]:
        s = dict(
            arch=default_arch,
            data=default_data,
            setting=setting,
            attacks=_atk.WHITEBOX_ATTACKS
        )
        exp_combs.append(s)
    # dataset ablation
    for data in datasets[1:]:
        if data == "tinyimagenet":
            arch = "resnet18"
        elif data == "mnist":
            continue
        else:
            arch = default_arch
        s = dict(
            arch=arch,
            data=data,
            setting=default_setting,
            attacks=_atk.WHITEBOX_ATTACKS
        )
        exp_combs.append(s)
    return exp_combs


EXPS = _get_exps()
