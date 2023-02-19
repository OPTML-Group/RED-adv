import os

WORKSPACE2_DIR = "/localscratch2/ljcc"
WORKSPACE_DIR = "/localscratch/ljcc"

ATK_DIR = os.path.join(WORKSPACE2_DIR, "attack_img")
MODEL_DIR = os.path.join(WORKSPACE2_DIR, "results")
PARSING_DIR = os.path.join(WORKSPACE2_DIR, "parsing_models")
PARSING_LOG_DIR = os.path.join(WORKSPACE2_DIR, "test_log")

GREP_DIR = os.path.join(WORKSPACE_DIR, "grep_datas")


KERNEL_SIZES = [3, 5, 7]
ACTIVATION_FUNCTIONS = ["relu", "tanh", "elu"]
PRUNING_RATIOS = [0.0, 0.375, 0.625]


ATTACKS = []
ATTACKS += [
    {'attack': 'pgd', "eps": 4, "alpha": 0.5},
    {'attack': 'pgd', "eps": 8, "alpha": 1},
    {'attack': 'pgd', "eps": 12, "alpha": 2},
    {'attack': 'pgd', "eps": 16, "alpha": 2},
    {'attack': 'pgdl2', "eps": 0.125, "alpha": 0.025},
    {'attack': 'pgdl2', "eps": 0.25, "alpha": 0.05},
    {'attack': 'pgdl2', "eps": 0.5, "alpha": 0.1},
    {'attack': 'pgdl2', "eps": 1, "alpha": 0.2},
    {'attack': 'pgdl2', "eps": 1.5, "alpha": 0.3},
    {'attack': 'fgsm', "eps": 4},
    {'attack': 'fgsm', "eps": 8},
    {'attack': 'fgsm', "eps": 12},
    {'attack': 'fgsm', "eps": 16},
]
ATTACKS += [
    {'attack': 'square', "eps": 4, "norm": 'Linf'},
    {'attack': 'square', "eps": 8, "norm": 'Linf'},
    {'attack': 'square', "eps": 12, "norm": 'Linf'},
    {'attack': 'square', "eps": 16, "norm": 'Linf'},
    {'attack': 'square', "eps": 0.25, "norm": 'L2'},
    {'attack': 'square', "eps": 0.5, "norm": 'L2'},
    {'attack': 'square', "eps": 0.75, "norm": 'L2'},
    {'attack': 'square', "eps": 1.0, "norm": 'L2'},
    {'attack': 'autoattack', "eps": 4, "norm": 'Linf'},
    {'attack': 'autoattack', "eps": 8, "norm": 'Linf'},
    {'attack': 'autoattack', "eps": 12, "norm": 'Linf'},
    {'attack': 'autoattack', "eps": 16, "norm": 'Linf'},
    {'attack': 'autoattack', "eps": 0.25, "norm": 'L2'},
    {'attack': 'autoattack', "eps": 0.5, "norm": 'L2'},
    {'attack': 'autoattack', "eps": 0.75, "norm": 'L2'},
    {'attack': 'autoattack', "eps": 1.0, "norm": 'L2'},
    {'attack': 'cw', "cw-c": 1, "cw-kappa": 0},
    {'attack': 'cw', "cw-c": 1, "cw-kappa": 0.1},
    {'attack': 'cw', "cw-c": 1, "cw-kappa": 1},
    {'attack': 'cw', "cw-c": 10, "cw-kappa": 0},
    {'attack': 'cw', "cw-c": 10, "cw-kappa": 0.1},
    {'attack': 'cw', "cw-c": 10, "cw-kappa": 1},
    {'attack': 'cw', "cw-c": 100, "cw-kappa": 0},
    {'attack': 'cw', "cw-c": 100, "cw-kappa": 0.1},
    {'attack': 'cw', "cw-c": 100, "cw-kappa": 1},
]
ATTACKS += [
    {'attack': 'zosignsgd', 'eps': 4, 'norm': 'Linf'},
    {'attack': 'zosignsgd', 'eps': 8, 'norm': 'Linf'},
    {'attack': 'zosignsgd', 'eps': 12, 'norm': 'Linf'},
    {'attack': 'zosignsgd', 'eps': 16, 'norm': 'Linf'},
    {'attack': 'zosgd', 'eps': 4, 'norm': 'Linf'},
    {'attack': 'zosgd', 'eps': 8, 'norm': 'Linf'},
    {'attack': 'zosgd', 'eps': 12, 'norm': 'Linf'},
    {'attack': 'zosgd', 'eps': 16, 'norm': 'Linf'},
    {'attack': 'nes', 'eps': 4, 'norm': 'Linf'},
    {'attack': 'nes', 'eps': 8, 'norm': 'Linf'},
    {'attack': 'nes', 'eps': 12, 'norm': 'Linf'},
    {'attack': 'nes', 'eps': 16, 'norm': 'Linf'},
]