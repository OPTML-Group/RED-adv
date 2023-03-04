import itertools

PGD_ATTACKS = [
    {'attack': 'pgd', "eps": 4, "alpha": 0.5},
    {'attack': 'pgd', "eps": 8, "alpha": 1},
    {'attack': 'pgd', "eps": 12, "alpha": 2},
    {'attack': 'pgd', "eps": 16, "alpha": 2},
]
PGD_L2_ATTACKS = [
    {'attack': 'pgdl2', "eps": 0.25, "alpha": 0.05},
    {'attack': 'pgdl2', "eps": 0.5, "alpha": 0.1},
    {'attack': 'pgdl2', "eps": 0.75, "alpha": 0.15},
    {'attack': 'pgdl2', "eps": 1.0, "alpha": 0.2},
]
FGSM_ATTACKS = [
    {'attack': 'fgsm', "eps": 4},
    {'attack': 'fgsm', "eps": 8},
    {'attack': 'fgsm', "eps": 12},
    {'attack': 'fgsm', "eps": 16},
]
CW_ATTACKS = [
    {'attack': 'cw', "cw-c": 0.1, "cw-kappa": 0},
    {'attack': 'cw', "cw-c": 1, "cw-kappa": 0},
    {'attack': 'cw', "cw-c": 10, "cw-kappa": 0},

    # {'attack': 'cw', "cw-c": 10, "cw-kappa": 0},
    # {'attack': 'cw', "cw-c": 10, "cw-kappa": 0.1},
    # {'attack': 'cw', "cw-c": 10, "cw-kappa": 1},
    # {'attack': 'cw', "cw-c": 100, "cw-kappa": 0},
    # {'attack': 'cw', "cw-c": 100, "cw-kappa": 0.1},
    # {'attack': 'cw', "cw-c": 100, "cw-kappa": 1},
]
AUTO_LINF_ATTACKS = [
    {'attack': 'autoattack', "eps": 4, "norm": 'Linf'},
    {'attack': 'autoattack', "eps": 8, "norm": 'Linf'},
    {'attack': 'autoattack', "eps": 12, "norm": 'Linf'},
    {'attack': 'autoattack', "eps": 16, "norm": 'Linf'},
]
AUTO_L2_ATTACKS = [
    {'attack': 'autoattack', "eps": 0.25, "norm": 'L2'},
    {'attack': 'autoattack', "eps": 0.5, "norm": 'L2'},
    {'attack': 'autoattack', "eps": 0.75, "norm": 'L2'},
    {'attack': 'autoattack', "eps": 1.0, "norm": 'L2'},
]
SQUARE_LINF_ATTACKS = [
    {'attack': 'square', "eps": 4, "norm": 'Linf'},
    {'attack': 'square', "eps": 8, "norm": 'Linf'},
    {'attack': 'square', "eps": 12, "norm": 'Linf'},
    {'attack': 'square', "eps": 16, "norm": 'Linf'},
]
SQUARE_L2_ATTACKS = [
    {'attack': 'square', "eps": 0.25, "norm": 'L2'},
    {'attack': 'square', "eps": 0.5, "norm": 'L2'},
    {'attack': 'square', "eps": 0.75, "norm": 'L2'},
    {'attack': 'square', "eps": 1.0, "norm": 'L2'},
]
ZOSIGNSGD_ATTACKS = [
    {'attack': 'zosignsgd', 'eps': 4, 'norm': 'Linf'},
    {'attack': 'zosignsgd', 'eps': 8, 'norm': 'Linf'},
    {'attack': 'zosignsgd', 'eps': 12, 'norm': 'Linf'},
    {'attack': 'zosignsgd', 'eps': 16, 'norm': 'Linf'},
]
ZOSGD_ATTACKS = [
    {'attack': 'zosgd', 'eps': 4, 'norm': 'Linf'},
    {'attack': 'zosgd', 'eps': 8, 'norm': 'Linf'},
    {'attack': 'zosgd', 'eps': 12, 'norm': 'Linf'},
    {'attack': 'zosgd', 'eps': 16, 'norm': 'Linf'},
]
NES_ATTACKS = [
    {'attack': 'nes', 'eps': 4, 'norm': 'Linf'},
    {'attack': 'nes', 'eps': 8, 'norm': 'Linf'},
    {'attack': 'nes', 'eps': 12, 'norm': 'Linf'},
    {'attack': 'nes', 'eps': 16, 'norm': 'Linf'},
]
WHITEBOX_GROUP = [PGD_ATTACKS, PGD_L2_ATTACKS, FGSM_ATTACKS, CW_ATTACKS]
ENSEMBLE_GROUP = [AUTO_LINF_ATTACKS, AUTO_L2_ATTACKS]
BLACKBOX_GROUP = [SQUARE_LINF_ATTACKS, SQUARE_L2_ATTACKS,
                  ZOSIGNSGD_ATTACKS, NES_ATTACKS]  # , ZOSGD_ATTACKS]
ALL_GROUP = WHITEBOX_GROUP + ENSEMBLE_GROUP + BLACKBOX_GROUP

WHITEBOX_ATTACKS = list(itertools.chain(*WHITEBOX_GROUP))
ENSEMBLE_ATTACKS = list(itertools.chain(*ENSEMBLE_GROUP))
BLACKBOX_ATTACKS = list(itertools.chain(*BLACKBOX_GROUP))

ALL_ATTACKS = WHITEBOX_ATTACKS + ENSEMBLE_ATTACKS + BLACKBOX_ATTACKS

L2_ATTACKS = list(filter(lambda x: x.get('norm') ==
                  'L2' or 'l2' in x['attack'], ALL_ATTACKS))
LINF_ATTACKS = list(filter(lambda x: x.get('norm') !=
                    'L2' and 'l2' not in x['attack'], ALL_ATTACKS))
