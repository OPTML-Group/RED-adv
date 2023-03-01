import global_args as gargs


def get_model_name(seed, kernel_size, activation_function, pruning_ratio, struct=False, robust=False):
    assert kernel_size in gargs.KERNEL_SIZES
    assert activation_function in gargs.ACTIVATION_FUNCTIONS
    assert pruning_ratio in gargs.PRUNING_RATIOS

    name = f"seed{seed}_kernel{kernel_size}_act{activation_function}_prune{pruning_ratio}"
    if struct:
        name += "_struct"
    if robust:
        name += "_robust"
    return name


def get_attack_name(atk):
    assert atk in gargs.ALL_ATTACKS, f"Attack not found! {atk}"

    dir_name = []
    for key, val in atk.items():
        dir_name.append(f"{key}_{val}")
    return '_'.join(dir_name)
