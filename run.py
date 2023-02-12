from utils import run_commands

attacks = [
    {'attack': 'pgd', "eps": 4, "alpha": 0.5},
    {'attack': 'pgd', "eps": 8, "alpha": 1},
    {'attack': 'pgd', "eps": 12, "alpha": 2},
    {'attack': 'pgd', "eps": 16, "alpha": 2},
    {'attack': 'fgsm', "eps": 8},
    {'attack': 'fgsm', "eps": 16},
    {'attack': 'pgdl2', "eps": 0.5, "alpha": 0.1},
    {'attack': 'pgdl2', "eps": 1, "alpha": 0.2},
    {'attack': 'autoattack', "eps": 8, "norm": 'Linf'},
    {'attack': 'cw'},
    {'attack': 'square', "eps": 8}
]


def gen_commands():
    epss = [0.3, 0.6]
    steps = [1, 2, 4, 6, 8, 10]

    commands = []
    for eps in epss:
        for step in steps:
            dataset_dir = f"./RED/PGD{step}_{eps}"
            output_path = f"./result/RED_PGD{step}_{eps}.log"
            command = f"python main.py --dataset-dir {dataset_dir} --input-type x_adv > {output_path}"
            commands.append(command)
    return commands


def gen_commands_victim():
    commands = []

    kernels = [3, 5, 7]
    acts = ["relu", "tanh", "elu"]
    ratios = [0.0, 0.375, 0.625]
    # struct = [True, False]
    struct = [True]
    for atk in attacks:
        for k in kernels:
            for a in acts:
                for r in ratios:
                    for s in struct:
                        if r == 0.0 and s:
                            continue
                        command = f"python main_victim.py --kernel-size {k} --act-func {a} --pruning-ratio {r}"
                        if s:
                            command += " --structured-pruning"
                        dir_name = []
                        for key, val in atk.items():
                            command += f" --{key} {val}"
                            dir_name.append(f"{key}_{val}")
                        atk_path = f"attack_img/{'_'.join(dir_name)}"
                        command += f" --attack-save-dir {atk_path}"
                        commands.append(command)
    return commands


if __name__ == "__main__":
    # commands = gen_commands()
    # run_commands([0, 1] * 2, commands, call=True,
    #              dir="commands", shuffle=False, delay=0.5)

    commands = gen_commands_victim()
    run_commands([4, 5, 6, 7] * 5, commands, call=True, ext_command=" --dataset-dir /tmp/cifar{i}",
                 suffix="commands", shuffle=False, delay=1)
