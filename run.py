from utils import run_commands
import os


workspace = "/localscratch2/ljcc"
atk_dir = os.path.join(workspace, "attack_img")
model_dir = os.path.join(workspace, "results")
parsing_dir = os.path.join(workspace, "parsing_models")

kernels = [3, 5, 7]
acts = ["relu", "tanh", "elu"]
ratios = [0.0, 0.375, 0.625]
struct = [True, False]

attacks = []
attacks += [
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
    {'attack': 'fgsm', "eps": 16},
    {'attack': 'fgsm', "eps": 32},
]
attacks += [
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

def get_atk_name(atk):
    dir_name = []
    for key, val in atk.items():
        dir_name.append(f"{key}_{val}")
    return '_'.join(dir_name)


def gen_commands():
    input_types = ["delta", "x_adv"]
    ext = "--batch-size 512 --tensorboard"

    commands = []
    for atk in attacks:
        for tp in input_types:
            akt_name = get_atk_name(atk)
            atk_path = os.path.join(atk_dir, akt_name)
            output_path = os.path.join(parsing_dir, akt_name, tp)

            command = f"python main_parsing.py --dataset-dir {atk_path} --input-type {tp} --save-dir {output_path} {ext}"
            commands.append(command)
    return commands


def gen_commands_old():
    input_types = ["delta", "x_adv"]

    commands = []
    for atk in attacks:
        for tp in input_types:
            akt_name = get_atk_name(atk)
            atk_path = os.path.join(atk_dir, akt_name)
            output_path = os.path.join(parsing_dir, akt_name, tp)
            if not os.path.exists(os.path.join(output_path, "final.pt")):
                command = f"python old_parser.py --input_folder {atk_path} --input-type {tp} --save_folder {output_path}"
                commands.append(command)
    return commands


def gen_commands_eval_old():
    input_types = ["delta", "x_adv"]

    commands = []
    for tp in input_types:
        for data_atk in attacks:
            for model_atk in attacks:
                data_atk_name = get_atk_name(data_atk)
                atk_path = os.path.join(atk_dir, data_atk_name)
                model_atk_name = get_atk_name(model_atk)
                output_path = os.path.join(parsing_dir, model_atk_name, tp)
                command = f"python old_eval_parser.py --input_folder {atk_path} --input-type {tp} --save_folder {output_path}"
                if not os.path.exists(os.path.join("/localscratch2/ljcc/test_log", f"data_{data_atk_name}___model_{model_atk_name}__{tp}.log")):
                    commands.append(command)
    return commands


def gen_commands_victim():
    commands = []
    # struct = [True]
    for atk in attacks:
        for k in kernels:
            for a in acts:
                for r in ratios:
                    for s in struct:
                        if r == 0.0 and s:
                            continue
                        command = f"python main_victim.py --kernel-size {k} --act-func {a} --pruning-ratio {r}"
                        command += f" --save-dir {model_dir}"
                        if s:
                            command += " --structured-pruning"

                        for key, val in atk.items():
                            command += f" --{key} {val}"

                        akt_name = get_atk_name(atk)
                        atk_path = os.path.join(atk_dir, akt_name)
                        command += f" --attack-save-dir {atk_path}"

                        model_name = "seed{}_kernel{}_act{}_prune{}".format(
                            2, k, a, r)
                        if s:
                            model_name += "_struct"

                        if not os.path.exists(os.path.join(atk_path, model_name)):
                            commands.append(command)
    return commands


if __name__ == "__main__":
    commands = gen_commands_eval_old()
    print(len(commands))
    run_commands(list(range(6)) * 8, commands, call=True,
                 suffix="commands", shuffle=False, delay=0.5)

    # commands = gen_commands_victim()
    # print(len(commands))
    # run_commands([0, 1, 4, 5, 6, 7] * 4, commands, call=True, ext_command=" --dataset-dir /localscratch2/tmp/cifar{i}",
    #              suffix="commands", shuffle=False, delay=1)
