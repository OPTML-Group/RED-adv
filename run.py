from utils import run_commands
import os

import global_args as gargs

_atk_dir = gargs.ATK_DIR
_model_dir = gargs.MODEL_DIR
_parsing_dir = gargs.PARSING_DIR
_grep_dir = gargs.GREP_DIR
_log_dir = gargs.PARSING_LOG_DIR

kernels = gargs.KERNEL_SIZES
acts = gargs.ACTIVATION_FUNCTIONS
ratios = gargs.PRUNING_RATIOS
struct = [False, True]

_attacks = gargs.ATTACKS


def get_attack_name(atk):
    dir_name = []
    for key, val in atk.items():
        dir_name.append(f"{key}_{val}")
    return '_'.join(dir_name)


def gen_commands():
    input_types = ["delta", "x_adv"]
    ext = "--batch-size 512 --tensorboard"

    commands = []
    for atk in _attacks:
        for tp in input_types:
            akt_name = get_attack_name(atk)
            atk_path = os.path.join(_atk_dir, akt_name)
            output_path = os.path.join(_parsing_dir, akt_name, tp)

            command = f"python main_parsing.py --dataset-dir {atk_path} --input-type {tp} --save-dir {output_path} {ext}"
            commands.append(command)
    return commands


def gen_commands_old(setting):
    input_types = ["delta", "x_adv"]

    commands = []
    for atk in _attacks:
        for tp in input_types:
            akt_name = get_attack_name(atk)

            grep_path = os.path.join(_grep_dir, "cifar10_resnet9", setting, akt_name)
            output_path = os.path.join(_parsing_dir, "cifar10_resnet9", setting, akt_name, tp)

            if not os.path.exists(grep_path):
                continue

            if not os.path.exists(os.path.join(output_path, "final.pt")):
                command = f"python old_parser.py --input_folder {grep_path} --input-type {tp} --save_folder {output_path}"
                commands.append(command)
    return commands


def gen_commands_eval_old(setting):
    input_types = ["delta", "x_adv"]

    commands = []
    for tp in input_types:
        for data_atk in _attacks:
            for model_atk in _attacks:
                data_atk_name = get_attack_name(data_atk)
                atk_path = os.path.join(_grep_dir, "cifar10_resnet9", setting, data_atk_name)
                model_atk_name = get_attack_name(model_atk)
                output_path = os.path.join(_parsing_dir, "cifar10_resnet9", setting, model_atk_name, tp)
                log_dir = os.path.join(_log_dir, "cifar10_resnet9", setting)
                command = f"python old_eval_parser.py --input_folder {atk_path} --input-type {tp} --save_folder {output_path} --log_dir {log_dir}"
                if not os.path.exists(os.path.join(log_dir, f"data_{data_atk_name}___model_{model_atk_name}__{tp}.log")):
                    commands.append(command)
    return commands


def gen_commands_victim(robust=True):
    commands = []
    # struct = [True]
    for idx, atk in enumerate(_attacks):
        for k in kernels:
            for a in acts:
                for s in struct:
                    for r in ratios:
                        if r == 0.0 and s:
                            continue
                        command = f"python main_victim.py --kernel-size {k} --act-func {a} --pruning-ratio {r} --tensorboard"
                        command += f" --save-dir {_model_dir}"
                        if s:
                            command += " --structured-pruning"
                        if robust:
                            command += ' --robust-train'

                        for key, val in atk.items():
                            command += f" --{key} {val}"

                        akt_name = get_attack_name(atk)
                        atk_path = os.path.join(_atk_dir, akt_name)
                        command += f" --attack-save-dir {atk_path}"

                        model_name = "seed{}_kernel{}_act{}_prune{}".format(
                            2, k, a, r)
                        if s:
                            model_name += "_struct"
                        if robust:
                            model_name += '_robust'

                        # commands.append(command)

                        if not os.path.exists(os.path.join(atk_path, model_name, 'ori_pred.pt')) and \
                            not os.path.exists(os.path.join(atk_path, model_name, 'x_adv.pt')):
                            path = f"/localscratch2/ljcc/results/{model_name}_omp_2/checkpoint_75.pt"
                            # print(path)
                            if idx == 0 or idx > 0 and os.path.exists(path):
                                commands.append(command)
    return commands


if __name__ == "__main__":
    debug = False
    # commands = gen_commands_old(setting="origin")
    # print(len(commands))
    # commands = gen_commands_old(setting="robust")
    # print(len(commands))
    # commands = gen_commands_old(setting="robust_all")
    # print(len(commands))
    # # exit(0)
    # run_commands([1, 2, 3, 4, 5, 6, 7, 0] * 4 if not debug else [0], commands, call=not debug,
    #              suffix="commands", shuffle=False, delay=0.5)
    # commands = gen_commands_eval_old(setting="robust")
    # commands += gen_commands_eval_old(setting="robust_all")
    # commands = gen_commands_eval_old(setting="origin")
    # print(len(commands))
    # run_commands([1, 2, 3, 4, 5, 6, 7, 0] * 6 if not debug else [0], commands, call=not debug,
    #              suffix="commands", shuffle=False, delay=0.5)

    commands = gen_commands_victim(robust=False)
    print(len(commands))
    run_commands([1, 2, 3, 4, 5, 6, 7, 0] * 2 if not debug else [0], commands, call=not debug, 
                 ext_command=" --dataset-dir /localscratch2/tmp/cifar{i}",
                 suffix="commands", shuffle=False, delay=1)
