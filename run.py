from utils import run_commands
import os

import global_args as gargs
from training_utils import get_attack_name, get_model_name


_kernels = gargs.KERNEL_SIZES
_acts = gargs.ACTIVATION_FUNCTIONS
_ratios = gargs.PRUNING_RATIOS
_struct = [False]
_input_types = ["delta", "x_adv"]

def gen_commands_train_victim(dataset, arch, robust):
    _data_arch_name = f"{dataset}_{arch}"

    _model_dir = os.path.join(gargs.MODEL_DIR, _data_arch_name)

    commands = []
    for k in _kernels:
        for a in _acts:
            for s in _struct:
                for r in _ratios:
                    if r == 0.0 and s:
                        continue
                    command = f"python main_victim.py --kernel-size {k} --act-func {a} --pruning-ratio {r} --tensorboard"
                    command += f" --dataset-dir {gargs.DATASET_DIRS[dataset]}"
                    command += f" --num-classes {gargs.DATASET_NUM_CLASSES[dataset]}"
                    command += f" --dataset {dataset}"
                    command += f" --save-dir {_model_dir}"
                    command += f" --arch {arch}"
                    command += gargs.TRAINING_ARGS[dataset]
                    if s:
                        command += " --structured-pruning"
                    if robust:
                        command += ' --robust-train'


                    model_name = get_model_name(2, k, a, r, s, robust)

                    path = os.path.join(
                        _model_dir, f"{model_name}_omp_2/checkpoint_75.pt")
                    # commands.append(command)
                    if not os.path.exists(path):
                        commands.append(command)
    return commands

def gen_commands_attack_victim(dataset, arch, attacks, robust):
    _data_arch_name = f"{dataset}_{arch}"

    _atk_dir = os.path.join(gargs.ATK_DIR, _data_arch_name)
    _model_dir = os.path.join(gargs.MODEL_DIR, _data_arch_name)

    commands = []
    for idx, atk in enumerate(attacks):
        for k in _kernels:
            for a in _acts:
                for s in _struct:
                    for r in _ratios:
                        if r == 0.0 and s:
                            continue
                        command = f"python main_victim.py --kernel-size {k} --act-func {a} --pruning-ratio {r} --tensorboard"
                        command += f" --dataset-dir {gargs.DATASET_DIRS[dataset]}"
                        command += f" --num-classes {gargs.DATASET_NUM_CLASSES[dataset]}"
                        command += f" --dataset {dataset}"
                        command += f" --save-dir {_model_dir}"
                        command += f" --arch {arch}"
                        command += gargs.TRAINING_ARGS[dataset]
                        if s:
                            command += " --structured-pruning"
                        if robust:
                            command += ' --robust-train'

                        for key, val in atk.items():
                            command += f" --{key} {val}"

                        akt_name = get_attack_name(atk)
                        atk_path = os.path.join(_atk_dir, akt_name)
                        command += f" --attack-save-dir {atk_path}"

                        model_name = get_model_name(2, k, a, r, s, robust)

                        path = os.path.join(
                            _model_dir, f"{model_name}_omp_2/checkpoint_75.pt")
                        # commands.append(command)
                        if os.path.exists(path):
                            if not (os.path.exists(os.path.join(atk_path, model_name, 'ori_pred.pt')) and
                                    os.path.exists(os.path.join(atk_path, model_name, 'attack_acc.log'))):
                                if idx == 0 or idx > 0 and os.path.exists(path):
                                    commands.append(command)
    return commands


def gen_commands_parsing(exp, attr_arch, specific_type=None):
    dataset = exp['data']
    arch = exp['arch']
    setting = exp['setting']
    attacks = exp['attacks']
    _data_arch_name = f"{dataset}_{arch}"

    _atk_dir = os.path.join(gargs.ATK_DIR, _data_arch_name)
    _model_dir = os.path.join(gargs.MODEL_DIR, _data_arch_name)
    _parsing_dir = os.path.join(gargs.PARSING_DIR, attr_arch, _data_arch_name)
    _grep_dir = os.path.join(gargs.GREP_DIR, _data_arch_name)
    _log_dir = os.path.join(gargs.PARSING_LOG_DIR, attr_arch, _data_arch_name)

    input_types = [specific_type] if specific_type else _input_types

    commands = []
    for atk in attacks:
        for tp in input_types:
            akt_name = get_attack_name(atk)

            grep_path = os.path.join(_grep_dir, setting, akt_name)
            output_path = os.path.join(_parsing_dir, setting, akt_name, tp)

            if not os.path.exists(grep_path):
                continue

            if not os.path.exists(os.path.join(output_path, "final.pt")):
                command = f"python main_parser.py --input_folder {grep_path} --input-type {tp} --save_folder {output_path}"
                command += f" --attr-arch {attr_arch}"
                command += f" --dataset {dataset}"
                commands.append(command)
    return commands


def gen_commands_large_set(dataset, arch, setting, attr_arch, specific_type=None):
    _data_arch_name = f"{dataset}_{arch}"

    _atk_dir = os.path.join(gargs.ATK_DIR, _data_arch_name)
    _model_dir = os.path.join(gargs.MODEL_DIR, _data_arch_name)
    _parsing_dir = os.path.join(gargs.PARSING_DIR, attr_arch, _data_arch_name)
    _grep_dir = os.path.join(gargs.GREP_DIR, _data_arch_name)
    _log_dir = os.path.join(gargs.PARSING_LOG_DIR, attr_arch, _data_arch_name)

    setting_dir = os.path.join(_grep_dir, setting)
    attack_names = os.listdir(setting_dir)

    input_types = [specific_type] if specific_type else _input_types

    commands = []
    for atk_name in attack_names:
        for tp in input_types:
            grep_path = os.path.join(setting_dir, atk_name)
            output_path = os.path.join(_parsing_dir, setting, atk_name, tp)

            if not os.path.exists(grep_path):
                continue

            if not os.path.exists(os.path.join(output_path, "final.pt")):
                command = f"python main_parser.py --input_folder {grep_path} --input-type {tp} --save_folder {output_path}"
                command += f" --attr-arch {attr_arch}"
                command += f" --dataset {dataset}"
                commands.append(command)
    return commands


def gen_commands_eval_parsing_cross(exp_model, exp_data, attr_arch, specific_type=None):
    if exp_model['data'] != exp_data['data']:
        return []
    if exp_model['arch'] != exp_data['arch'] and exp_model['setting'] != exp_data['setting']:
        return []

    dataset = exp_model['data']
    arch_model = exp_model['arch']
    setting_model = exp_model['setting']
    attacks = [atk for atk in exp_model['attacks']
               if atk in exp_data['attacks']]

    arch_data = exp_data['arch']
    setting_data = exp_data['setting']

    _data_arch_model = os.path.join(f"{dataset}_{arch_model}", setting_model)
    _data_arch_data = os.path.join(f"{dataset}_{arch_data}", setting_data)
    _log_dir = os.path.join(f"{dataset}_{arch_model}" if arch_model == arch_data else f"{dataset}_model_{arch_model}_data_{arch_data}",
                            setting_model if setting_model == setting_data else f"model_{setting_model}_data_{setting_data}")

    _parsing_dir = os.path.join(gargs.PARSING_DIR, attr_arch, _data_arch_model)
    _grep_dir = os.path.join(gargs.GREP_DIR, _data_arch_data)
    _log_dir = os.path.join(gargs.PARSING_LOG_DIR, attr_arch, _log_dir)

    input_types = [specific_type] if specific_type else _input_types

    commands = []
    for tp in input_types:
        for data_atk in attacks:
            for model_atk in attacks:
                data_atk_name = get_attack_name(data_atk)
                atk_path = os.path.join(_grep_dir, data_atk_name)
                model_atk_name = get_attack_name(model_atk)
                output_path = os.path.join(
                    _parsing_dir, model_atk_name, tp)
                log_dir = os.path.join(_log_dir)

                command = f"python main_parser_eval.py --input_folder {atk_path} --input-type {tp} --save_folder {output_path} --log_dir {log_dir}"
                command += f" --attr-arch {attr_arch}"
                command += f" --dataset {dataset}"

                if os.path.exists(os.path.join(output_path, 'final.pt')) and os.path.exists(atk_path):
                    if not os.path.exists(os.path.join(log_dir, f"data_{data_atk_name}___model_{model_atk_name}__{tp}.log")):
                        commands.append(command)
    return commands


def gen_commands_eval_parsing(exp, attr_arch, specific_type=None):
    dataset = exp['data']
    arch = exp['arch']
    setting = exp['setting']
    attacks = exp['attacks']

    _data_arch_model = os.path.join(f"{dataset}_{arch}", setting)
    _data_arch_data = os.path.join(f"{dataset}_{arch}", setting)
    _log_dir = os.path.join(f"{dataset}_{arch}", setting)

    _parsing_dir = os.path.join(gargs.PARSING_DIR, attr_arch, _data_arch_model)
    _grep_dir = os.path.join(gargs.GREP_DIR, _data_arch_data)
    _log_dir = os.path.join(gargs.PARSING_LOG_DIR, attr_arch, _log_dir)

    input_types = [specific_type] if specific_type else _input_types

    commands = []
    for tp in input_types:
        for data_atk in attacks:
            model_atk = data_atk
            data_atk_name = get_attack_name(data_atk)
            atk_path = os.path.join(_grep_dir, data_atk_name)
            model_atk_name = get_attack_name(model_atk)
            output_path = os.path.join(
                _parsing_dir, model_atk_name, tp)
            log_dir = os.path.join(_log_dir)

            command = f"python main_parser_eval.py --input_folder {atk_path} --input-type {tp} --save_folder {output_path} --log_dir {log_dir}"
            command += f" --attr-arch {attr_arch}"
            command += f" --dataset {dataset}"

            if os.path.exists(os.path.join(output_path, 'final.pt')) and os.path.exists(atk_path):
                if not os.path.exists(os.path.join(log_dir, f"data_{data_atk_name}___model_{model_atk_name}__{tp}.log")):
                    commands.append(command)
    return commands


def gen_commands_large_set_test(dataset, arch, setting, attr_arch, specific_type=None):
    _data_arch_name = f"{dataset}_{arch}"

    _parsing_dir = os.path.join(gargs.PARSING_DIR, attr_arch, _data_arch_name, setting)
    _grep_dir = os.path.join(gargs.GREP_DIR, _data_arch_name, setting)
    _log_dir = os.path.join(gargs.PARSING_LOG_DIR, attr_arch, _data_arch_name, setting)

    attack_names = os.listdir(_grep_dir)

    input_types = [specific_type] if specific_type else _input_types

    commands = []
    for tp in input_types:
        for data_atk_name in attack_names:
            for model_atk_name in attack_names:
                atk_path = os.path.join(_grep_dir, data_atk_name)
                output_path = os.path.join(
                    _parsing_dir, model_atk_name, tp)
                log_dir = os.path.join(_log_dir)

                command = f"python main_parser_eval.py --input_folder {atk_path} --input-type {tp} --save_folder {output_path} --log_dir {log_dir}"
                command += f" --attr-arch {attr_arch}"
                command += f" --dataset {dataset}"

                if os.path.exists(os.path.join(output_path, 'final.pt')) and os.path.exists(atk_path):
                    if not os.path.exists(os.path.join(log_dir, f"data_{data_atk_name}___model_{model_atk_name}__{tp}.log")):
                        commands.append(command)
    # commands = []
    # for atk_name in attack_names:
    #     for tp in input_types:
    #         grep_path = os.path.join(setting_dir, atk_name)
    #         output_path = os.path.join(_parsing_dir, setting, atk_name, tp)

    #         if not os.path.exists(grep_path):
    #             continue

    #         if not os.path.exists(os.path.join(output_path, "final.pt")):
    #             command = f"python main_parser.py --input_folder {grep_path} --input-type {tp} --save_folder {output_path}"
    #             command += f" --attr-arch {attr_arch}"
    #             command += f" --dataset {dataset}"
    #             commands.append(command)
    return commands

def train_victim_commands():
    commands = []
    for exp in gargs.EXPS:
        if exp['setting'] not in 'origin robust':
            continue
        robust = 'robust' in exp['setting']

        commands += gen_commands_train_victim(
            dataset=exp['data'], arch=exp['arch'], robust=robust)
    print(len(commands))
    return commands


def attack_victim_commands():
    commands = []
    for exp in gargs.EXPS:
        if exp['setting'] not in 'origin robust':
            continue
        robust = 'robust' in exp['setting']

        commands += gen_commands_attack_victim(
            dataset=exp['data'], arch=exp['arch'], attacks=exp['attacks'], robust=robust)
    print(len(commands))
    return commands


def train_parsing_commands(attr_arch, specific_type = None):
    commands = []
    if attr_arch not in ['conv4']:
        commands += gen_commands_parsing(gargs.EXPS[0], attr_arch, specific_type)
    else:
        for exp in gargs.EXPS:
            commands += gen_commands_parsing(exp, attr_arch, specific_type)
    print(len(commands))
    return commands


def train_large_set_parsing_commands(attr_arch, specific_type = None):
    commands = []
    exps = [
        ("cifar10", "full_archs", "origin"),
        ("cifar10", "partial_archs", "origin"),
        ("cifar10", "resnet9", "grouped_attack_origin"),
    ]
    for data, arch, setting in exps:
        commands += gen_commands_large_set(data, arch, setting, attr_arch, specific_type)
    print("ext: ", len(commands))
    return commands

def test_large_set_parsing_commands(attr_arch, specific_type = None):
    commands = []
    exps = [
        ("cifar10", "full_archs", "origin"),
        ("cifar10", "partial_archs", "origin"),
        ("cifar10", "resnet9", "grouped_attack_origin"),
    ]
    for data, arch, setting in exps:
        commands += gen_commands_large_set_test(data, arch, setting, attr_arch, specific_type)
    print("ext: ", len(commands))
    return commands

def test_parsing_commands(attr_arch):
    commands = []
    for exp in gargs.EXPS:
        commands += gen_commands_eval_parsing(exp=exp, attr_arch=attr_arch)
    print(len(commands))
    return commands


def cross_test_parsing_commands(attr_arch, specific_type=None):
    commands = []
    for exp1 in gargs.EXPS[:5]:
        for exp2 in gargs.EXPS[:5]:
            commands += gen_commands_eval_parsing_cross(exp1, exp2, attr_arch, specific_type)
    exp1 = gargs.EXPS[0]
    for exp2 in gargs.EXPS[5:]:
        commands += gen_commands_eval_parsing_cross(exp2, exp2, attr_arch, specific_type)
    for exp2 in gargs.EXPS[5:]:
        commands += gen_commands_eval_parsing_cross(exp1, exp2, attr_arch, specific_type)
        commands += gen_commands_eval_parsing_cross(exp2, exp1, attr_arch, specific_type)
    print(len(commands))
    return commands


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("train")
    parser.add_argument('--stage', type=int, help="To decide which part of commands to execute. (1, 2, 3)")
    parser.add_argument('--gpus', type=str,
                        default="0,1,2,3,4,5,6,7", help="Run on which gpus. e.g.: --gpus 0,1,2,3")
    parser.add_argument('--thread', type=int, default=1, help="Number of commands running parallel in one gpu.")
    parser.add_argument('--debug', action="store_true", help="Only generate commands without executing if tagged.")
    parser.add_argument('--denoise', action="store_true", help='Using denoiser when training attribute models.')
    args = parser.parse_args()
    debug = args.debug
    stage = args.stage
    th = args.thread
    gpus = [int(g) for g in args.gpus.split(',')]

    # call each code block seperatly

    if stage == 0:
        # victim training
        ext = f" --ffcv-dir {gargs.FFCV_FORMAT}"

        commands = train_victim_commands()
        run_commands(gpus * th if not debug else [0], commands, call=not debug,
                     ext_command=ext, suffix="commands0", shuffle=False, delay=1)
    elif stage == 1:
        # victim training
        ext = f" --ffcv-dir {gargs.FFCV_FORMAT}_1"

        commands = attack_victim_commands()
        run_commands(gpus * th if not debug else [0], commands, call=not debug,
                     ext_command=ext, suffix="commands1", shuffle=False, delay=1)
    elif stage == 2:
        # parsing training
        # need call grep_data.py before training parsing models
        commands = []
        commands += train_parsing_commands(attr_arch="conv4")
        commands += train_large_set_parsing_commands(attr_arch="conv4")
        for at_arch in gargs.VALID_ATTR_ARCHS:
            if at_arch != "conv4":
                commands += train_parsing_commands(attr_arch=at_arch)
        # commands += train_parsing_commands(attr_arch="mlp")
        if args.denoise:
            commands = []
            print("denoise")
            commands += train_parsing_commands("conv4", "denoise")
            # commands += train_large_set_parsing_commands(attr_arch="conv4", specific_type="denoise")
        run_commands(gpus * th if not debug else [0], commands, call=not debug,
                     suffix="commands2", shuffle=False, delay=1)
    elif stage == 3:
        commands = []

        # parsing cross testing
        # commands += cross_test_parsing_commands(attr_arch="attrnet")
        commands += cross_test_parsing_commands(attr_arch="conv4")
        # commands += test_parsing_commands(attr_arch="mlp")

        # parsing testing
        for at_arch in gargs.VALID_ATTR_ARCHS:
            if at_arch != "conv4":
                commands += test_parsing_commands(attr_arch=at_arch)
        commands += test_large_set_parsing_commands(attr_arch="conv4")

        if args.denoise:
            commands = []
            print("denoise")
            # commands += gen_commands_eval_parsing_cross(gargs.EXPS[0], gargs.EXPS[0], "conv4", "denoise")
            # print(len(commands))
            commands += cross_test_parsing_commands("conv4", "denoise")
            commands += test_large_set_parsing_commands(attr_arch="conv4", specific_type="denoise")

        run_commands(gpus * th if not debug else [0], commands, call=not debug,
                     suffix="commands3", shuffle=False, delay=2)
