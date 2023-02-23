from utils import run_commands
import os

import global_args as gargs


kernels = gargs.KERNEL_SIZES
acts = gargs.ACTIVATION_FUNCTIONS
ratios = gargs.PRUNING_RATIOS
struct = [False, True]


def get_attack_name(atk):
    dir_name = []
    for key, val in atk.items():
        dir_name.append(f"{key}_{val}")
    return '_'.join(dir_name)


def gen_commands_victim(dataset, arch, attacks, robust):
    _data_arch_name = f"{dataset}_{arch}"

    _atk_dir = os.path.join(gargs.ATK_DIR, _data_arch_name)
    _model_dir = os.path.join(gargs.MODEL_DIR, _data_arch_name)
    _parsing_dir = os.path.join(gargs.PARSING_DIR, _data_arch_name)
    _grep_dir = os.path.join(gargs.GREP_DIR, _data_arch_name)
    _log_dir = os.path.join(gargs.PARSING_LOG_DIR, _data_arch_name)

    commands = []
    # struct = [True]
    for idx, atk in enumerate(attacks):
        for k in kernels:
            for a in acts:
                for s in struct:
                    for r in ratios:
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

                        model_name = "seed{}_kernel{}_act{}_prune{}".format(
                            2, k, a, r)
                        if s:
                            model_name += "_struct"
                        if robust:
                            model_name += '_robust'

                        path = os.path.join(
                            _model_dir, f"{model_name}_omp_2/checkpoint_75.pt")
                        # commands.append(command)
                        if idx == 0 and not os.path.exists(path):
                            commands.append(command)
                        else:
                            if not (os.path.exists(os.path.join(atk_path, model_name, 'ori_pred.pt')) and
                                    os.path.exists(os.path.join(atk_path, model_name, 'attack_acc.log'))) and \
                                    not os.path.exists(os.path.join(atk_path, model_name, 'x_adv.pt')):
                                # print(path)
                                if idx == 0 or idx > 0 and os.path.exists(path):
                                    commands.append(command)
    return commands


def gen_commands_parsing(exp):
    dataset=exp['data']
    arch=exp['arch']
    setting=exp['setting']
    attacks=exp['attacks']
    _data_arch_name = f"{dataset}_{arch}"

    _atk_dir = os.path.join(gargs.ATK_DIR, _data_arch_name)
    _model_dir = os.path.join(gargs.MODEL_DIR, _data_arch_name)
    _parsing_dir = os.path.join(gargs.PARSING_DIR, _data_arch_name)
    _grep_dir = os.path.join(gargs.GREP_DIR, _data_arch_name)
    _log_dir = os.path.join(gargs.PARSING_LOG_DIR, _data_arch_name)

    input_types = ["delta", "x_adv"]

    commands = []
    for atk in attacks:
        for tp in input_types:
            akt_name = get_attack_name(atk)

            grep_path = os.path.join(_grep_dir, setting, akt_name)
            output_path = os.path.join(_parsing_dir, setting, akt_name, tp)

            if not os.path.exists(grep_path):
                continue

            if not os.path.exists(os.path.join(output_path, "final.pt")):
                command = f"python old_parser.py --input_folder {grep_path} --input-type {tp} --save_folder {output_path}"
                commands.append(command)
    return commands


def gen_commands_eval_parsing_cross(exp_model, exp_data):
    if exp_model['data'] != exp_data['data']:
        return []
    if exp_model['arch'] != exp_data['arch'] and exp_model['setting'] != exp_data['setting']:
        return []

    dataset=exp_model['data']
    arch_model=exp_model['arch']
    setting_model=exp_model['setting']
    attacks = [atk for atk in exp_model['attacks'] if atk in exp_data['attacks']]

    arch_data=exp_data['arch']
    setting_data=exp_data['setting']

    _data_arch_model = os.path.join(f"{dataset}_{arch_model}", setting_model)
    _data_arch_data = os.path.join(f"{dataset}_{arch_data}", setting_data)
    _log_dir = os.path.join(f"{dataset}_{arch_model}" if arch_model == arch_data else f"{dataset}_model_{arch_model}_data_{arch_data}", 
                            setting_model if setting_model == setting_data else f"model_{setting_model}_data_{setting_data}")

    _parsing_dir = os.path.join(gargs.PARSING_DIR, _data_arch_model)
    _grep_dir = os.path.join(gargs.GREP_DIR, _data_arch_data)
    _log_dir = os.path.join(gargs.PARSING_LOG_DIR, _log_dir)

    input_types = ["delta", "x_adv"]

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
                command = f"python old_eval_parser.py --input_folder {atk_path} --input-type {tp} --save_folder {output_path} --log_dir {log_dir}"
                if os.path.exists(os.path.join(output_path, 'final.pt')) and os.path.exists(atk_path):
                    if not os.path.exists(os.path.join(log_dir, f"data_{data_atk_name}___model_{model_atk_name}__{tp}.log")):
                        commands.append(command)
    return commands


def gen_commands_eval_parsing(exp):
    # dataset=exp['data']
    # arch=exp['arch']
    # setting=exp['setting']
    # attacks=exp['attacks']
    # _data_arch_name = f"{dataset}_{arch}"

    # _atk_dir = os.path.join(gargs.ATK_DIR, _data_arch_name)
    # _model_dir = os.path.join(gargs.MODEL_DIR, _data_arch_name)
    # _parsing_dir = os.path.join(gargs.PARSING_DIR, _data_arch_name)
    # _grep_dir = os.path.join(gargs.GREP_DIR, _data_arch_name)
    # _log_dir = os.path.join(gargs.PARSING_LOG_DIR, _data_arch_name)

    # input_types = ["delta", "x_adv"]

    # commands = []
    # for tp in input_types:
    #     for data_atk in attacks:
    #         for model_atk in attacks:
    #             data_atk_name = get_attack_name(data_atk)
    #             atk_path = os.path.join(_grep_dir, setting, data_atk_name)
    #             model_atk_name = get_attack_name(model_atk)
    #             output_path = os.path.join(
    #                 _parsing_dir, setting, model_atk_name, tp)
    #             log_dir = os.path.join(_log_dir, setting)
    #             command = f"python old_eval_parser.py --input_folder {atk_path} --input-type {tp} --save_folder {output_path} --log_dir {log_dir}"
    #             if os.path.exists(os.path.join(output_path, 'final.pt')) and os.path.exists(atk_path):
    #                 if not os.path.exists(os.path.join(log_dir, f"data_{data_atk_name}___model_{model_atk_name}__{tp}.log")):
    #                     commands.append(command)
    return gen_commands_eval_parsing_cross(exp, exp)


def train_victim_commands():
    commands = []
    for exp in gargs.EXPS[::-1][:1]:
        if exp['setting'] not in 'origin robust':
            continue
        robust = 'robust' in exp['setting']

        commands += gen_commands_victim(
            dataset=exp['data'], arch=exp['arch'], attacks=exp['attacks'], robust=robust)
    print(len(commands))
    return commands


def train_parsing_commands():
    commands = []
    for exp in gargs.EXPS:
        commands += gen_commands_parsing(exp)
    print(len(commands))
    return commands


def test_parsing_commands():
    commands = []
    for exp in gargs.EXPS:
        commands += gen_commands_eval_parsing(exp=exp)
    print(len(commands))
    return commands


def cross_test_parsing_commands():
    commands = []
    for exp1 in gargs.EXPS[:5]:
        for exp2 in gargs.EXPS:
            commands += gen_commands_eval_parsing_cross(exp1, exp2)
    # exp2 = gargs.EXPS[0]
    # for exp1 in gargs.EXPS[5:] + gargs.EXPS[1:2]:
    #     commands += gen_commands_eval_parsing_cross(exp1, exp2)
    #     commands += gen_commands_eval_parsing_cross(exp2, exp1)
    print(len(commands))
    return commands


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("train")
    parser.add_argument('--stage', type=int)
    parser.add_argument('--debug', action="store_true")
    args = parser.parse_args()
    debug = args.debug
    stage = args.stage

    # call each code block seperatly

    if stage == 1:
        # victim training
        ext = f" --ffcv-dir {gargs.FFCV_FORMAT}"

        commands = train_victim_commands()
        run_commands([1, 2, 3, 4, 5, 6, 7, 0] * 3 if not debug else [0], commands, call=not debug,
                    ext_command=ext, suffix="commands1", shuffle=False, delay=1)
    elif stage == 2:
        # parsing training
        # need call grep_data.py before training parsing models
        commands = train_parsing_commands()
        run_commands([1, 2, 3, 4, 5, 6, 7, 0] * 2 if not debug else [0], commands, call=not debug,
                    suffix="commands2", shuffle=False, delay=1)
    elif stage == 3:
        # parsing testing
        # commands = test_parsing_commands()
        # run_commands([1, 2, 3, 4, 5, 6, 7, 0] * 4 if not debug else [0], commands, call=not debug,
        #              suffix="commands3", shuffle=False, delay=0.5)

        # parsing cross testing
        commands = cross_test_parsing_commands()
        run_commands([1, 2, 3, 4, 5, 6, 7, 0] * 4 if not debug else [0], commands, call=not debug,
                     suffix="commands3", shuffle=False, delay=4)
