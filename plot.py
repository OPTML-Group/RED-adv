import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import shutil

import global_args as gargs
import run

input_types = ["delta", "x_adv"]


def check_group(group, atks):
    return all(a in atks for a in group)


def get_attack_display_name(atk):
    dir_name = [atk['attack']]
    if atk.get('norm') == 'L2':
        dir_name.append('L2')

    for key, val in atk.items():
        if key not in ['attack', 'norm', 'alpha']:
            dir_name.append(f"{key}={val}")
    return ' '.join(dir_name)


def load_file(exp, model_atk_name, data_atk_name, tp):
    dataset = exp['data']
    arch = exp['arch']
    setting = exp['setting']
    data_arch = f"{dataset}_{arch}"
    log_path = os.path.join(gargs.PARSING_LOG_DIR, data_arch, setting,
                            f"data_{data_atk_name}___model_{model_atk_name}__{tp}.log")

    if os.path.exists(log_path):
        with open(log_path, 'r') as fin:
            a = []
            for line in fin:
                a.append(float(line[:-1]))
        return a
    return [0, 0, 0]


def plot_range(attacks, exp, prefix="", annot=False):
    dataset = exp['data']
    arch = exp['arch']
    setting = exp['setting']

    data_arch = f"{dataset}_{arch}"

    attack_names = [run.get_attack_name(atk) for atk in attacks]
    display_names = [get_attack_display_name(atk) for atk in attacks]
    for atk in attacks:
        name = atk['attack']
        if atk.get('norm') == 'L2':
            name += '_L2'

    plt.rcParams["font.family"] = "DeJavu Serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]

    n_dim = len(attacks)
    for tp in input_types:
        a = np.zeros([4, n_dim, n_dim])

        for idx, model_atk_name in enumerate(attack_names):
            for idy, data_atk_name in enumerate(attack_names):
                a[:3, idx, idy] = load_file(
                    exp, model_atk_name, data_atk_name, tp)

        a[3] = np.mean(a[:3], axis=0)

        for i in range(4):
            dir = os.path.join('figs', data_arch, setting, f"{tp}_{i}")
            os.makedirs(dir, exist_ok=True)
            plt.clf()
            heatmap = sns.heatmap(a[i], annot=annot, fmt=".2f",
                                  linewidths=0.5 * annot, cmap="vlag",
                                  xticklabels=display_names, yticklabels=display_names)
            if prefix == "all":
                heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize = 5)
                heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize = 5)
            plt.xticks(rotation=45, ha='right')
            plt.savefig(os.path.join(
                dir, f"{prefix}_{i}.png"), bbox_inches='tight', dpi=300)


if __name__ == "__main__":
    shutil.rmtree("figs", ignore_errors=True)

    for exp in gargs.EXPS:
        plot_range(exp['attacks'], exp, prefix="all", annot=False)

        for group in gargs.ALL_GROUP:
            atks = exp['attacks']
            if check_group(group, atks):
                name = group[0]['attack']
                if group[0].get('norm') == 'L2':
                    name += '_L2'
                print(name)
                plot_range(group, exp, prefix=name, annot=True)
