import os
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import plot
import global_args as gargs
import run

def load_attr(attr_arch, dataset, arch, setting, attacks):
    attr_dir = os.path.join(gargs.PARSING_LOG_DIR, attr_arch)
    da_dir = os.path.join(attr_dir, f"{dataset}_{arch}")
    settting_dir = os.path.join(da_dir, setting)

    mats = {}
    for tp in plot.input_types:
        a = np.zeros([4, len(attacks)])
        for idx, atk in enumerate(attacks):
            atk_name = run.get_attack_name(atk)
            a[:3, idx] = plot.load_file(settting_dir, atk_name, atk_name, tp)
        a[3] = np.mean(a[:3], axis=0)
        mats[tp] = a
    return mats

def plot_attr(attacks, attack_name, save_dir="attr_perf", annot=False):
    exp = gargs.EXPS[0]

    dataset = exp['data']
    arch = exp['arch']
    setting = exp['setting']

    display_names = [plot.get_attack_display_name(atk) for atk in attacks]

    attr_archs = [gargs.VALID_ATTR_ARCHS[i] for i in [1, 0, 2, 3]]

    mats = {tp: np.zeros([4, len(attr_archs), len(attacks)]) for tp in plot.input_types}
    for idx, attr_arch in enumerate(attr_archs):
        a = load_attr(attr_arch, dataset, arch, setting, attacks)
        for tp in plot.input_types:
            mats[tp][:, idx] = a[tp]

    plt.rcParams["font.family"] = "DeJavu Serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]

    name = ["Kernel Size", "Activation Function", "Pruning Ratio", "All"]
    
    for tp, a in mats.items():
        for i in range(4):
            dir = os.path.join('figs', save_dir, f"{tp}_{i}")
            os.makedirs(dir, exist_ok=True)
            plt.clf()
            heatmap = sns.heatmap(a[i], annot=annot, fmt=".2f",
                                  linewidths=0.5 * annot, cmap="vlag",
                                  xticklabels=display_names, yticklabels=attr_archs)
            heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=8)
            heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=5)

            plt.title(
                f"Model Parsing Accuracy on {name[i]} from {tp}(%)", fontsize=15)
            plt.ylabel("Attribution Network",
                       fontsize=13)  # x-axis label with fontsize 15
            plt.xlabel("Attack Methods", fontsize=13)
            plt.xticks(rotation=45, ha='right')
            plt.savefig(os.path.join(dir, f"{attack_name}_{i}.png"), bbox_inches='tight', dpi=300)


if __name__ == "__main__":
    plot_attr(gargs.WHITEBOX_ATTACKS, 'whitebox')
    plot_attr(gargs.BLACKBOX_ATTACKS, 'blackbox')
    plot_attr(gargs.ALL_ATTACKS, 'all')