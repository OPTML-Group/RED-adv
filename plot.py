import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

import run

input_types = ["delta", "x_adv"]
os.makedirs("figs", exist_ok=True)

def load_file(data_atk, model_atk, tp):
    data_atk_name = run.get_atk_name(data_atk)
    model_atk_name = run.get_atk_name(model_atk)
    log_path = os.path.join("/localscratch2/ljcc/test_log", f"data_{data_atk_name}___model_{model_atk_name}__{tp}.log")
    if os.path.exists(log_path):
        with open(log_path, 'r') as fin:
            a = []
            for line in fin:
                a.append(float(line[:-1]))
        return a
    return [0, 0, 0]

def plot_all():
    for tp in input_types:
        n_dim = len(run.attacks)
        a = np.zeros([3, n_dim, n_dim])
        for idx, model_atk in enumerate(run.attacks):
            for idy, data_atk in enumerate(run.attacks):
                a[:, idx, idy] = load_file(data_atk, model_atk, tp)
        for i in range(3):
            plt.clf()
            sns.heatmap(a[i])
            plt.savefig(f"figs/full_{i}_{tp}.png")

rg = {
    'pgd': list(range(0, 4)),
    'pgdl2': list(range(4, 9)),
    'fgsm': list(range(9, 13)),
    'squareinf': list(range(13, 17)),
    'squarel2': list(range(17, 21)),
    'autoattackinf': list(range(21,  25)),
    'autoattackl2': list(range(25, 29)),
    'cw': list(range(29, 38)),
    'autofull': list(range(21, 29)),
    'squarefull': list(range(13, 21)),
    "full": list(range(0, len(run.attacks)))
}
def plot_all():
    for tp in input_types:
        n_dim = len(run.attacks)
        a = np.zeros([3, n_dim, n_dim])
        for idx, model_atk in enumerate(run.attacks):
            for idy, data_atk in enumerate(run.attacks):
                a[:, idx, idy] = load_file(data_atk, model_atk, tp)
        for i in range(3):
            plt.clf()
            sns.heatmap(a[i])
            plt.savefig(f"figs/full_{i}_{tp}.png")

def plot_range(lis, prefix="",annot=False):
    n_dim = len(lis)
    for tp in input_types:
        a = np.zeros([4, n_dim, n_dim])
        for idx, i in enumerate(lis):
            for idy, j in enumerate(lis):
                model_atk = run.attacks[i]
                data_atk = run.attacks[j]
                a[:3, idx, idy] = load_file(data_atk, model_atk, tp)
        a[3] = np.mean(a[:3], axis=0)
        for i in range(4):
            dir = f"figs/{tp}_{i}"
            os.makedirs(dir,exist_ok=True)
            plt.clf()
            sns.heatmap(a[i], annot=annot, fmt=".2f", linewidths=0.5 * annot)
            plt.savefig(os.path.join(dir, f"{prefix}_{i}.png"))
            

if __name__ == "__main__":
    import shutil
    if True:
        shutil.rmtree("figs", ignore_errors=True)
    for key, item in rg.items():
        plot_range(item, key, annot=key!="full")