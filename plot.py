import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import global_args as gargs
import training_utils

input_types = ["delta", "x_adv"]


def check_group(group, atks):
    return all(a in atks for a in group)


def get_attack_display_name(atk, exp_name=None):
    dir_name = []

    if exp_name is not None:
        dir_name.append(exp_name)

    dir_name.append(atk["attack"])
    if atk.get("norm") == "L2":
        dir_name.append("L2")

    for key, val in atk.items():
        if key not in ["attack", "norm", "alpha"]:
            dir_name.append(f"{key}={val}")
    return " ".join(dir_name)


def load_file(log_dir, model_atk_name, data_atk_name, tp):
    log_path = os.path.join(
        log_dir, f"data_{data_atk_name}___model_{model_atk_name}__{tp}.log"
    )

    if os.path.exists(log_path):
        with open(log_path, "r") as fin:
            a = []
            for line in fin:
                a.append(float(line[:-1]))
        return a
    return [33, 33, 20]


def load_exp(exp_model, exp_data, attacks, attr_arch):
    if exp_model["data"] != exp_data["data"]:
        raise Exception(
            "Dataset mismatch! {} {}".format(exp_model["data"], exp_data["data"])
        )
    if (
        exp_model["arch"] != exp_data["arch"]
        and exp_model["setting"] != exp_data["setting"]
    ):
        raise Exception(
            "Arch and setting mismatch! {} {}, {} {}".format(
                exp_model["arch"],
                exp_data["arch"],
                exp_model["setting"],
                exp_data["setting"],
            )
        )

    dataset = exp_model["data"]
    arch_model = exp_model["arch"]
    setting_model = exp_model["setting"]

    arch_data = exp_data["arch"]
    setting_data = exp_data["setting"]

    _log_dir = os.path.join(
        f"{dataset}_{arch_model}"
        if arch_model == arch_data
        else f"{dataset}_model_{arch_model}_data_{arch_data}",
        setting_model
        if setting_model == setting_data
        else f"model_{setting_model}_data_{setting_data}",
    )

    _log_dir = os.path.join(gargs.PARSING_LOG_DIR, attr_arch, _log_dir)

    n_dim = len(attacks)
    attack_names = [training_utils.get_attack_name(atk) for atk in attacks]
    mats = {}

    for tp in input_types:
        a = np.zeros([4, n_dim, n_dim])

        for idx, model_atk_name in enumerate(attack_names):
            for idy, data_atk_name in enumerate(attack_names):
                a[:3, idx, idy] = load_file(_log_dir, model_atk_name, data_atk_name, tp)

        a[3] = np.mean(a[:3], axis=0)
        mats[tp] = a

    return mats


def plot_range(attacks, exp, attr_arch, prefix="", annot=False):
    dataset = exp["data"]
    arch = exp["arch"]
    setting = exp["setting"]

    data_arch = f"{dataset}_{arch}"

    display_names = [get_attack_display_name(atk) for atk in attacks]
    for atk in attacks:
        name = atk["attack"]
        if atk.get("norm") == "L2":
            name += "_L2"

    plt.rcParams["font.family"] = "DeJavu Serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]

    name = ["Kernel Size", "Activation Function", "Pruning Ratio", "All"]

    mats = load_exp(exp, exp, attacks, attr_arch)

    for tp, a in mats.items():
        for i in range(4):
            dir = os.path.join("figs", data_arch, setting, f"{tp}_{i}")
            os.makedirs(dir, exist_ok=True)
            plt.clf()
            heatmap = sns.heatmap(
                a[i],
                annot=annot,
                fmt=".2f",
                linewidths=0.5 * annot,
                cmap="vlag",
                xticklabels=display_names,
                yticklabels=display_names,
            )
            if prefix in ["all", "blackbox", "whitebox"]:
                heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=5)
                heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=5)

            plt.title(
                f"Model Parsing Accuracy on {name[i]} from {tp}(%)\ndataset: {dataset}, victim: {arch}, setting: {setting}, attacks: {prefix}",
                fontsize=15,
            )
            plt.ylabel(
                "Attack Methods to Train the Parser", fontsize=13
            )  # x-axis label with fontsize 15
            plt.xlabel("Attack Methods to Evaluate the Parser", fontsize=13)
            plt.xticks(rotation=45, ha="right")
            plt.savefig(
                os.path.join(dir, f"{prefix}_{i}.png"), bbox_inches="tight", dpi=300
            )


def plot_huge(
    attacks,
    exps,
    exp_names,
    attr_arch,
    save_dir,
    prefix="",
    annot=False,
    title_suff="",
    scale=False,
    mask_same=False,
):
    display_names = [
        get_attack_display_name(atk, exp) for exp in exp_names for atk in attacks
    ]
    n_sq = len(attacks)
    n_exps = len(exps)
    for atk in attacks:
        name = atk["attack"]
        if atk.get("norm") == "L2":
            name += "_L2"

    plt.rcParams["font.family"] = "DeJavu Serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]

    name = ["Kernel Size", "Activation Function", "Pruning Ratio", "All"]

    mats = {tp: np.zeros([4, n_sq * n_exps, n_sq * n_exps]) for tp in input_types}

    for idx_model, exp_model in enumerate(exps):
        for idx_data, exp_data in enumerate(exps):
            if mask_same and idx_model == idx_data:
                exp_mat = {
                    tp: np.zeros([4, n_sq, n_sq])
                    + np.array([33, 33, 20, 86 / 3])[:, None, None]
                    for tp in input_types
                }
            else:
                exp_mat = load_exp(exp_model, exp_data, attacks, attr_arch)
            for tp, a in exp_mat.items():
                if scale:
                    a -= a.min(axis=2).min(axis=1)[:, None, None]
                    a /= a.max(axis=2).max(axis=1)[:, None, None] + 1e-15
                mats[tp][
                    :,
                    n_sq * idx_model : n_sq * (idx_model + 1),
                    n_sq * idx_data : n_sq * (idx_data + 1),
                ] = a

    for tp, a in mats.items():
        for i in range(4):
            dir = os.path.join("figs", save_dir, f"{tp}_{i}")
            os.makedirs(dir, exist_ok=True)
            plt.clf()
            heatmap = sns.heatmap(
                a[i],
                annot=annot,
                fmt=".2f",
                linewidths=0.5 * annot,
                cmap="vlag",
                xticklabels=display_names,
                yticklabels=display_names,
            )
            if prefix in ["all", "blackbox", "whitebox"]:
                heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=3)
                heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=3)

            plt.title(
                f"Model Parsing Accuracy on {name[i]} from {tp}(%)\n{title_suff}",
                fontsize=15,
            )
            plt.ylabel(
                "Attack Methods to Train the Parser", fontsize=13
            )  # x-axis label with fontsize 15
            plt.xlabel("Attack Methods to Evaluate the Parser", fontsize=13)
            plt.xticks(rotation=45, ha="right")
            plt.savefig(
                os.path.join(dir, f"{prefix}_{i}.png"), bbox_inches="tight", dpi=300
            )


def draw_plot(group, exp, attr_arch, name, annot=False):
    atks = exp["attacks"]
    if check_group(group, atks):
        if group[0].get("norm") == "L2":
            name += "_L2"
        print(name)
        plot_range(group, exp, attr_arch, prefix=name, annot=annot)


if __name__ == "__main__":
    # shutil.rmtree("figs", ignore_errors=True)

    attr_arch = "conv4"

    plot_huge(
        gargs.WHITEBOX_ATTACKS,
        [gargs.EXPS[0], gargs.EXPS[5]],
        ["Standard", "Robust"],
        attr_arch,
        "default_origin_robust",
        prefix="whitebox",
        annot=False,
        title_suff="dataset: cifar10, victim: resnet9, setting: origin vs robust, attacks: whitebox",
    )
    plot_huge(
        gargs.WHITEBOX_ATTACKS,
        [gargs.EXPS[0], gargs.EXPS[5]],
        ["Standard", "Robust"],
        attr_arch,
        "default_origin_robust_scaled",
        prefix="whitebox",
        annot=False,
        title_suff="dataset: cifar10, victim: resnet9, setting: origin vs robust, attacks: whitebox",
        scale=True,
    )
    plot_huge(
        gargs.WHITEBOX_ATTACKS,
        [gargs.EXPS[0], gargs.EXPS[6]],
        ["Standard", "Robust"],
        attr_arch,
        "default_origin_robust_all",
        prefix="whitebox",
        annot=False,
        title_suff="dataset: cifar10, victim: resnet9, setting: origin vs robust all, attacks: whitebox",
    )
    plot_huge(
        gargs.WHITEBOX_ATTACKS,
        [gargs.EXPS[0], gargs.EXPS[6]],
        ["Standard", "Robust"],
        attr_arch,
        "default_origin_robust_all_scaled",
        prefix="whitebox",
        annot=False,
        title_suff="dataset: cifar10, victim: resnet9, setting: origin vs robust all, attacks: whitebox",
        scale=True,
    )
    exps = gargs.EXPS[:5]
    exp_names = [exp["arch"] for exp in exps]
    print(exp_names)

    exps = [exps[i] for i in [1, 3, 0, 4, 2]]
    exp_names = [exp["arch"] for exp in exps]
    plot_huge(
        gargs.WHITEBOX_ATTACKS,
        exps,
        exp_names,
        attr_arch,
        "unseen_archs",
        prefix="whitebox",
        annot=False,
        title_suff="dataset: cifar10, setting: origin, attacks: whitebox",
    )
    plot_huge(
        gargs.WHITEBOX_ATTACKS,
        exps,
        exp_names,
        attr_arch,
        "unseen_archs_scaled",
        prefix="whitebox",
        annot=False,
        title_suff="dataset: cifar10, setting: origin, attacks: whitebox",
        scale=True,
    )
    plot_huge(
        gargs.WHITEBOX_ATTACKS,
        exps,
        exp_names,
        attr_arch,
        "unseen_archs_masked",
        prefix="whitebox",
        annot=False,
        title_suff="dataset: cifar10, setting: origin, attacks: whitebox",
        mask_same=True,
    )
    plot_huge(
        gargs.PGD_ATTACKS,
        exps,
        exp_names,
        attr_arch,
        "unseen_archs_masked",
        prefix="pgd",
        annot=False,
        title_suff="dataset: cifar10, setting: origin, attacks: pgd",
        mask_same=True,
    )
    plot_huge(
        [gargs.PGD_ATTACKS[1]],
        exps,
        exp_names,
        attr_arch,
        "unseen_archs_masked",
        prefix="pgd8",
        annot=True,
        title_suff="dataset: cifar10, setting: origin, attacks: pgd8",
        mask_same=True,
    )
    plot_huge(
        gargs.PGD_ATTACKS,
        exps,
        exp_names,
        attr_arch,
        "unseen_archs",
        prefix="pgd",
        annot=False,
        title_suff="dataset: cifar10, setting: origin, attacks: pgd",
    )
    plot_huge(
        [gargs.PGD_ATTACKS[1]],
        exps,
        exp_names,
        attr_arch,
        "unseen_archs",
        prefix="pgd8",
        annot=True,
        title_suff="dataset: cifar10, setting: origin, attacks: pgd8",
    )

    plot_huge(
        gargs.FGSM_ATTACKS,
        exps,
        exp_names,
        attr_arch,
        "unseen_archs_masked",
        prefix="fgsm",
        annot=False,
        title_suff="dataset: cifar10, setting: origin, attacks: fgsm",
        mask_same=True,
    )
    plot_huge(
        [gargs.FGSM_ATTACKS[1]],
        exps,
        exp_names,
        attr_arch,
        "unseen_archs_masked",
        prefix="fgsm8",
        annot=True,
        title_suff="dataset: cifar10, setting: origin, attacks: fgsm8",
        mask_same=True,
    )
    plot_huge(
        gargs.FGSM_ATTACKS,
        exps,
        exp_names,
        attr_arch,
        "unseen_archs",
        prefix="fgsm",
        annot=False,
        title_suff="dataset: cifar10, setting: origin, attacks: fgsm",
    )
    plot_huge(
        [gargs.FGSM_ATTACKS[1]],
        exps,
        exp_names,
        attr_arch,
        "unseen_archs",
        prefix="fgsm8",
        annot=True,
        title_suff="dataset: cifar10, setting: origin, attacks: fgsm8",
    )

    for exp in gargs.EXPS:
        draw_plot(gargs.ALL_ATTACKS, exp, attr_arch, "all", False)
        draw_plot(gargs.BLACKBOX_ATTACKS, exp, attr_arch, "blackbox", False)
        draw_plot(gargs.WHITEBOX_ATTACKS, exp, attr_arch, "whitebox", False)
        draw_plot(gargs.L2_ATTACKS, exp, attr_arch, "l2", False)
        draw_plot(gargs.LINF_ATTACKS, exp, attr_arch, "linf", False)
        for group in gargs.ALL_GROUP:
            name = group[0]["attack"]
            draw_plot(group, exp, attr_arch, name, True)
