import os

import torch

import global_args as gargs
import training_utils


def check_group(group, atks):
    return all(a in atks for a in group)


def grep_attack(dataset, arch, setting, attacks, attack_name):
    da_dir = os.path.join(gargs.GREP_DIR, f"{dataset}_{arch}")
    setting_dir = os.path.join(da_dir, setting)
    save_dir = os.path.join(da_dir, f"grouped_attack_{setting}", attack_name)

    os.makedirs(save_dir, exist_ok=True)

    for name in gargs.SPLIT_OUTPUT_NAMES:
        items = []

        save_path = os.path.join(save_dir, name)
        if os.path.exists(save_path):
            continue

        for atk in attacks:
            atk_name = training_utils.get_attack_name(atk)
            atk_dir = os.path.join(setting_dir, atk_name)

            path = os.path.join(atk_dir, name)
            print(f"Load from {path}.")
            item = torch.load(path)
            items.append(item)

        items = torch.cat(items, axis=0)
        print(items.shape)

        torch.save(items, save_path)
        print(f"Dump to {save_path}.")


def grep_class(dataset, setting, attack_name, model_archs, archs_name):
    save_dir = os.path.join(
        gargs.GREP_DIR, f"{dataset}_{archs_name}", setting, attack_name
    )
    os.makedirs(save_dir, exist_ok=True)

    for name in gargs.SPLIT_OUTPUT_NAMES:
        save_path = os.path.join(save_dir, name)
        if os.path.exists(save_path):
            print(f"{save_path} exists!")
            continue
        print(f"Dump to {save_path}.")

        flag = True
        for idx_arch, arch in enumerate(model_archs):
            attack_dir = os.path.join(
                gargs.GREP_DIR, f"{dataset}_{arch}", setting, attack_name
            )
            path = os.path.join(attack_dir, name)
            if not os.path.exists(path):
                flag = False
                print(f"{path} does not exists!")
                break

        if not flag:
            continue

        items = []
        for idx_arch, arch in enumerate(model_archs):
            attack_dir = os.path.join(
                gargs.GREP_DIR, f"{dataset}_{arch}", setting, attack_name
            )
            path = os.path.join(attack_dir, name)
            print(f"Load from {path}.")
            item = torch.load(path)
            if "attr_labels" in name:
                item = (
                    torch.cat(
                        [item, torch.zeros(item.shape[0], 1, dtype=int) + idx_arch],
                        axis=1,
                    )
                    .detach()
                    .long()
                )
            items.append(item)

        items = torch.cat(items, axis=0).detach()
        print(items.shape)

        torch.save(items, save_path)
        print(f"Dump to {save_path} success.")


if __name__ == "__main__":
    for attacks in gargs.ALL_GROUP:
        name = attacks[0]["attack"]
        if attacks[0].get("norm") == "L2":
            name += "l2"
        grep_attack("cifar10", "resnet9", "origin", attacks, name)
    selected_groups = [
        gargs.PGD_ATTACKS,
        gargs.PGD_L2_ATTACKS,
        gargs.CW_ATTACKS,
        gargs.ZOSIGNSGD_ATTACKS,
    ]
    import itertools

    selected_attacks = list(itertools.chain(*selected_groups))
    grep_attack("cifar10", "resnet9", "origin", selected_attacks, "selected")

    for data in ["cifar10", "cifar100"]:
        for attack in gargs.WHITEBOX_ATTACKS:
            attack_name = training_utils.get_attack_name(attack)
            grep_class(
                data, "origin", attack_name, gargs.VALID_ARCHITECTURES, "full_archs"
            )
            # grep_class(data, "origin", attack_name, ["resnet9", "vgg11", "resnet20s"], "partial_archs")
