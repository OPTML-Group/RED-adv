import os

import torch

import run
import global_args as gargs
import grep_data


def check_group(group, atks):
    return all(a in atks for a in group)


def grep_attack(dataset, arch, setting, attacks, attack_name):
    da_dir = os.path.join(gargs.GREP_DIR, f"{dataset}_{arch}")
    setting_dir = os.path.join(da_dir, setting)
    save_dir = os.path.join(da_dir, f"grouped_attack_{setting}", attack_name)
    
    os.makedirs(save_dir, exist_ok=True)

    for name in grep_data.output_names:
        items = []

        save_path = os.path.join(save_dir, name)
        if os.path.exists(save_path):
            continue

        for atk in attacks:
            atk_name = run.get_attack_name(atk)
            atk_dir = os.path.join(setting_dir, atk_name)

            path = os.path.join(atk_dir, name)
            print(f"Load from {path}.")
            item = torch.load(path)
            items.append(item)

        items = torch.cat(items, axis=0)
        print(items.shape)

        torch.save(items, save_path)
        print(f"Dump to {save_path}.")


def grep_class(dataset, setting, attack_name):
    save_dir = os.path.join(gargs.GREP_DIR, f"{dataset}_archs", setting, attack_name)
    os.makedirs(save_dir, exist_ok=True)

    for name in grep_data.output_names:
        save_path = os.path.join(save_dir, name)
        if os.path.exists(save_path):
            print(f"{save_path} exists!")
            continue

        items = []
        for idx_arch, arch in enumerate(gargs.VALID_ARCHITECTURES):
            attack_dir = os.path.join(gargs.GREP_DIR, f"{dataset}_{arch}", setting, attack_name)
            path = os.path.join(attack_dir, name)
            print(f"Load from {path}.")
            item = torch.load(path)
            if name == "attr_labels.pt":
                item = torch.cat([item, torch.ones(item.shape[0], 1) * idx_arch], axis=1).detach()
            items.append(item)

        items = torch.cat(items, axis=0).detach()
        print(items.shape)

        torch.save(items, save_path)
        print(f"Dump to {save_path}.")


if __name__ == "__main__":
    for attacks in gargs.ALL_GROUP:
        name = attacks[0]['attack']
        grep_attack("cifar10", "resnet9", "origin", attacks, name)

    for attack in gargs.WHITEBOX_ATTACKS:
        attack_name = run.get_attack_name(attack)
        grep_class("cifar10", "origin", attack_name)