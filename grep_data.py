import os
import shutil

import torch

import global_args as gargs
import run

partial_result_names = ["x_adv.pt", "delta.pt"]
full_result_names = ["adv_all.pt", "delta_all.pt",
                     "adv_pred.pt", "ori_pred.pt", "targets.pt"]
output_names = ["x_adv.pt", "delta.pt", "attr_labels.pt"]


def _check_data_exists(dir, names):
    for name in names:
        if not os.path.exists(os.path.join(dir, name)):
            return False
    return True


def _load_datas(dir, names):
    ret = []
    for name in names:
        path = os.path.join(dir, name)
        item = torch.load(path)
        ret.append(item)
    return ret


def _save_datas(dir, names, datas):
    ret = []
    for name, data in zip(names, datas):
        path = os.path.join(dir, name)
        item = torch.save(data, path)
    return ret


def _check_datas(datas, after=False, log_path=None):
    n_items = datas[0].shape[0]
    # check adv and delta shape
    assert datas[0].shape == datas[1].shape
    # check item number
    for data in datas:
        assert data.shape[0] == n_items
    if len(datas) > 2:  # full data
        # check pred shape
        if after:
            for pred in datas[2:]:
                assert pred.ndim == 1
            assert log_path is not None and os.path.exists(log_path)
            with open(log_path, 'r') as fin:
                ncs = float(fin.readline()[:-1].split(': ')[-1])
                ns = float(fin.readline()[:-1].split(': ')[-1])
            succ = datas[2].ne(datas[4])
            corr = datas[3].eq(datas[4])
            n_correct_success = sum(succ * corr).item() / 100
            n_success = sum(succ).item() / 100

            assert abs(n_correct_success - ncs) < 1e-8
            assert abs(n_success - ns) < 1e-8

        else:
            for pred in datas[2:]:
                assert pred.ndim == 1 or pred.ndim == 2 and pred.shape[1] == 10

def _backup(dir):
    shutil.rmtree('./temp', ignore_errors=True)
    os.system(f"cp -r {dir} ./temp")

def clean_up_result(dir_path):
    if _check_data_exists(dir_path, full_result_names):
        print(f"Check '{dir_path}'")
        datas = _load_datas(dir_path, full_result_names)
        _check_datas(datas)
        for idx, data in enumerate(datas):
            datas[idx] = data.detach()

        if datas[3].ndim == 2:
            print(f"argmax datas[3]!!")
            datas[3] = datas[3].argmax(dim=1)
        _check_datas(datas, after=True, log_path=os.path.join(dir_path, 'attack_acc.log'))

        # over write!!! careful!!!!
        _backup(dir_path)
        _save_datas(dir_path, full_result_names, datas)

        print(f"Check '{dir_path}'")
        datas = _load_datas(dir_path, full_result_names)
        _check_datas(datas, after=True, log_path=os.path.join(dir_path, 'attack_acc.log'))
        print(f"Check '{dir_path}' success")
        for name in partial_result_names:
            path = os.path.join(dir_path, name)
            if os.path.exists(path):
                if False:
                    print(f"Remove {path}?")
                    from IPython import embed
                    embed()
                os.remove(path)

def _clean_up_all():
    path = gargs.ATK_DIR
    atk_names = os.listdir(path)
    for atk_name in atk_names:
        atk_path = os.path.join(path, atk_name)
        for output_name in output_names:
            output_path = os.path.join(atk_path, output_name)
            if os.path.exists(output_path):
                print(output_path)
                grep_dir = os.path.join(gargs.GREP_DIR, "cifar10_resnet9", "origin", atk_name)
                os.makedirs(grep_dir, exist_ok=True)
                os.system(f"mv {output_path} {grep_dir}")
    #     model_names = os.listdir(atk_path)
    #     for model_name in model_names:
    #         dir_path = os.path.join(atk_path, model_name)
    #         clean_up_result(dir_path)
# if __name__ == "__main__":
#     _clean_up_all()
#     exit(0)

def grep_from_full_result(dir_path, full=False):
    datas = _load_datas(
        dir_path, full_result_names)
    _check_datas(datas, after=True, log_path=os.path.join(dir_path, 'attack_acc.log'))
    x_adv, delta, adv_pred, ori_pred, target = datas

    if full:
        return x_adv, delta

    succ = adv_pred.ne(target)
    corr = ori_pred.eq(target)

    corr_succ_idx = succ * corr
    partial_delta = delta[corr_succ_idx]
    partial_adv = x_adv[corr_succ_idx]
    return partial_adv, partial_delta


def grep_from_partial_result(dir_path):
    datas = _load_datas(dir_path, partial_result_names)
    _check_datas(datas)
    return datas


def _concat_and_save(x_advs, deltas, labels, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    delta_name, x_adv_name, label_name = output_names

    x_advs = torch.cat(x_advs, axis=0).float()
    deltas = torch.cat(deltas, axis=0).float()
    labels = torch.cat(labels, axis=0).long()
    print(x_advs.shape, deltas.shape, labels.shape)
    assert x_advs.shape == deltas.shape and x_advs.shape[0] == labels.shape[0]

    torch.save(x_advs, os.path.join(save_dir, x_adv_name))
    torch.save(deltas, os.path.join(save_dir, delta_name))
    torch.save(labels, os.path.join(save_dir, label_name))


def load_dir_data(dir_path, full=False):
    if _check_data_exists(dir_path, full_result_names):
        return grep_from_full_result(dir_path, full)
    else:
        return grep_from_partial_result(dir_path)


def grep_data_correct(dir, save_dir, robust, full_data):
    ks = [3, 5, 7]
    acts = ["relu", "tanh", "elu"]
    prunes = ["0.0", "0.375", "0.375_struct", "0.625", "0.625_struct"]

    dirs, lbs = [], []

    x_advs, deltas, labels = [], [], []

    # grep dirs
    for idx_k, k in enumerate(ks):
        for idx_a, act in enumerate(acts):
            for idx_p, prune in enumerate(prunes):
                dir_name = f"seed2_kernel{k}_act{act}_prune{prune}"
                if robust:
                    dir_name += "_robust"
                dir_path = os.path.join(dir, dir_name)
                assert _check_data_exists(dir_path, full_result_names) or (not full_data and _check_data_exists(dir_path, partial_result_names))
                dirs.append(dir_path)
                lbs.append([idx_k, idx_a, idx_p])

    for dir, lb in zip(dirs, lbs):
        adv, dt = load_dir_data(dir, full=full_data)
        n_item = adv.shape[0]

        x_advs.append(adv)
        deltas.append(dt)

        labels.append(torch.Tensor([lb] * n_item))

    _concat_and_save(x_advs, deltas, labels, save_dir)

def grep_setting(atk_dir, save_dir, setting_name, robust, full):
    print(f"Setting: {setting_name}")
    for atk in gargs.ATTACKS:
        atk_name = run.get_attack_name(atk)

        grep_atk_dir = os.path.join(atk_dir, atk_name)
        grep_save_dir = os.path.join(save_dir, setting_name, atk_name)

        if not full and _check_data_exists(grep_save_dir, output_names):
            continue

        print(f"Grep from: '{grep_atk_dir}'")
        print(f"Grep to: '{grep_save_dir}'")

        try:
            grep_data_correct(grep_atk_dir, grep_save_dir, robust=robust, full_data=full)
        except AssertionError as inst:
            print(inst)
        except Exception as inst:
            raise inst
        finally:
            print(f"Successful grep to: {grep_save_dir}")


def grep_data_settings():
    dataset_name = "cifar10"
    arch_name = "resnet9"
    data_arch = f"{dataset_name}_{arch_name}"
    atk_dir = gargs.ATK_DIR
    save_dir = os.path.join(gargs.GREP_DIR, data_arch)
    grep_setting(atk_dir, save_dir, "origin", False, False)
    grep_setting(atk_dir, save_dir, "robust", True, False)
    grep_setting(atk_dir, save_dir, "robust_all", True, True)

if __name__ == "__main__":
    grep_data_settings()