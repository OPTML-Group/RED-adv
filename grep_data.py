import os
import shutil
import tqdm

import torch

import global_args as gargs
import training_utils

# partial_result_names = ["x_adv.pt", "delta.pt"]
# full_result_names = ["adv_all.pt", "delta_all.pt",
#                      "adv_pred.pt", "ori_pred.pt", "targets.pt"]
# output_names_no_split = ["x_adv.pt", "delta.pt", "attr_labels.pt"]
# output_names_split = ["x_adv_train.pt", "delta_train.pt", "attr_labels_train.pt",
#                       "x_adv_test.pt", "delta_test.pt", "attr_labels_test.pt"]


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
    if training_utils.check_data_exists(dir_path, gargs.FULL_RESULT_NAMES):
        print(f"Check '{dir_path}'")
        datas = training_utils.load_datas(dir_path, gargs.FULL_RESULT_NAMES)
        _check_datas(datas)
        for idx, data in enumerate(datas):
            datas[idx] = data.detach()

        if datas[3].ndim == 2:
            print(f"argmax datas[3]!!")
            datas[3] = datas[3].argmax(dim=1)
        _check_datas(datas, after=True, log_path=os.path.join(
            dir_path, 'attack_acc.log'))

        # over write!!! careful!!!!
        _backup(dir_path)
        training_utils.save_datas(dir_path, gargs.FULL_RESULT_NAMES, datas)

        print(f"Check '{dir_path}'")
        datas = training_utils.load_datas(dir_path, gargs.FULL_RESULT_NAMES)
        _check_datas(datas, after=True, log_path=os.path.join(
            dir_path, 'attack_acc.log'))
        print(f"Check '{dir_path}' success")
        for name in gargs.PARTIAL_RESULT_NAMES:
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
        # for output_name in output_names:
        #     output_path = os.path.join(atk_path, output_name)
        #     if os.path.exists(output_path):
        #         print(output_path)
        #         grep_dir = os.path.join(gargs.GREP_DIR, "cifar10_resnet9", "origin", atk_name)
        #         os.makedirs(grep_dir, exist_ok=True)
        #         os.system(f"mv {output_path} {grep_dir}")
        model_names = os.listdir(atk_path)
        for model_name in model_names:
            dir_path = os.path.join(atk_path, model_name)
            clean_up_result(dir_path)
# if __name__ == "__main__":
#     _clean_up_all()
#     exit(0)


def _get_dataset_name(path):
    for name in ["cifar100", "cifar10", 'tinyimagenet']:
        if name in path:
            return name
    raise NotImplementedError(f"Not implemented! {path}")


_exist_order = {}


def _check_data_order(datas, path):
    origin = datas[0] - datas[1]
    dataset_name = _get_dataset_name(path)
    global _exist_order
    if _exist_order.get(dataset_name) is None:
        _exist_order[dataset_name] = origin
        # print(f"Save order {dataset_name}! {path}")
    else:
        assert (_exist_order[dataset_name] -
                origin).abs().max() < 1e-6, f"Order different {dataset_name}! {path}"
        # print(f"Check order success {dataset_name}! {path}")


def process_full_result(x_adv, delta, adv_pred, ori_pred, target, full):
    if full:
        return (x_adv, delta)
    succ = adv_pred.ne(target)
    corr = ori_pred.eq(target)

    corr_succ_idx = succ * corr
    partial_delta = delta[corr_succ_idx]
    partial_adv = x_adv[corr_succ_idx]
    return (partial_adv, partial_delta)


def grep_from_full_result(dir_path, full, split):
    datas = training_utils.load_datas(dir_path, gargs.FULL_RESULT_NAMES)
    _check_datas(datas, after=True, log_path=os.path.join(
        dir_path, 'attack_acc.log'))
    _check_data_order(datas, dir_path)
    if not split:
        return [process_full_result(*datas, full)]
    split_n = int(len(datas[0]) * 0.8)
    d_train = [a[:split_n] for a in datas]
    d_test = [a[split_n:] for a in datas]
    return [process_full_result(*d_train, full), process_full_result(*d_test, full)]


def grep_from_partial_result(dir_path):
    datas = training_utils.load_datas(dir_path, gargs.PARTIAL_RESULT_NAMES)
    _check_datas(datas)
    return datas


def _concat_and_save_patch(save_dir, datas, names):
    os.makedirs(save_dir, exist_ok=True)

    x_adv_name, delta_name, label_name = names

    x_advs, deltas, labels = datas

    x_advs = torch.cat(x_advs, axis=0).float()
    deltas = torch.cat(deltas, axis=0).float()
    labels = torch.cat(labels, axis=0).long()

    print(x_advs.shape, deltas.shape, labels.shape)
    assert x_advs.shape == deltas.shape and x_advs.shape[0] == labels.shape[0]

    torch.save(x_advs, os.path.join(save_dir, x_adv_name))
    torch.save(deltas, os.path.join(save_dir, delta_name))
    torch.save(labels, os.path.join(save_dir, label_name))


def _concat_and_save(save_dir, datas, split):
    if not split:
        _concat_and_save_patch(save_dir, datas[0], gargs.NO_SPLIT_OUTPUT_NAMES)
    else:
        _concat_and_save_patch(save_dir, datas[0], gargs.SPLIT_OUTPUT_NAMES[:3])
        _concat_and_save_patch(save_dir, datas[1], gargs.SPLIT_OUTPUT_NAMES[3:])


def load_dir_data(dir_path, full, split):
    if split or full or training_utils.check_data_exists(dir_path, gargs.FULL_RESULT_NAMES):
        return grep_from_full_result(dir_path, full, split)
    else:
        raise NotImplementedError("No partial!!!!")
        return grep_from_partial_result(dir_path)


def grep_data_correct(dir, save_dir, robust, full_data, split):
    ks = gargs.KERNEL_SIZES
    acts = gargs.ACTIVATION_FUNCTIONS
    # ["0.0", "0.375", "0.375_struct", "0.625", "0.625_struct"]
    prunes = gargs.PRUNING_RATIOS

    dirs, lbs = [], []

    # grep dirs
    for idx_k, k in enumerate(ks):
        for idx_a, act in enumerate(acts):
            for idx_p, prune in enumerate(prunes):
                dir_name = training_utils.get_model_name(2, k, act, prune, robust=robust)
                dir_path = os.path.join(dir, dir_name)
                assert training_utils.check_data_exists(dir_path, gargs.FULL_RESULT_NAMES)
                # or (not full_data and training_utils.check_data_exists(dir_path, gargs.PARTIAL_RESULT_NAMES))
                dirs.append(dir_path)
                lbs.append([idx_k, idx_a, idx_p])

    export_datas = [[[], [], []]] if not split else [
        [[], [], []], [[], [], []]]

    for dir, lb in tqdm.tqdm(list(zip(dirs, lbs))):
        datas = load_dir_data(dir, full=full_data, split=split)

        for data, export in zip(datas, export_datas):
            n_item = data[0].shape[0]
            export[0].append(data[0])
            export[1].append(data[1])
            export[2].append(torch.LongTensor([lb] * n_item).cpu())

    _concat_and_save(save_dir, export_datas, split)


def grep_setting(dataset, arch, setting_name, attacks, split):
    data_arch = f"{dataset}_{arch}"
    atk_dir = os.path.join(gargs.ATK_DIR, data_arch)
    save_dir = os.path.join(gargs.GREP_DIR, data_arch)
    print(f"Setting: {setting_name}")
    robust = "robust" in setting_name
    full = "all" in setting_name

    output_names = gargs.SPLIT_OUTPUT_NAMES if split else gargs.NO_SPLIT_OUTPUT_NAMES
    for atk in attacks:
        atk_name = training_utils.get_attack_name(atk)

        grep_atk_dir = os.path.join(atk_dir, atk_name)
        grep_save_dir = os.path.join(save_dir, setting_name, atk_name)

        if training_utils.check_data_exists(grep_save_dir, output_names):
            continue

        print(f"Grep from: '{grep_atk_dir}'")
        print(f"Grep to: '{grep_save_dir}'")

        try:
            grep_data_correct(grep_atk_dir, grep_save_dir,
                              robust=robust, full_data=full, split=split)
        except AssertionError as inst:
            print(inst)
        except Exception as inst:
            raise inst
        finally:
            print(f"Successful grep to: {grep_save_dir}")


if __name__ == "__main__":
    for exp in gargs.EXPS:
        grep_setting(exp['data'], exp['arch'], exp['setting'],
                     exp["attacks"], split=True)
