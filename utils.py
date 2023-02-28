import os
import time
import torch
import shutil
import random
import numpy as np


def plot_tensorboard(writer, path, obj, idx, labels=None):
    if isinstance(obj, dict):
        writer.add_scalars(path, obj, idx)
    elif type(obj) in np.ScalarType:
        writer.add_scalar(path, obj, idx)
    elif isinstance(obj, np.ndarray) or torch.is_tensor(obj):
        assert obj.ndim == 1
        if labels is None:
            n_item = len(obj)
            labels = [str(i+1) for i in range(n_item)]
        dic = {labels[i]: item for i, item in enumerate(obj)}
        writer.add_scalars(path, dic, idx)
    else:
        raise NotImplemented(
            "Type {} plotting is not implemented!".format(type(obj)))


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def count_element(arr):
    keys = torch.unique(arr)
    cnt = []
    for key in keys:
        cnt.append((arr == key).int().sum())
    return cnt


def count_element_vector(arr):  # hard coded
    q = arr[:, 0] * 15 + arr[:, 1] * 5 + arr[:, 2]
    return count_element(q)


def get_datasets_from_tensor_with_cnt(data, label, cnt, cuda=False):
    train_data, train_label, test_data, test_label = [], [], [], []
    st = 0
    for num in cnt:
        test_num = int(num * 0.2)

        train_data.append(data[st + test_num: st + num])
        train_label.append(label[st + test_num: st + num])
        test_data.append(data[st: st + test_num])
        test_label.append(label[st: st + test_num])

        st += num

    train_data = torch.cat(train_data, dim=0)
    train_label = torch.cat(train_label, dim=0)
    test_data = torch.cat(test_data, dim=0)
    test_label = torch.cat(test_label, dim=0)

    if cuda:
        train_data = train_data.cuda()
        train_label = train_label.cuda()
        test_data = test_data.cuda()
        test_label = test_label.cuda()

    train_set = torch.utils.data.TensorDataset(train_data, train_label)
    test_set = torch.utils.data.TensorDataset(test_data, test_label)

    return train_set, test_set


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Feb. 11, 2023 version
def run_commands(gpus, commands, suffix, call=False, shuffle=True, delay=0.5, ext_command=""):
    command_dir = os.path.join("commands", suffix)
    if len(commands) == 0:
        return
    if os.path.exists(command_dir):
        shutil.rmtree(command_dir)
    if shuffle:
        random.shuffle(commands)
        random.shuffle(gpus)
    os.makedirs(command_dir, exist_ok=True)

    stop_path = os.path.join('commands', 'stop_{}.sh'.format(suffix))
    with open(stop_path, 'w') as fout:
        print("kill $(ps aux|grep 'bash " + command_dir +
              "'|awk '{print $2}')", file=fout)

    n_gpu = len(gpus)
    for i, gpu in enumerate(gpus):
        i_commands = commands[i::n_gpu]
        if len(i_commands) == 0:
            continue
        prefix = "CUDA_VISIBLE_DEVICES={} ".format(gpu)
        ext_command_i = ext_command.format(i=i)

        sh_path = os.path.join(command_dir, "run{}.sh".format(i))
        fout = open(sh_path, 'w')
        for com in i_commands:
            print(prefix + com + ext_command_i, file=fout)
        fout.close()
        if call:
            os.system("bash {}&".format(sh_path))
            time.sleep(delay)
