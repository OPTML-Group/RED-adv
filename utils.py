import os
import time
import torch
import shutil
import random
import numpy as np


def count_element(arr):
    keys = torch.unique(arr)
    cnt = []
    for key in keys:
        cnt.append((arr == key).int().sum())
    return np.array(cnt)


def count_element_vector(arr):  # hard coded
    q = arr[:, 0] * 15 + arr[:, 1] * 5 + arr[:, 2]
    return count_element(q)


def get_datasets_from_tensor_with_cnt(data, label, cnt, cuda=False):
    train_data, train_label, test_data, test_label = [], [], [], []
    st = 0
    for num in cnt:
        test_num = num // 10

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


def run_commands(gpus, commands, call=False, dir="commands", shuffle=True, delay=0.5):
    if len(commands) == 0:
        return 
    if os.path.exists(dir):
        shutil.rmtree(dir)
    if shuffle:
        random.shuffle(commands)
        random.shuffle(gpus)
    os.makedirs(dir, exist_ok=True)

    fout = open('stop_{}.sh'.format(dir), 'w')
    print("kill $(ps aux|grep 'bash " + dir + "'|awk '{print $2}')", file=fout)
    fout.close()

    n_gpu = len(gpus)
    for i, gpu in enumerate(gpus):
        i_commands = commands[i::n_gpu]
        if len(i_commands) == 0:
            continue 
        prefix = "CUDA_VISIBLE_DEVICES={} ".format(gpu)

        sh_path = os.path.join(dir, "run{}.sh".format(i))
        fout = open(sh_path, 'w')
        for com in i_commands:
            print(prefix + com, file=fout)
        fout.close()
        if call:
            os.system("bash {}&".format(sh_path))
            time.sleep(delay)