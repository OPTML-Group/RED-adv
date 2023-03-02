import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.utils.data
from torch.cuda.amp import autocast

import global_args as gargs
import models
import pruner
import attr_models
import training_utils


class TransferAttack:
    def __init__(self, dataset, arch, atk_name, attr_arch, setting, tp):
        self.setting = setting
        self.arch = arch
        self.tp = tp
        self.attr_arch = attr_arch
        self.atk_name = atk_name
        self.dataset = dataset
        self.data_arch = f"{dataset}_{arch}"

    def get_model_lists(self):
        ks = gargs.KERNEL_SIZES
        acts = gargs.ACTIVATION_FUNCTIONS
        prunes = gargs.PRUNING_RATIOS

        self.model_names = []
        self.model_setting = {}
        self.name_mapping = {}
        robust = "robust" in self.setting

        for idx_k, k in enumerate(ks):
            for idx_a, act in enumerate(acts):
                for idx_p, prune in enumerate(prunes):
                    model_name = training_utils.get_model_name(
                        2, k, act, prune, robust=robust)
                    self.model_names.append(((idx_k, idx_a, idx_p), model_name))
                    self.model_setting[model_name] = (k, act, prune)
                    self.name_mapping[(idx_k, idx_a, idx_p)] = model_name

    def get_attr_model(self):
        attr_model_path = os.path.join(
            gargs.PARSING_DIR, self.attr_arch, self.data_arch,
            self.setting, self.atk_name, self.tp, "best.pt")

        self.n_class = 3
        self.n_output = 3
        attr_model = attr_models.get_model(
            name=self.attr_arch,
            num_channel=gargs.DATASET_NUM_CHANNEL[self.dataset],
            num_class=self.n_class,
            num_output=self.n_output,
            img_size=gargs.DATASET_INPUT_SIZE[self.dataset]).cuda()
        print(f"Load from {attr_model_path}")
        attr_model.load_state_dict(torch.load(attr_model_path))
        self.attr_model = attr_model

    def get_model(self, model_name):
        k, act, prune = self.model_setting[model_name]

        class Arg:
            pass
        args = Arg()
        args.num_classes = gargs.DATASET_NUM_CLASSES[self.dataset]
        args.kernel_size = k
        args.act_func = act
        model = models.get_model(arch, args).cuda()
        return model

    def load_model(self, model_name):
        model = self.get_model(model_name)

        last_epoch = 100 if self.dataset == "tinyimagenet" else 75

        model_path = os.path.join(
            gargs.MODEL_DIR, self.data_arch, f"{model_name}_omp_2", f"checkpoint_{last_epoch}.pt")

        print(f"Load model from {model_path}")
        item = torch.load(model_path)["model"]
        current_mask = pruner.extract_mask(item)
        if len(current_mask) > 0:
            pruner.prune_model_custom(model, current_mask)
        model.load_state_dict(item, strict=False)
        return model

    def load_data(self, model_name):
        atk_data_dir = os.path.join(
            gargs.ATK_DIR, self.data_arch, self.atk_name, model_name)
        print(f"Load from {atk_data_dir}.")
        datas = training_utils.load_datas(
            atk_data_dir, gargs.FULL_RESULT_NAMES)
        split_n = int(len(datas[0]) * 0.8)
        d_test = [a[split_n:] for a in datas]
        x_adv, delta, adv_pred, ori_pred, target = d_test
        succ = adv_pred.ne(target)
        corr = ori_pred.eq(target)
        idxs = succ * corr

        x_adv = x_adv[idxs].contiguous()
        input = x_adv if self.tp == "x_adv" else delta[idxs].contiguous()

        self.attr_model.eval()
        pred = self.attr_model(input.cuda()).argmax(-2).cpu()

        datas = (x_adv, target[idxs].contiguous(), pred)

        return datas

    def load_datas(self):
        model_datas = {}
        for label, name in self.model_names:
            model_datas[name] = self.load_data(name)
        self.model_datas = model_datas

    def transfer_attack_single(self, model, model_label, data_label, data_name):
        inputs, targets, preds = self.model_datas[data_name]

        model.eval()
        pred = model(inputs.cuda()).argmax(axis=-1).cpu()
        succ = pred.ne(targets)
        asr = succ.float().mean()
        succ_preds = preds[succ]
        pred_as_data = (succ_preds == torch.LongTensor(
            data_label)).all(dim=-1)  # succ
        pred_as_model = (succ_preds == torch.LongTensor(
            model_label)).all(dim=-1)  # mis class
        succ_rate = pred_as_data.float().mean()
        transfer_rate = pred_as_model.float().mean()
        return asr, succ_rate, transfer_rate

    def transfer_attack_all(self):
        n_item = len(self.model_names)
        results = np.zeros([n_item, n_item, 3])
        for idx_model, (model_label, model_name) in enumerate(self.model_names):
            model = self.load_model(model_name)
            for idx_data, (data_label, data_name) in enumerate(self.model_names):
                asr, succ_rate, transfer_rate = self.transfer_attack_single(
                    model, model_label, data_label, data_name)
                results[idx_data, idx_model] = np.array((asr, succ_rate, transfer_rate))

        self.attack_results = results

    def pre_process(self):
        self.get_model_lists()
        self.get_attr_model()
        # self.load_grep_datas()
        self.load_datas()

    def main(self):
        self.pre_process()
        self.transfer_attack_all()

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        display_names = [x[1] for x in self.model_names]
        for i, name in enumerate(["asr", "success", "transfer"]):
            a = self.attack_results[:, :, i]
            plt.clf()
            sns.heatmap(a, xticklabels=[], yticklabels=[])
            plt.ylabel("Origin Victim Model", fontsize=13)
            plt.xlabel("Transffered Victim Model", fontsize=13)
            # plt.xticks(rotation=45, ha='right')
            plt.savefig(os.path.join(save_dir, name + ".png"), bbox_inches='tight', dpi=300)
        
        xs, ys, names = [], [], []
        for i, name in enumerate([None, "Parsed as Origin Victim Model", "Parsed as Transferred Victim Model"]):
            if i == 0:
                continue
            a = self.attack_results[:, :, i]
            x = np.zeros((a.shape[0], a.shape[0] - 1))
            t = np.ones_like(a, dtype=bool)
            for i in range(a.shape[0]):
                t[i, i] = False
                x[i] += a[i, i]
            a = a.reshape(-1)
            t = t.reshape(-1)
            x = x.reshape(-1)
            y = a[t]
            xs.append(x)
            ys.append(y)
            names += [name] * y.shape[0]
        xs = np.concatenate(xs, axis=0) * 100
        ys = np.concatenate(ys, axis=0) * 100
        names = np.array(names)
        plt.clf()
        plt.plot([-10, 110], [-10, 110], linestyle='--', color = 'black')
        x_name = "Parsing Accuracy on Victim Model(%)"
        y_name = "Ratio of Parsing Result(%)"
        legend_name = "Parsing Result Type"
        sns.scatterplot(data={x_name: xs, y_name: ys, legend_name: names}, x=x_name, y=y_name, hue=legend_name, style=legend_name)
        plt.xlim(-5, 105)
        plt.ylim(-5, 105)
        plt.savefig(os.path.join(save_dir, "correlation" + ".png"), bbox_inches='tight', dpi=300)


if __name__ == "__main__":
    dataset = "cifar10"
    arch = "resnet9"
    atk_name = "attack_pgd_eps_8_alpha_1"
    attr_arch = "conv4"
    setting = "origin"
    setting = "robust"
    tp = "delta"
    # eval = TransferAttack(dataset, arch, atk_name, attr_arch, setting, tp)
    # eval.main()
    # eval.save(os.path.join("./figs/transfer_attacks", atk_name, setting))
    
    setting = "robust"
    # eval = TransferAttack(dataset, arch, atk_name, attr_arch, setting, tp)
    # eval.main()
    # eval.save(os.path.join("./figs/transfer_attacks", atk_name, setting))
    setting = "origin"

    tp = "x_adv"
    eval = TransferAttack(dataset, arch, atk_name, attr_arch, setting, tp)
    eval.main()
    eval.save(os.path.join("./figs/transfer_attacks", atk_name, setting, tp))

    tp = "delta"
    atk_name = "attack_fgsm_eps_8"
    # eval = TransferAttack(dataset, arch, atk_name, attr_arch, setting, tp)
    # eval.main()
    # eval.save(os.path.join("./figs/transfer_attacks", atk_name, setting))

    atk_name = "attack_zosignsgd_eps_8_norm_Linf"
    # eval = TransferAttack(dataset, arch, atk_name, attr_arch, setting, tp)
    # eval.main()
    # eval.save(os.path.join("./figs/transfer_attacks", atk_name, setting))
    
    atk_name = "attack_pgd_eps_4_alpha_0.5"
    # eval = TransferAttack(dataset, arch, atk_name, attr_arch, setting, tp)
    # eval.main()
    # eval.save(os.path.join("./figs/transfer_attacks", atk_name, setting))