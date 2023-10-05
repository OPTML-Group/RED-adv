import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

import attr_models
import global_args as gargs


class EvaluateParsing:
    def __init__(self, dataset, arch, atk_name, attr_arch, setting, input_type):
        self.setting = setting
        self.arch = arch
        self.input_type = input_type
        self.attr_arch = attr_arch
        self.atk_name = atk_name
        self.dataset = dataset
        self.data_arch = f"{dataset}_{arch}"

    def get_attr_model(self):
        suffix = "best"
        self.denoiser = None
        attr_model_dir = os.path.join(
            gargs.PARSING_DIR,
            self.attr_arch,
            self.data_arch,
            self.setting,
            self.atk_name,
            self.input_type,
        )

        if self.input_type == "denoise":
            suffix = "final"
            attr_denoiser_path = os.path.join(attr_model_dir, f"denoiser_{suffix}.pt")
            denoiser = attr_models.DnCNN(
                image_channels=3, depth=17, n_channels=64
            ).cuda()
            denoiser.load_state_dict(torch.load(attr_denoiser_path))

        attr_model_path = os.path.join(attr_model_dir, f"{suffix}.pt")

        self.n_class = 3
        self.n_output = 3
        attr_model = attr_models.get_model(
            name=self.attr_arch,
            num_channel=gargs.DATASET_NUM_CHANNEL[self.dataset],
            num_class=self.n_class,
            num_output=self.n_output,
            img_size=gargs.DATASET_INPUT_SIZE[self.dataset],
        ).cuda()
        print(f"Load from {attr_model_path}")
        attr_model.load_state_dict(torch.load(attr_model_path))
        self.attr_model = attr_model

    def predict_attr_batch(self, input):
        if self.denoiser:
            self.denoiser.eval()
            attr_input = input - self.denoiser(input)
        else:
            attr_input = input
        self.attr_model.eval()
        return self.attr_model(attr_input).argmax(-2).cpu()

    def predict_attr(self, inputs):
        dataset = torch.utils.data.TensorDataset(inputs.cuda())
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=2048, shuffle=False
        )
        pred_labels = []
        for (x,) in dataloader:
            pred_labels.append(self.predict_attr_batch(x))
        pred_labels = torch.cat(pred_labels, axis=0)
        return pred_labels

    def load_grep_data(self, grep_dir=None):
        if grep_dir is None:
            grep_dir = os.path.join(
                gargs.GREP_DIR, self.data_arch, self.setting, self.atk_name
            )

        input_name = "delta" if self.input_type == "delta" else "x_adv"
        inputs = torch.load(os.path.join(grep_dir, f"{input_name}_test.pt"))
        attr_labels = torch.load(os.path.join(grep_dir, f"attr_labels_test.pt"))

        return inputs, attr_labels

    def get_confusion(self, data_dir=None):
        inputs, labels = self.load_grep_data(data_dir)
        preds = self.predict_attr(inputs)
        shape_pred = preds.numpy().max(axis=0) + 1
        shape_labels = labels.numpy().max(axis=0) + 1

        cnt_pred = 0
        cnt_label = 0
        for i in range(len(shape_pred)):
            cnt_pred = cnt_pred * shape_pred[i] + preds[:, i]
        for i in range(len(shape_labels)):
            cnt_label = cnt_label * shape_labels[i] + labels[:, i]

        confusion = np.zeros([np.prod(shape_labels), np.prod(shape_pred)])
        for pred, label in zip(cnt_pred, cnt_label):
            confusion[label, pred] += 1
        self.confusion = confusion

    def pre_process(self):
        self.get_attr_model()

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        plt.clf()
        sns.heatmap(self.confusion)
        plt.savefig(
            os.path.join(save_dir, "confusion" + ".png"), bbox_inches="tight", dpi=300
        )

    def main(self):
        self.pre_process()
        self.get_confusion()


if __name__ == "__main__":
    save_dir = "./figs/confusion_matrix"
    # import shutil
    # shutil.rmtree(save_dir, ignore_errors=True)
    dataset = "cifar10"
    arch = "resnet9"
    atk_name = "attack_pgd_eps_8_alpha_1"
    attr_arch = "conv4"
    setting = "origin"
    input_type = "delta"
    eval = EvaluateParsing(dataset, arch, atk_name, attr_arch, setting, input_type)
    eval.main()
    eval.save(os.path.join(save_dir, atk_name, setting, input_type))

    setting = "robust"
    eval = EvaluateParsing(dataset, arch, atk_name, attr_arch, setting, input_type)
    eval.main()
    eval.save(os.path.join(save_dir, atk_name, setting, input_type))
    setting = "origin"

    input_type = "x_adv"
    eval = EvaluateParsing(dataset, arch, atk_name, attr_arch, setting, input_type)
    eval.main()
    eval.save(os.path.join(save_dir, atk_name, setting, input_type))

    input_type = "delta"
    atk_name = "attack_fgsm_eps_8"
    eval = EvaluateParsing(dataset, arch, atk_name, attr_arch, setting, input_type)
    eval.main()
    eval.save(os.path.join(save_dir, atk_name, setting, input_type))

    atk_name = "attack_zosignsgd_eps_8_norm_Linf"
    eval = EvaluateParsing(dataset, arch, atk_name, attr_arch, setting, input_type)
    eval.main()
    eval.save(os.path.join(save_dir, atk_name, setting, input_type))

    atk_name = "attack_pgd_eps_4_alpha_0.5"
    eval = EvaluateParsing(dataset, arch, atk_name, attr_arch, setting, input_type)
    eval.main()
    eval.save(os.path.join(save_dir, atk_name, setting, input_type))
