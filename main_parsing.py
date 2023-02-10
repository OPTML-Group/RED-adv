import os

import torch

import utils
import models
import trainer
import arg_parser

adv_name = "x_adv.pt"
delta_name = "delta.pt"
label_name = "attr_labels.pt"


def main():
    args = arg_parser.parse_args_model_parsing()
    dir = args.dataset_dir
    utils.set_seed(args.seed)

    if args.input_type == "x_adv":
        suffix = adv_name
    elif args.input_type == "delta":
        suffix = delta_name
    else:
        exit(0)

    with torch.no_grad():
        data = torch.load(os.path.join(dir, suffix)).float()
        mean = torch.mean(data, dim=[0, 2, 3])
        std = torch.std(data, dim=[0, 2, 3])
        print(mean, std)
        data = ((data - mean[:, None, None]) / std[:, None, None]).detach()
        label = torch.load(os.path.join(dir, label_name)).long().detach()

    n_channels = data.shape[1]
    n_outputs = label.shape[1]

    cnt = utils.count_element_vector(label)
    train_set, test_set = utils.get_datasets_from_tensor_with_cnt(
        data, label, cnt, cuda=True)

    loaders = {
        "train": torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True),
        "test": torch.utils.data.DataLoader(
            test_set, batch_size=args.batch_size, shuffle=False)
    }

    model = models.ConvNet(num_channels=n_channels, num_classes=3,
                           num_outputs=n_outputs).cuda()

    train_params = trainer.get_training_params(model, "model_parser", args)

    trainer.train_with_rewind(train_params, loaders, args)


if __name__ == "__main__":
    main()
