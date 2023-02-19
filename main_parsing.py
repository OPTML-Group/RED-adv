# # deprecated
# import os

# import torch

# import utils
# import models
# import trainer
# import arg_parser

# adv_name = "x_adv.pt"
# delta_name = "delta.pt"
# label_name = "attr_labels.pt"

# def grep_datas(dir, name):
#     ks = [3, 5, 7]
#     acts = ["relu", "tanh", "elu"]
#     prunes = ["0.0", "0.375", "0.375_struct", "0.625", "0.625_struct"]

#     datas = []
#     labels = []

#     for idx_k, k in enumerate(ks):
#         for idx_a, act in enumerate(acts):
#             for idx_p, prune in enumerate(prunes):
#                 dir_name = f"seed2_kernel{k}_act{act}_prune{prune}"
#                 path = os.path.join(dir, dir_name, name)
#                 assert os.path.exists(path)
#                 item = torch.load(path)
#                 assert item.ndim == 4
#                 attr_label = torch.Tensor([[idx_k, idx_a, idx_p]] * item.shape[0])
#                 assert attr_label.ndim == 2 and attr_label.shape[1] == 3
#                 datas.append(item)
#                 labels.append(attr_label)

#     datas = torch.cat(datas, axis=0).float()
#     labels = torch.cat(labels, axis=0).long()

#     assert datas.shape[0] == labels.shape[0]

#     torch.save(datas, os.path.join(dir, name))
#     torch.save(labels, os.path.join(dir, label_name))
#     return datas, labels


# def load_data(args):
#     if args.input_type == "x_adv":
#         suffix = adv_name
#     elif args.input_type == "delta":
#         suffix = delta_name
#     else:
#         exit(0)
#     dir = args.dataset_dir
#     input_path = os.path.join(dir, suffix)

#     if not os.path.exists(input_path):
#         data, label = grep_datas(dir, suffix)
#     else:
#         data = torch.load(input_path)
#         label = torch.load(os.path.join(dir, label_name))

#     with torch.no_grad():
#         data = data.float()
#         label = label.long().detach()
#         mean = torch.mean(data, dim=[0, 2, 3])
#         std = torch.std(data, dim=[0, 2, 3])
#         print(mean, std)
#         data = ((data - mean[:, None, None]) / std[:, None, None]).detach()
#     return data, label


# def main():
#     args = arg_parser.parse_args_model_parsing()
#     utils.set_seed(args.seed)

#     data, label = load_data(args)
#     exit(0)

#     n_channels = data.shape[1]
#     n_outputs = label.shape[1]

#     cnt = utils.count_element_vector(label)
#     train_set, test_set = utils.get_datasets_from_tensor_with_cnt(
#         data, label, cnt, cuda=True)

#     loaders = {
#         "train": torch.utils.data.DataLoader(
#             train_set, batch_size=args.batch_size, shuffle=True),
#         "test": torch.utils.data.DataLoader(
#             test_set, batch_size=args.batch_size, shuffle=False)
#     }

#     model = models.ConvNet(num_channels=n_channels, num_classes=5,
#                            num_outputs=n_outputs).cuda()

#     train_params = trainer.get_training_params(
#         model, args.input_type, args, use_scaler=False)

#     trainer.train_with_rewind(train_params, loaders, args)


# if __name__ == "__main__":
#     main()
