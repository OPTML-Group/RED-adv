
import torch
from torch.cuda.amp import autocast
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

import os
import attr_models
import global_args as gargs

import arg_parser


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_data(args, train):
    suffix = "train" if train else "test"

    data = torch.load(os.path.join(
        args.input_folder, f"{args.input_type}_{suffix}.pt")).detach().cuda()
    label = torch.load(os.path.join(
        args.input_folder, f"attr_labels_{suffix}.pt")).detach().cuda()
    dataset = torch.utils.data.TensorDataset(data, label)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=train)
    print(data.shape, label.shape)
    return dataloader


def get_model(args, n_class, n_output):
    model = attr_models.get_model(
        args.attr_arch,
        num_channel=gargs.DATASET_NUM_CHANNEL[args.dataset],
        num_class=n_class,
        num_output=n_output,
        img_size=gargs.DATASET_INPUT_SIZE[args.dataset]
    ).cuda()
    return model


def train_epoch(model, train_dl, criterion, optimizer):
    model.train()
    train_size = train_dl.dataset.tensors[0].shape[0]
    correct_train = 0
    for x,  y in train_dl:
        with autocast():
            assert not x.isnan().any()
            optimizer.zero_grad()
            x = x.cuda()
            y = y.cuda().long()
            outputs = model(x)
            correct_train += (outputs.argmax(-2) == y).sum(0)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

    train_acc = list((correct_train/train_size*100).cpu().numpy())
    return train_acc


def validate(model, test_dl):
    model.eval()
    test_size = test_dl.dataset.tensors[0].shape[0]
    correct_test = 0
    for x,  y in test_dl:
        with autocast():
            assert not x.max().isnan().any()
            x = x.cuda()
            y = y.cuda().long()
            outputs = model(x)
            correct_test += (outputs.argmax(-2) == y).sum(0)

    test_acc = list((correct_test/test_size*100).cpu().numpy())
    return test_acc


def main():
    args = arg_parser.parse_args_model_parsing(train=True)
    set_seed(args.seed)

    # train data
    train_dl = get_data(args, train=True)

    # test data
    test_dl = get_data(args, train=False)

    n_class = test_dl.dataset.tensors[1].max() + 1
    n_output = test_dl.dataset.tensors[1].shape[1]
    model = get_model(args, n_class, n_output)
    print(f"class num: {n_class}, output num: {n_output}")

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = torch.nn.CrossEntropyLoss()

    os.makedirs(args.save_folder, exist_ok=True)

    fout = open(os.path.join(args.save_folder, "output.log"), 'w')

    best_acc = 0
    best_epoch = 0

    for i in range(args.epochs):
        print(f"Epoch: {i}", file=fout)
        train_acc = train_epoch(model, train_dl, criterion, optimizer)
        train_print = ", ".join("{:.2f}".format(x) for x in train_acc)
        print(f"Epoch: {i}, Train acc: {train_print}")
        print(f"Train acc: {train_print}", file=fout)
        scheduler.step()

        test_acc = validate(model, test_dl)

        if np.mean(test_acc) > best_acc:
            best_acc = np.mean(test_acc)
            best_epoch = i
            torch.save(model.state_dict(), os.path.join(
                args.save_folder, "best.pt"))

        test_print = ", ".join("{:.2f}".format(x) for x in test_acc)

        print(
            f"Epoch: {i}, Test acc: {test_print}, best acc: {best_acc:.2f}, best epoch: {best_epoch}")
        print(
            f"Test acc: {test_print}, best acc: {best_acc:.2f}, best epoch: {best_epoch}", file=fout)

    torch.save(model.state_dict(), os.path.join(args.save_folder, "final.pt"))
    # save path


if __name__ == "__main__":
    main()
