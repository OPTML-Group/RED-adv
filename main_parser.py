
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

    if args.input_type == "denoise":
        prefixes = ["x_adv", "delta", "attr_labels"]
    else:
        prefixes = [args.input_type, "attr_labels"]

    datas = [torch.load(os.path.join(
        args.input_folder, f"{prefix}_{suffix}.pt")).detach().cuda() for prefix in prefixes]
    dataset = torch.utils.data.TensorDataset(*datas)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=train)
    for data in datas:
        print(data.shape)
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


def train_epoch(model, train_dl, criterion, optimizer, denoiser=None, denoise_criterion=None, denoise_optimizer=None, gamma1=None, frozen_attr=False):
    model.train()
    total_loss = 0
    if denoiser:
        denoiser.train()

    train_size = train_dl.dataset.tensors[0].shape[0]
    correct_train = 0
    for datas in train_dl:
        input = datas[0].cuda()
        gt = datas[-1].cuda().long()
        with autocast():
            assert not input.isnan().any()
            losses = []
            if denoiser:
                delta = datas[1].cuda()
                denoised = denoiser(input)
                attr_input = input - denoised
                losses.append(gamma1 * denoise_criterion(attr_input, delta))
            else:
                attr_input = input

            outputs = model(attr_input)

            if not frozen_attr:
                losses.append(criterion(outputs, gt))
            losses = torch.stack(losses)

            loss = torch.sum(losses)
            loss.backward()

            if not frozen_attr:
                optimizer.step()
                optimizer.zero_grad()
            if denoiser:
                denoise_optimizer.step()
                denoise_optimizer.zero_grad()

            with torch.no_grad():
                total_loss += losses * input.shape[0]
                correct_train += (outputs.argmax(-2) == gt).sum(0)

    train_acc = list((correct_train / train_size * 100).cpu().numpy())
    avg_loss = total_loss / train_size
    return train_acc, avg_loss


def validate(model, test_dl, denoiser=None):
    model.eval()
    if denoiser:
        denoiser.eval()

    test_size = test_dl.dataset.tensors[0].shape[0]
    correct_test = 0

    for datas in test_dl:
        input = datas[0].cuda()
        gt = datas[-1].cuda().long()
        with autocast():
            assert not input.isnan().any()
            if denoiser:
                denoised = denoiser(input)
                attr_input = input - denoised
            else:
                attr_input = input

            outputs = model(attr_input)
            correct_test += (outputs.argmax(-2) == gt).sum(0)

    test_acc = list((correct_test/test_size*100).cpu().numpy())
    return test_acc


def main():
    args = arg_parser.parse_args_model_parsing(train=True)
    set_seed(args.seed)

    # train data
    train_dl = get_data(args, train=True)

    # test data
    test_dl = get_data(args, train=False)

    n_class = test_dl.dataset.tensors[-1].max() + 1
    n_output = test_dl.dataset.tensors[-1].shape[1]

    model = get_model(args, n_class, n_output)
    denoiser, d_optimizer, d_scheduler, d_criterion = None, None, None, None
    if args.input_type == "denoise":
        denoiser = attr_models.DnCNN(
            image_channels=3, depth=17, n_channels=64).cuda()
        item = torch.load(args.pretrained_denoiser_path)
        denoiser.load_state_dict(item['state_dict'])

        pretrain_attr_path = os.path.join(os.path.dirname(args.save_folder), "delta", "best.pt")
        assert os.path.exists(pretrain_attr_path)
        item = torch.load(pretrain_attr_path)
        model.load_state_dict(item)
    else:
        denoiser = None

    print(f"class num: {n_class}, output num: {n_output}")

    os.makedirs(args.save_folder, exist_ok=True)

    fout = open(os.path.join(args.save_folder, "output.log"), 'w')


    if denoiser:
        # denoiser pretrain
        d_optimizer = optim.SGD(denoiser.parameters(), lr=args.denoiser_pretrain_lr)
        d_scheduler = CosineAnnealingLR(d_optimizer, T_max=args.denoiser_pretrain_epoch)
        d_criterion = torch.nn.L1Loss(
            size_average=None, reduce=None, reduction='mean')
        for i in range(args.denoiser_pretrain_epoch):
            train_acc, train_loss = train_epoch(model, train_dl, None, None,
                                                denoiser, d_criterion, d_optimizer, args.gamma1, frozen_attr=True)
            train_acc_print = ", ".join("{:.2f}".format(x) for x in train_acc)
            train_loss_print = ", ".join("{:.6f}".format(x) for x in train_loss)
            d_scheduler.step()
            print(
                f"Pre Epoch: {i}, Train acc: {train_acc_print}, Train loss: {train_loss_print}")

            test_acc = validate(model, test_dl, denoiser)

            test_print = ", ".join("{:.2f}".format(x) for x in test_acc)

            print(
                f"Pre Epoch: {i}, Test acc: {test_print}")


    if denoiser:
        optimizer = optim.SGD(model.parameters(), lr=args.parser_cotrain_lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.cotrain_epoch)
        criterion = torch.nn.CrossEntropyLoss()
        d_optimizer = optim.SGD(denoiser.parameters(), lr=args.denoiser_cotrain_lr)
        d_scheduler = CosineAnnealingLR(d_optimizer, T_max=args.cotrain_epoch)
        d_criterion = torch.nn.L1Loss(
            size_average=None, reduce=None, reduction='mean')
        n_epochs = args.cotrain_epoch
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
        criterion = torch.nn.CrossEntropyLoss()
        n_epochs = args.epochs

    best_acc = 0
    best_epoch = 0

    for i in range(n_epochs):
        print(f"Epoch: {i}", file=fout)
        train_acc, train_loss = train_epoch(model, train_dl, criterion, optimizer,
                                            denoiser, d_criterion, d_optimizer, args.gamma1)
        train_acc_print = ", ".join("{:.2f}".format(x) for x in train_acc)
        train_loss_print = ", ".join("{:.6f}".format(x) for x in train_loss)

        print(
            f"Epoch: {i}, Train acc: {train_acc_print}, Train loss: {train_loss_print}")
        print(
            f"Train acc: {train_acc_print}, Train loss: {train_loss_print}", file=fout)
        scheduler.step()
        if denoiser:
            d_scheduler.step()

        test_acc = validate(model, test_dl, denoiser)

        if np.mean(test_acc) > best_acc:
            best_acc = np.mean(test_acc)
            best_epoch = i
            torch.save(model.state_dict(), os.path.join(
                args.save_folder, "best.pt"))
            if denoiser:
                torch.save(denoiser.state_dict(), os.path.join(
                    args.save_folder, "denoiser_best.pt"))

        test_print = ", ".join("{:.2f}".format(x) for x in test_acc)

        print(
            f"Epoch: {i}, Test acc: {test_print}, best acc: {best_acc:.2f}, best epoch: {best_epoch}")
        print(
            f"Test acc: {test_print}, best acc: {best_acc:.2f}, best epoch: {best_epoch}", file=fout)

    torch.save(model.state_dict(), os.path.join(args.save_folder, "final.pt"))
    if denoiser:
        torch.save(denoiser.state_dict(), os.path.join(args.save_folder, "denoiser_final.pt"))
    # save path


if __name__ == "__main__":
    main()
