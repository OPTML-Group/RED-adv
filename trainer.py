
import os
import copy
import time
import tqdm
import utils
import shutil

import torch
from torch.utils import tensorboard


def load_checkpoint(train_params, path, model_only=False):
    print("Loading checkpoint from {}".format(path))
    load_dict = torch.load(path)
    if model_only:
        train_params["model"].load_state_dict(item)
    else:
        for key, item in load_dict.items():
            train_params[key].load_state_dict(item)


def save_checkpoint(train_params, path, model_only=False):
    if model_only:
        save_dict = train_params["model"].state_dict()
    else:
        save_dict = {}
        for key, item in train_params.items():
            if hasattr(item, "state_dict"):
                save_dict[key] = item.state_dict()
    torch.save(save_dict, path)


def get_training_params(model, name, args, use_scaler=False):
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    scaler = None
    if use_scaler:
        scaler = torch.cuda.amp.GradScaler()
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    writer = None
    if args.tensorboard:
        log_dir = os.path.join(args.save_dir, name, "tensorboard")
        if not args.resume:
            shutil.rmtree(log_dir, ignore_errors=True)
        writer = tensorboard.SummaryWriter(log_dir=log_dir)

    save_dir = os.path.join(args.save_dir, name)

    train_params = {
        "optimizer": optimizer,
        "scheduler": scheduler,
        "criterion": criterion,
        "scaler": scaler,
        "writer": writer,
        "model": model,
        "name": name,
        "start_epoch": 0,
    }

    if args.resume:
        for epoch in range(args.epochs, 0, -1):
            path = os.path.join(save_dir, f"checkpoint_{epoch}.pt")
            if os.path.exists(path):
                load_checkpoint(train_params, path)
                train_params["start_epoch"] = epoch
                break
    return train_params


def train_epoch(train_params, train_loader, epoch):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    criterion = train_params['criterion']
    optimizer = train_params['optimizer']
    scaler = train_params['scaler']
    writer = train_params['writer']
    model = train_params['model']
    name = train_params['name']

    # switch to train mode
    model.train()
    n_iters = len(train_loader)
    lr = optimizer.state_dict()['param_groups'][0]['lr']

    for image, target in tqdm.tqdm(train_loader):

        image = image.cuda()
        target = target.cuda()

        # compute output
        output = model(image)
        loss = criterion(output, target)
        train_loss = loss.mean()
        optimizer.zero_grad()

        if scaler:
            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            train_loss.backward()
            optimizer.step()

        # measure accuracy and record loss
        with torch.no_grad():
            batch_size = image.shape[0]
            pred = output.data.argmax(axis=1)
            classwise_acc = torch.eq(pred, target).float().mean(axis=0)

            classwise_loss = loss.mean(axis=0)

            losses.update(classwise_loss.cpu().numpy(), batch_size)
            top1.update(classwise_acc.cpu().numpy(), batch_size)

            # if writer is not None:
            #     utils.plot_tensorboard(
            #         writer, 'Training_iter/acc', top1.val, epoch * (n_iters) + i)
            #     utils.plot_tensorboard(
            #         writer, 'Training_iter/loss', losses.val, epoch * (n_iters) + i)

    if writer is not None:
        utils.plot_tensorboard(
            writer, f'{name}_train_epoch/acc', top1.avg, epoch)
        utils.plot_tensorboard(
            writer, f'{name}_train_epoch/loss', losses.avg, epoch)
        utils.plot_tensorboard(
            writer, f'{name}_train_epoch/lr', lr, epoch)

    return top1.avg, losses.avg


def validate(model, val_loader, criterion):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # switch to evaluate mode
    model.eval()

    for image, target in val_loader:
        image = image.cuda()
        target = target.cuda()

        # compute output
        with torch.no_grad():
            output = model(image)
            loss = criterion(output, target)

            batch_size = image.shape[0]
            pred = output.data.argmax(axis=1)
            classwise_acc = torch.eq(pred, target).float().mean(axis=0)

            classwise_loss = loss.mean(axis=0)

            losses.update(classwise_loss.cpu().numpy(), batch_size)
            top1.update(classwise_acc.cpu().numpy(), batch_size)

    return top1.avg, losses.avg


def train_with_rewind(train_params, loaders, args):
    train_loader = loaders["train"]
    test_loader = loaders["test"]

    scheduler = train_params['scheduler']
    writer = train_params['writer']
    criterion = train_params["criterion"]
    model = train_params["model"]
    name = train_params['name']
    start_epoch = train_params["start_epoch"]

    epochs = args.epochs
    rewind = args.rewind_epoch if hasattr(args, "rewind_epoch") else None

    save_dir = os.path.join(args.save_dir, name)
    rewind_path = os.path.join(save_dir, 'rewind_weight.pt')


    os.makedirs(save_dir, exist_ok=True)

    rewind_state_dict = None
    for epoch in range(start_epoch, epochs):
        print(f"Epoch: {epoch}")
        if epoch == rewind:
            save_checkpoint(train_params, rewind_path, model_only=True)
            rewind_state_dict = copy.deepcopy(model.state_dict())

        start_time = time.time()
        train_acc, train_loss = train_epoch(train_params, train_loader, epoch)
        print("Train: Accuracy: {} Loss: {}".format(train_acc, train_loss))
        test_acc, test_loss = validate(model, test_loader, criterion)
        print("Test: Accuracy: {} Loss: {}".format(test_acc, test_loss))

        if writer is not None:
            utils.plot_tensorboard(
                writer, f'{name}_test/acc', test_acc, epoch)
            utils.plot_tensorboard(
                writer, f'{name}_test/loss', test_loss, epoch)

        scheduler.step()
        print("one epoch duration:{}".format(time.time()-start_time))

        if (epoch+1) % args.save_freq == 0:
            save_checkpoint(train_params, os.path.join(
                save_dir, f"checkpoint_{epoch+1}.pt"))
    return rewind_state_dict
