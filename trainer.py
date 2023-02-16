import os
import copy
import time
import tqdm
import utils
import shutil

import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils import tensorboard

import pruner
import attacks


def load_checkpoint(train_params, path, model_only=False):
    print("Loading checkpoint from {}".format(path))
    load_dict = torch.load(path)
    if model_only:
        train_params["model"].load_state_dict(item)
    else:
        for key, item in load_dict.items():
            if key == "start_epoch":
                train_params[key] = item
            elif key == "model":
                current_mask = pruner.extract_mask(item)
                if len(current_mask) > 0:
                    pruner.prune_model_custom(train_params[key], current_mask)
                train_params[key].load_state_dict(item, strict=False)
            else:
                train_params[key].load_state_dict(item)


def save_checkpoint(train_params, path, epoch, model_only=False):
    if model_only:
        save_dict = train_params["model"].state_dict()
    else:
        save_dict = {"start_epoch": epoch + 1}
        for key, item in train_params.items():
            if hasattr(item, "state_dict"):
                save_dict[key] = item.state_dict()
    torch.save(save_dict, path)


def get_training_params(model, name, args, use_scaler=True):
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    scaler = None
    if use_scaler:
        scaler = GradScaler()
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    writer = None

    save_dir = os.path.join(args.save_dir, name)
    if args.rerun:
        shutil.rmtree(save_dir, ignore_errors=True)
    os.makedirs(save_dir, exist_ok=True)

    if args.tensorboard:
        log_dir = os.path.join(save_dir, "tensorboard")
        writer = tensorboard.SummaryWriter(log_dir=log_dir)

    attack = None
    if args.robust_train:
        attack = attacks.atk.PGD(model, eps=8/255, alpha=1/255, steps=10)

        CIFAR_MEAN_1 = [125.307/255, 122.961/255, 113.8575/255]
        CIFAR_STD_1 = [51.5865/255, 50.847/255, 51.255/255]

        attack.set_normalization_used(mean=CIFAR_MEAN_1, std=CIFAR_STD_1)

    train_params = {
        "optimizer": optimizer,
        "scheduler": scheduler,
        "criterion": criterion,
        "scaler": scaler,
        "writer": writer,
        "model": model,
        "name": name,
        "attack": attack,
        "start_epoch": 0,
    }

    if not args.rerun:
        for epoch in range(args.epochs, 0, -1):
            path = os.path.join(save_dir, f"checkpoint_{epoch}.pt")
            if os.path.exists(path):
                load_checkpoint(train_params, path)
                if train_params.get("start_epoch") is None:
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
    atk = train_params['attack']

    # switch to train mode
    model.train()
    n_iters = len(train_loader)
    lr = optimizer.state_dict()['param_groups'][0]['lr']

    for image, target in tqdm.tqdm(train_loader):
        image = image.cuda()
        target = target.cuda()

        # compute output
        with autocast():
            if atk is not None:
                image = atk(image, target)
                model.train()
            output = model(image)
            loss = criterion(output, target)
            train_loss = loss.mean()

        optimizer.zero_grad()

        if scaler is not None:
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


def validate(model, val_loader, criterion, atk=None):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # switch to evaluate mode
    model.eval()

    for image, target in val_loader:
        image = image.cuda()
        target = target.cuda()

        with autocast():
            if atk is not None:
                image = atk(image, target)
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
    atk = train_params['attack']

    save_dir = os.path.join(args.save_dir, name)

    epochs = args.epochs
    rewind = args.rewind_epoch if hasattr(args, "rewind_epoch") else None
    rewind_path = os.path.join(
        save_dir, f'rewind_weight_{rewind}.pt') if rewind is not None else None

    for epoch in range(start_epoch, epochs):
        print(f"Epoch: {epoch}")
        if epoch == rewind:
            save_checkpoint(train_params, rewind_path, epoch)

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
            
        if atk is not None:
            attack_acc, attack_loss = validate(model, test_loader, criterion, atk=atk)
            print("Attack: Accuracy: {} Loss: {}".format(attack_acc, attack_loss))

            if writer is not None:
                utils.plot_tensorboard(
                    writer, f'{name}_test/attack_acc', attack_acc, epoch)
                utils.plot_tensorboard(
                    writer, f'{name}_test/attack_loss', attack_loss, epoch)

        scheduler.step()
        print("one epoch duration:{}".format(time.time()-start_time))

        if (epoch+1) % args.save_freq == 0:
            path = os.path.join(save_dir, f"checkpoint_{epoch+1}.pt")
            save_checkpoint(train_params, path, epoch)

    train_acc, train_loss = validate(model, train_loader, criterion)
    print("Final train: Accuracy: {} Loss: {}".format(train_acc, train_loss))
    test_acc, test_loss = validate(model, test_loader, criterion)
    print("Final test: Accuracy: {} Loss: {}".format(test_acc, test_loss))

    return rewind_path
