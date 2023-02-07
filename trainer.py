
import os
import copy
import time
import tqdm
import utils

import torch

def get_training_tools(model, args):
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    if args.tensorboard:
        log_dir = os.path.join(args.save_dir, "tensorboard")
        writer = torch.utils.tensorboard.SummaryWriter(log_dir = log_dir)
    else:
        writer = None
    return {
        "optimizer": optimizer, 
        "scheduler": scheduler,
        "criterion": criterion,
        "scaler": scaler,
        "writer": writer,
    }


def train_epoch(model, train_loader, train_tools, epoch):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    criterion = train_tools['criterion']
    optimizer = train_tools['optimizer']
    scaler = train_tools['scaler']
    writer = train_tools['writer']

    # switch to train mode
    model.train()
    n_epoch = len(train_loader)

    for i, (image, target) in enumerate(train_loader):

        image = image.cuda()
        target = target.cuda()

        # compute output
        output = model(image)
        loss = criterion(output, target)
        train_loss = loss.mean()
        optimizer.zero_grad()

        scaler.scale(train_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure accuracy and record loss
        with torch.no_grad():
            batch_size = image.shape[0]
            pred = output.data.argmax(axis=1)
            classwise_acc = torch.eq(pred, target).float().mean(axis=0)

            classwise_loss = loss.mean(axis=0)

            losses.update(classwise_loss.cpu().numpy(), batch_size)
            top1.update(classwise_acc.cpu().numpy(), batch_size)

            if writer is not None:
                writer.add_scalars('Training/iter/acc', top1.val, epoch * (n_epoch) + i)
                writer.add_scalars('Training/iter/loss', losses.val, epoch * (n_epoch) + i)

    writer.add_scalars('Training/epoch/acc', top1.avg, epoch)
    writer.add_scalars('Training/epoch/loss', losses.avg, epoch)

    return top1.avg, losses.avg


def validate(val_loader, model, criterion):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (image, target) in enumerate(val_loader):
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


def train_with_rewind(model, loaders, train_tools, args):
    train_loader = loaders["train"]
    test_loader = loaders["test"]

    scheduler = train_tools['scheduler']
    writer = train_tools['writer']

    rewind_state_dict = None
    for epoch in tqdm.tqdm(range(args.epochs)):
        if epoch == args.rewind_epoch:
            rewind_path = os.path.join(
                args.save_dir, 'epoch_{}_rewind_weight.pt'.format(epoch+1))
            torch.save(model.state_dict(), rewind_path)
            if args.prune_type == 'rewind_lt':
                rewind_state_dict = copy.deepcopy(model.state_dict())

        start_time = time.time()
        print(train_tools.state_dict()['param_groups'][0]['lr'])
        train_epoch(model, train_loader, train_tools, epoch)
        acc, loss = validate(test_loader, model, train_tools["criterion"])

        writer.add_scalars('Training/epoch/acc', acc, epoch)
        writer.add_scalars('Training/epoch/loss', loss, epoch)

        scheduler.step()
        print("one epoch duration:{}".format(time.time()-start_time))

    return rewind_state_dict
