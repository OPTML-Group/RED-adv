import utils
import time
import torch


def get_optimizer_and_scheduler(model, args):
    decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    return optimizer, scheduler


def train_epoch(train_loader, model, criterion, optimizer, epoch, args):

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    for i, (image, target) in enumerate(train_loader):

        image = image.cuda()
        target = target.cuda()

        # compute output
        output = model(image)
        loss = criterion(output, target)
        train_loss = loss.mean()
        optimizer.zero_grad()
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

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print('Epoch: [{0}][{1}/{2}]\n'
                      'Loss\t\t{loss_val} ({loss_avg})\n'
                      'Accuracy\t{top1_val} ({top1_avg})\n'
                      'Time {3:.2f}'.format(
                          epoch, i, len(train_loader), end-start,
                          loss_val=",".join(["{:.3f}".format(x)
                                            for x in losses.val]),
                          loss_avg=",".join(["{:.3f}".format(x)
                                            for x in losses.avg]),
                          top1_val=",".join(["{:.3f}".format(x)
                                            for x in top1.val]),
                          top1_avg=",".join(["{:.3f}".format(x) for x in top1.avg])))
                start = time.time()

    print('train_accuracy {top1_avg}'.format(
        top1_avg=",".join(["{:.3f}".format(x) for x in top1.avg])))

    return top1.avg


def validate(val_loader, model, criterion, args):

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

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print('Test: [{0}/{1}]\n'
                      'Loss\t\t{loss_val} ({loss_avg})\n'
                      'Accuracy\t{top1_val} ({top1_avg})\n'.format(
                          i, len(val_loader),
                          loss_val=",".join(["{:.3f}".format(x)
                                            for x in losses.val]),
                          loss_avg=",".join(["{:.3f}".format(x)
                                            for x in losses.avg]),
                          top1_val=",".join(["{:.3f}".format(x)
                                            for x in top1.val]),
                          top1_avg=",".join(["{:.3f}".format(x) for x in top1.avg])))
                start = time.time()

    print('valid_accuracy {top1_avg}'.format(
        top1_avg=",".join(["{:.3f}".format(x) for x in top1.avg])))

    return top1.avg


def train(model, optimizer, scheduler, train_loader, criterion, args):
    for epoch in range(args.epochs):
        start_time = time.time()
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        train_epoch(train_loader, model,
                    criterion, optimizer, epoch, args)

        scheduler.step()
        print("one epoch duration:{}".format(time.time()-start_time))
