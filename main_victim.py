import arg_parser
import os
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import time

import torch
from torch.optim import SGD, lr_scheduler
from torch.cuda.amp import GradScaler, autocast 
from torch.nn import CrossEntropyLoss

from utils import set_seed
from datasets import CIFAR10
from models import ResNet9
from pruner.pruner import remove_prune, check_sparsity, prune_model_custom, pruning_model_structured, pruning_model, extract_mask

def main():
    args = arg_parser.parse_args_victim_training()
    set_seed(args.seed)
    loader = CIFAR10(dir = args.dir, batch_size = args.batch_size)
    model = ResNet9(num_classes = args.num_classes, kernel_size = args.kernel_size)
    model = model.to(memory_format=torch.channels_last).cuda()

    opt = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    iters_per_epoch = len(loader["train"])
    lr_schedule = np.interp(np.arange((args.epochs + 1) * iters_per_epoch),\
         [0, 5 * iters_per_epoch, args.epochs * iters_per_epoch], [0, 1, 0])
    scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
    scaler = GradScaler()
    loss_fn = CrossEntropyLoss(label_smoothing=0.1)

    os.makedirs("model_pt", exist_ok = True)
    os.makedirs("model_pt/prune_init/", exist_ok = True)

    for ep in range(args.epochs):
        if ep == args.rewind_epoch:
            init = deepcopy(model.state_dict())
            torch.save(model.state_dict(), "model_pt/prune_init/resnet9_seed{}_ks{}_act{}_prune{}.pt".format(args.seed, args.kernel_size, args.act_func, 0.0))

        for ims, labs in tqdm(loader['train']):
            opt.zero_grad(set_to_none=True)
            with autocast():
                out = model(ims)
                loss = loss_fn(out, labs)

            scaler.scale(loss).backward()  # type: ignore
            scaler.step(opt)
            scaler.update()
            scheduler.step()

    if args.pruning_ratio > 0:
        init = torch.load("model_pt/prune_init/resnet9_seed{}_ks{}_act{}_prune{}.pt".format(args.seed, args.kernel_size, args.act_func, 0.0))

        if args.structured_pruning:
            pruning_model_structured(model, args.pruning_ratio)
        else:
            pruning_model(model, args.pruning_ratio) # when using unstructured, change view to reshape
        current_mask = extract_mask(model.state_dict())
        remove_prune(model)
        
        model.load_state_dict(init)
        prune_model_custom(model, current_mask)
        opt = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
        for _ in range(args.rewind_epoch):
                scheduler.step()

        for ep in range(args.epochs):
            for ims, labs in tqdm(loader['train']):
                opt.zero_grad(set_to_none=True)
                with autocast():
                    out = model(ims) 
                    loss = loss_fn(out, labs)

                scaler.scale(loss).backward()  # type: ignore
                scaler.step(opt)
                scaler.update()
                scheduler.step()
        remove_prune(model)
        check_sparsity(model)

    test(model, loader)
        
    if args.pruning_ratio > 0:
        torch.save(model.state_dict(), "model_pt/resnet9_seed{}_ks{}_act{}_prune{}_struct{}.pt".format(args.seed, args.kernel_size, args.act_func, args.pruning_ratio, args.structured_pruning))
    else:
        torch.save(model.state_dict(), "model_pt/resnet9_seed{}_ks{}_act{}_prune{}.pt".format(args.seed, args.kernel_size, args.act_func, args.pruning_ratio))

def test(model, loader):
    model.eval()
    with torch.no_grad():
        total_correct, total_num = 0., 0.
        for ims, labs in tqdm(loader['test']):
            with autocast():
                out = model(ims)  # Test-time augmentation
                total_correct += out.argmax(1).eq(labs).sum().cpu().item()
                total_num += ims.shape[0]
        print(f'accuracy: {total_correct / total_num * 100:.2f}%')

if __name__ == "__main__":
    main()
