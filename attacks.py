import tqdm
import os

import torch

import torchattack.torchattacks as atk


def generate_attack_images(model, loader, atk, save_dir=None):
    path_x_adv = os.path.join(save_dir, "x_adv.pt")
    path_delta = os.path.join(save_dir, "delta.pt")
    path_label = os.path.join(save_dir, "attr_labels.pt")
    # if os.path.exists(path_x_adv) and os.path.exists(path_delta) and os.path.exists(path_label):
    #     return torch.load(x_advs, path_x_adv), torch.load(deltas, path_delta), torch.load(x_advs, path_label)

    x_advs, deltas, targets = [], [], []

    n_datas, n_correct_success, n_success = 0, 0, 0

    model.eval()
    for image, target in tqdm.tqdm(loader['test']):
        image = image.float()
        target = target.long()

        image_adv = atk(image, target)
        ori_out = model(image)
        adv_out = model(image_adv)  # Test-time augmentation

        idx = ori_out.argmax(1).eq(target) * adv_out.argmax(1).ne(target)
        idx_adv = adv_out.argmax(1).ne(target)

        n_datas += len(idx)
        n_correct_success += sum(idx).item()
        n_success += sum(idx_adv).item()

        adv_delta = (image_adv-image)[idx]
        target = target[idx]

        x_advs.append(image_adv.cpu())
        deltas.append(adv_delta.cpu())
        targets.append(target.cpu())

    x_advs = torch.cat(x_advs, axis=0)
    deltas = torch.cat(deltas, axis=0)
    targets = torch.cat(targets, axis=0)

    os.makedirs(save_dir, exist_ok=True)

    if save_dir is not None:
        torch.save(x_advs, path_x_adv)
        torch.save(deltas, path_delta)
        torch.save(targets, path_label)
    
        with open('attack_acc.log', 'w') as fout:
            print("n_corr_succ: {}".format(n_correct_success / n_datas * 100), file=fout)
            print("n_succ: {}".format(n_success / n_datas * 100), file=fout)
    
    print("n_corr_succ: {}".format(n_correct_success / n_datas * 100))
    print("n_succ: {}".format(n_success / n_datas * 100))
    return x_advs , deltas, targets
    

def get_attack(model, name, args):
    if name == "pgd":
        print("PGD attack: eps: {}, alpha: {}, steps: {}".format(
            args.eps, args.alpha, args.steps))
        return atk.PGD(model, eps=args.eps/255, alpha=args.alpha/255, steps=args.steps)
    elif name == "cw":
        return atk.CW(model, c=args.cw_c, kappa=args.cw_kappa)
    elif name == "pgdl2":
        return atk.PGDL2(model, eps=args.eps, alpha=args.alpha, steps=args.steps)
    elif name == "autoattack":
        if args.norm == "Linf":
            args.eps /= 255
        return atk.AutoAttack(
            model, norm=args.norm, eps=args.eps)  # AutoAttack
    elif name == "fgsm":
        return atk.FGSM(model, eps=args.eps/255)  # FGSM (l_inf)
    elif name == "square":
        if args.norm == "Linf":
            args.eps /= 255
        return atk.Square(model, norm=args.norm, eps=args.eps, n_queries=args.n_queries)
    else:
        raise NotImplementedError(
            "Attack method {} is not implemented!".format(name))


def get_attack_normalized(model, name, args):
    CIFAR_MEAN_1 = [125.307/255, 122.961/255, 113.8575/255]
    CIFAR_STD_1 = [51.5865/255, 50.847/255, 51.255/255]

    atk = get_attack(model, name, args)
    atk.set_normalization_used(mean=CIFAR_MEAN_1, std=CIFAR_STD_1)

    return atk
