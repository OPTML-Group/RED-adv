import tqdm
import os
import shutil

import torch

import torchattack.torchattacks as atk
import impl_atk


def generate_attack_images(model, loader, atk, save_dir=None):
    # path_x_adv = os.path.join(save_dir, "x_adv.pt")
    # path_delta = os.path.join(save_dir, "delta.pt")
    path_delta_all = os.path.join(save_dir, "delta_all.pt")
    path_adv_all = os.path.join(save_dir, "adv_all.pt")
    
    path_ori_pred = os.path.join(save_dir, "ori_pred.pt")
    path_adv_pred = os.path.join(save_dir, "adv_pred.pt")
    path_target = os.path.join(save_dir, "targets.pt")

    path_acc = os.path.join(save_dir, 'attack_acc.log')
    # if os.path.exists(path_x_adv) and os.path.exists(path_delta) and os.path.exists(path_target) and os.path.exists(path_acc):
    #     return torch.load(path_x_adv), torch.load(path_delta), torch.load(path_target)
    if os.path.exists(path_target) and os.path.exists(path_delta_all) and os.path.exists(path_adv_all) and os.path.exists(path_adv_pred) and os.path.exists(path_ori_pred) and os.path.exists(path_acc):
        return 
    shutil.rmtree(save_dir, ignore_errors=True)

    # x_advs, deltas,  = [], []
    targets, adv_all, adv_preds, delta_all, ori_preds = [], [], [], [], []

    n_datas, n_correct_success, n_success = 0, 0, 0

    model.eval()
    for image, target in tqdm.tqdm(loader['test']):
        image = image.float().cuda()
        target = target.long().cuda()

        image_adv = atk(image, target).detach()

        with torch.no_grad():
            ori_out = model(image).detach()
            adv_out = model(image_adv).detach()  # Test-time augmentation

            adv_pred = adv_out.argmax(1)
            ori_pred = ori_out.argmax(1)

            idx = ori_pred.eq(target) * adv_pred.ne(target)
            idx_adv = adv_pred.ne(target)

            n_datas += len(idx)
            n_correct_success += sum(idx).item()
            n_success += sum(idx_adv).item()

            delta = image_adv-image

            # adv_delta = delta[idx]
            # x_adv = image_adv[idx]

            # x_advs.append(x_adv.cpu())
            # deltas.append(adv_delta.cpu())

            adv_all.append(image_adv.detach().cpu())
            delta_all.append(delta.detach().cpu())

            ori_preds.append(ori_pred.cpu())
            adv_preds.append(adv_pred.cpu())
            targets.append(target.detach().cpu())

    # x_advs = torch.cat(x_advs, axis=0)
    # deltas = torch.cat(deltas, axis=0)

    adv_all = torch.cat(adv_all, axis=0)
    delta_all = torch.cat(delta_all, axis=0)

    targets = torch.cat(targets, axis=0)
    adv_preds = torch.cat(adv_preds, axis=0)
    ori_preds = torch.cat(ori_preds, axis=0)

    os.makedirs(save_dir, exist_ok=True)

    if save_dir is not None:
        # torch.save(x_advs, path_x_adv)
        # torch.save(deltas, path_delta)
        torch.save(adv_all, path_adv_all)
        torch.save(delta_all, path_delta_all)

        torch.save(targets, path_target)
        torch.save(adv_preds, path_adv_pred)
        torch.save(ori_preds, path_ori_pred)
    
        with open(path_acc, 'w') as fout:
            print("n_corr_succ: {}".format(n_correct_success / n_datas * 100), file=fout)
            print("n_succ: {}".format(n_success / n_datas * 100), file=fout)
    
    print("n_corr_succ: {}".format(n_correct_success / n_datas * 100))
    print("n_succ: {}".format(n_success / n_datas * 100))
    # return x_advs, deltas, targets
    

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
    elif name == "zosignsgd":
        if args.norm == "Linf":
            args.eps /= 255
        return impl_atk.ZoSignSgd(model, eps=args.eps, norm=args.norm)
    elif name == "zosgd":
        if args.norm == "Linf":
            args.eps /= 255
        return impl_atk.ZoSgd(model, eps=args.eps, norm=args.norm)
    elif name == "nes":
        if args.norm == "Linf":
            args.eps /= 255
        return impl_atk.Nes(model, eps=args.eps, norm=args.norm)
    else:
        raise NotImplementedError(
            "Attack method {} is not implemented!".format(name))


def get_attack_normalized(model, name, args):
    CIFAR_MEAN_1 = [125.307/255, 122.961/255, 113.8575/255]
    CIFAR_STD_1 = [51.5865/255, 50.847/255, 51.255/255]
    TINYIMAGENET_MEAN_1 = [x * 255 for x in [0.4802, 0.4481, 0.3975]]
    TinyImageNet_STD_1 = [x * 255 for x in [0.2302, 0.2265, 0.2262]]

    atk = get_attack(model, name, args)
    if args.dataset in ["cifar10", "cifar100"]:
        atk.set_normalization_used(mean=CIFAR_MEAN_1, std=CIFAR_STD_1)
    elif args.dataset == 'tinyimagenet':
        atk.set_normalization_used(mean=TINYIMAGENET_MEAN_1, std=TinyImageNet_STD_1)
    else:
        pass

    return atk
