import torchattcks


def get_attack(model, name, args):
    if name == "pgd":
        return torchattacks.PGD(model, eps=args.eps, alpha=args.alpha, steps=args.steps)
    elif name == "cw":
        return torchattacks.CW(model)
    elif name == "pgdl2":
        return torchattacks.PGDL2(model, eps=args.eps, alpha=args.alpha, steps=args.steps)
    elif name == "autoattack":
        return torchattacks.AutoAttack(
            model, norm='Linf', eps=args.steps)  # AutoAttack
    elif name == "fgsm":
        return torchattacks.FGSM(model, eps=args.steps)  # FGSM (l_inf)
    else:
        raise NotImplementedError(
            "Attack method {} is not implemented!".format(name))
