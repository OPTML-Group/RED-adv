import torchattack.torchattacks as atk


def get_attack(model, name, args):
    if name == "pgd":
        return atk.PGD(model, eps=args.eps, alpha=args.alpha, steps=args.steps)
    elif name == "cw":
        return atk.CW(model)
    elif name == "pgdl2":
        return atk.PGDL2(model, eps=args.eps, alpha=args.alpha, steps=args.steps)
    elif name == "autoattack":
        return atk.AutoAttack(
            model, norm=args.norm, eps=args.eps)  # AutoAttack
    elif name == "fgsm":
        return atk.FGSM(model, eps=args.eps)  # FGSM (l_inf)
    elif name == "square":
        return atk.Square(model, norm='Linf', eps=args.eps, n_queries=args.n_queries)
    else:
        raise NotImplementedError(
            "Attack method {} is not implemented!".format(name))
