import os

import arg_parser
import attacks
import datasets
import models
import pruner
import training_utils
import utils


def main():
    args = arg_parser.parse_args_victim_training()
    utils.set_seed(args.seed)

    if args.dataset == "cifar10":
        loader = datasets.CIFAR10(
            dir=args.dataset_dir, ffcv_dir=args.ffcv_dir, batch_size=args.batch_size
        )
    elif args.dataset == "cifar100":
        loader = datasets.CIFAR100(
            dir=args.dataset_dir, ffcv_dir=args.ffcv_dir, batch_size=args.batch_size
        )
    elif args.dataset == "tinyimagenet":
        loader = datasets.TinyImageNet(
            dir=args.dataset_dir, ffcv_dir=args.ffcv_dir, batch_size=args.batch_size
        )
    elif args.dataset == "mnist":
        loader = datasets.MNIST(dir=args.dataset_dir, batch_size=args.batch_size)
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented!")
    model = models.get_model(args.arch, args)
    model = model.cuda()

    if args.dataset != "mnist":
        prefix = (
            training_utils.get_model_name(
                seed=args.seed,
                kernel_size=args.kernel_size,
                activation_function=args.act_func,
                pruning_ratio=args.pruning_ratio,
                struct=args.structured_pruning,
                robust=args.robust_train,
            )
            + "_"
        )
    else:
        prefix = "seed{}_conv{}_fc{}_kernel{}_act{}_prune{}_".format(
            args.seed,
            args.num_conv,
            args.num_fc,
            args.kernel_size,
            args.act_func,
            args.pruning_ratio,
        )
        if args.structured_pruning:
            prefix += "struct_"
        if args.robust_train:
            prefix += "robust_"

    pruner.omp(model, loader, args, prefix=prefix)

    if args.attack is not None:
        atk = attacks.get_attack_normalized(model, args.attack, args)
        prefix = prefix[:-1]
        attack_dir = os.path.join(args.attack_save_dir, prefix)
        attacks.generate_attack_images(model, loader, atk, attack_dir)


if __name__ == "__main__":
    main()
