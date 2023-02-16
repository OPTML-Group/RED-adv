import arg_parser
import utils
import datasets
import models
import pruner
import attacks
import os


def main():
    args = arg_parser.parse_args_victim_training()
    utils.set_seed(args.seed)

    loader = datasets.CIFAR10(dir=args.dataset_dir, batch_size=args.batch_size)
    model = models.ResNet9(num_classes=args.num_classes,
                           kernel_size=args.kernel_size, act_func=args.act_func)
    model = model.cuda()

    prefix = "seed{}_kernel{}_act{}_prune{}_".format(
        args.seed, args.kernel_size, args.act_func, args.pruning_ratio)
    if args.structured_pruning:
        prefix += "struct_"
    if args.robust_train:
        prefix += "robust_"

    pruner.omp(model, loader, args, prefix=prefix)

    if args.attack is not None:
        atk = attacks.get_attack_normalized(model, args.attack, args)
        prefix = prefix[:-1]
        attack_dir = os.path.join(args.attack_save_dir, prefix)
        attacks.generate_attack_images(
            model, loader, atk, attack_dir)


if __name__ == "__main__":
    main()
