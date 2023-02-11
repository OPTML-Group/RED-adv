import arg_parser
import utils
import datasets
import models
import pruner

def main():
    args = arg_parser.parse_args_victim_training()
    utils.set_seed(args.seed)

    loader = datasets.CIFAR10(dir = args.dataset_dir, batch_size = args.batch_size)
    model = models.ResNet9(num_classes = args.num_classes, kernel_size = args.kernel_size, act_func=args.act_func)
    model = model.cuda()

    prefix = "seed{}_kernel{}_act{}_prune{}_".format(args.seed, args.kernel_size, args.act_func, args.pruning_ratio)
    if args.structured_pruning:
        prefix += "struct_"

    pruner.omp(model, loader, args, prefix=prefix)

if __name__ == "__main__":
    main()
