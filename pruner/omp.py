import trainer
from . import utils


def prune_model(model, args):
    ratio = args.pruning_ratio

    if args.structured_pruning:
        print('Structured pruning')
        utils.pruning_model_structured(model, ratio)
    else:
        print('L1 pruning')
        utils.pruning_model(model, ratio)

    utils.check_sparsity(model)


def prune_and_rewind(train_params, path, args):
    model = train_params["model"]

    prune_model(model, args)

    current_mask = utils.extract_mask(model.state_dict())
    utils.remove_prune(model)
    trainer.load_checkpoint(train_params, path)
    utils.prune_model_custom(model, current_mask)


def omp(model, loaders, args, prefix=""):
    # ================================training================================
    train_params = trainer.get_training_params(model, f"{prefix}omp_1", args)
    rewind_path = trainer.train_with_rewind(train_params, loaders, args)

    # ================================pruning================================
    train_params = trainer.get_training_params(model, f"{prefix}omp_2", args)

    if train_params["start_epoch"] <= args.rewind_epoch:
        prune_and_rewind(train_params, rewind_path, args)

    # ================================retraining================================
    utils.check_sparsity(model)
    trainer.train_with_rewind(train_params, loaders, args)
