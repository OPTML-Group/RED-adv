import trainer
from . import utils


def omp(model, train_loader, test_loader, criterion, scaler, args):
    # ================================training================================
    train_tools = trainer.get_training_tools(model, args)

    utils.check_sparsity(model)
    rewind_state_dict = trainer.train_with_rewind(
        model, optimizer, scheduler, train_loader, criterion, args)

    # report result
    utils.check_sparsity(model)
    print("Performance on the test data set")
    trainer.validate(test_loader, model, criterion, args)

    # ================================pruning================================
    if args.structured_pruning:
        print('Structured pruning')
        utils.pruning_model_structured(model, args.pruning_ratio)
    else:
        print('L1 pruning')
        utils.pruning_model(model, args.pruning_ratio)

    utils.check_sparsity(model)
    current_mask = utils.extract_mask(model.state_dict())
    utils.remove_prune(model)

    # weight rewinding
    # rewind, rewind_state_dict is a full model architecture without masks
    model.load_state_dict(rewind_state_dict, strict=False)
    utils.prune_model_custom(model, current_mask)

    # ================================retraining================================
    optimizer, scheduler = trainer.get_optimizer_and_scheduler(model, args)
    if args.rewind_epoch:
        # learning rate rewinding
        for _ in range(args.rewind_epoch):
            scheduler.step()

    utils.check_sparsity(model)
    trainer.train_with_rewind(
        model, optimizer, scheduler, train_loader, criterion, args)
