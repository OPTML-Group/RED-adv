from .omp import omp
from .utils import *


def get_prune_method(name):
    """method usage:

    function(model, train_loader, test_loader, criterion, args)"""
    if name == "omp":
        return omp
    else:
        raise NotImplementedError(f"Pruning method {name} not implemented!")
