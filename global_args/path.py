import os as _os

# feel free to editing paths
_WORKSPACE_DIR = "./results"

FFCV_FORMAT = "/localscratch/tmp/ffcv{i}"
DATASET_DIRS = {
    "tinyimagenet": "/localscratch/tmp/tiny-imagenet-200",
    "cifar10": "/localscratch/tmp/cifar10",
    "cifar100": "/localscratch/tmp/cifar100",
    "mnist": "/localscratch/tmp/mnist",
}


ATK_DIR = _os.path.join(_WORKSPACE_DIR, "attack_img")
GREP_DIR = _os.path.join(_WORKSPACE_DIR, "grep_datas")
MODEL_DIR = _os.path.join(_WORKSPACE_DIR, "victim_models")
PARSING_DIR = _os.path.join(_WORKSPACE_DIR, "parsing_models")
PARSING_LOG_DIR = _os.path.join(_WORKSPACE_DIR, "test_log")


PARTIAL_RESULT_NAMES = ["x_adv.pt", "delta.pt"]
FULL_RESULT_NAMES = [
    "adv_all.pt",
    "delta_all.pt",
    "adv_pred.pt",
    "ori_pred.pt",
    "targets.pt",
]

NO_SPLIT_OUTPUT_NAMES = ["x_adv.pt", "delta.pt", "attr_labels.pt"]
SPLIT_OUTPUT_NAMES = [
    "x_adv_train.pt",
    "delta_train.pt",
    "attr_labels_train.pt",
    "x_adv_test.pt",
    "delta_test.pt",
    "attr_labels_test.pt",
]
