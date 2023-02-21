import os

# feel free to editing paths
_WORKSPACE_DIR = "./results"

FFCV_FORMAT = "/localscratch/tmp/ffcv{i}"
DATASET_DIRS = {
    "tinyimagenet": "/localscratch/tmp/tiny-imagenet-200",
    "cifar10": "/localscratch/tmp/cifar10",
    "cifar100": "/localscratch/tmp/cifar100",
}


__all__ = ["ATK_DIR", "MODEL_DIR", "PARSING_DIR",
           "PARSING_LOG_DIR", "GREP_DIR", "FFCV_FORMAT", "DATASET_DIRS"]

ATK_DIR = os.path.join(_WORKSPACE_DIR, "attack_img")
GREP_DIR = os.path.join(_WORKSPACE_DIR, "grep_datas")
MODEL_DIR = os.path.join(_WORKSPACE_DIR, "victim_models")
PARSING_DIR = os.path.join(_WORKSPACE_DIR, "parsing_models")
PARSING_LOG_DIR = os.path.join(_WORKSPACE_DIR, "test_log")
