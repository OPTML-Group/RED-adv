import os

# feel free to editing paths
_WORKSPACE2_DIR = "/localscratch2/ljcc"
_WORKSPACE_DIR = "/localscratch/ljcc"
_WORKSPACE3_DIR = "./results"

FFCV_FORMAT = "/localscratch/tmp/ffcv{i}"
DATASET_DIRS = {
    "tinyimagenet": "/localscratch/tmp/tiny-imagenet-200",
    "cifar10": "/localscratch/tmp/cifar10"
}


__all__ = ["ATK_DIR", "MODEL_DIR", "PARSING_DIR",
           "PARSING_LOG_DIR", "GREP_DIR", "FFCV_FORMAT", "DATASET_DIRS"]

ATK_DIR = os.path.join(_WORKSPACE3_DIR, "attack_img")
MODEL_DIR = os.path.join(_WORKSPACE3_DIR, "victim_models")
PARSING_DIR = os.path.join(_WORKSPACE3_DIR, "parsing_models")
PARSING_LOG_DIR = os.path.join(_WORKSPACE2_DIR, "test_log")

GREP_DIR = os.path.join(_WORKSPACE_DIR, "grep_datas")
