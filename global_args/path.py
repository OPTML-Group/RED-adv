import os

# feel free to editing paths
_WORKSPACE2_DIR = "/localscratch2/ljcc"
_WORKSPACE_DIR = "/localscratch/ljcc"

__all__ = ["ATK_DIR", "MODEL_DIR", "PARSING_DIR", "PARSING_LOG_DIR", "GREP_DIR"]

ATK_DIR = os.path.join(_WORKSPACE2_DIR, "attack_img")
MODEL_DIR = os.path.join(_WORKSPACE2_DIR, "victim_models")
PARSING_DIR = os.path.join(_WORKSPACE2_DIR, "parsing_models")
PARSING_LOG_DIR = os.path.join(_WORKSPACE2_DIR, "test_log")

GREP_DIR = os.path.join(_WORKSPACE_DIR, "grep_datas")
