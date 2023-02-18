### ZOSignSGD
```
CUDA_VISIBLE_DEVICES=0 python main_victim.py --kernel-size 3 --act-func relu --pruning-ratio 0.0 --tensorboard --save-dir /localscratch2/ljcc/results --attack zosignsgd --eps 8 --norm Linf --attack-save-dir /localscratch2/ljcc/attack_img/attack_zosignsgd_eps_8_norm_Linf --dataset-dir /localscratch2/tmp/cifar0
```