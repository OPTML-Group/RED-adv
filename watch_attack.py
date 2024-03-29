import os

import global_args as gargs
import run
import training_utils


def get_models(dataset, arch, robust=True, omp=2):
    _data_arch_name = f"{dataset}_{arch}"
    _model_dir = os.path.join(gargs.MODEL_DIR, _data_arch_name)
    last_epoch = 100 if dataset == "tinyimagenet" else 75

    cnt = 0
    for k in run._kernels:
        for a in run._acts:
            for r in run._ratios:
                for s in run._struct:
                    if r == 0.0 and s:
                        continue
                    model_name = training_utils.get_model_name(2, k, a, r, s, robust)

                    path = os.path.join(
                        _model_dir, f"{model_name}_omp_{omp}/checkpoint_{last_epoch}.pt"
                    )
                    # print(path)

                    if not os.path.exists(path):
                        cnt += 1
    return cnt


if __name__ == "__main__":
    datasets = ["cifar10", "cifar100", "tinyimagenet"]
    archs = ["resnet9", "resnet20s", "resnet18", "vgg11", "vgg13"]

    dataset = datasets[0]
    for arch in archs:
        print(f"arch: {arch}", end="\t")
        n = get_models(dataset, arch, robust=False, omp=1)
        print(f"omp1: {n: 4d}", end="\t")
        n = get_models(dataset, arch, robust=False, omp=2)
        print("omp2: ", n)

    arch = archs[0]
    for dataset, arch in zip(datasets[1:], ["resnet9", "resnet18"]):
        print(f"data: {dataset}", end="\t")
        n = get_models(dataset, arch, robust=False, omp=1)
        print(f"omp1: {n: 4d}", end="\t")
        n = get_models(dataset, arch, robust=False, omp=2)
        print("omp2: ", n)

    # victim training
    commands = []
    for exp in gargs.EXPS:
        print(exp["data"], exp["arch"], exp["setting"], end=" ")

        if not (exp["setting"] not in "origin robust"):
            robust = "robust" in exp["setting"]
            n = get_models(exp["data"], exp["arch"], robust=robust, omp=1)
            print(n, end=" ")
            robust = "robust" in exp["setting"]
            n = get_models(exp["data"], exp["arch"], robust=robust, omp=2)
            print(n, end=" ")
            cmds = run.gen_commands_attack_victim(
                dataset=exp["data"],
                arch=exp["arch"],
                attacks=exp["attacks"],
                robust=robust,
            )
            print(len(cmds), end=" ")
        for attr_arch in ["conv4"]:
            cmds = run.gen_commands_parsing(exp, attr_arch=attr_arch)
            cmds2 = run.gen_commands_eval_parsing(exp, attr_arch=attr_arch)
            print(attr_arch, len(cmds), len(cmds2), end=" ")
        print()
    # for exp1 in gargs.EXPS:
    #     for exp2 in gargs.EXPS:
    #         cmds = run.gen_commands_eval_parsing_cross(
    #             exp1, exp2, attr_arch="conv4")
    #         print(len(cmds), end='\t')
    #     print()
    for attr in gargs.VALID_ATTR_ARCHS:
        cmds = run.gen_commands_parsing(gargs.EXPS[0], attr)
        cmds2 = run.gen_commands_eval_parsing(gargs.EXPS[0], attr)
        print(attr, len(cmds), len(cmds2), end=" ")
    print()
    cnt = 0
    for exp1 in gargs.EXPS[:5]:
        for exp2 in gargs.EXPS[:5]:
            cmds = run.gen_commands_eval_parsing_cross(exp1, exp2, "conv4")
            cnt += len(cmds)
    print("conv4", cnt)
    run.train_large_set_parsing_commands(attr_arch="conv4")
    run.test_large_set_parsing_commands(attr_arch="conv4")

    print(f"denoise:")
    commands = run.train_parsing_commands("conv4", "denoise")
    run.train_large_set_parsing_commands(attr_arch="conv4", specific_type="denoise")
    # commands = run.gen_commands_eval_parsing_cross(gargs.EXPS[0], gargs.EXPS[0], "conv4", "denoise")
    commands = run.cross_test_parsing_commands("conv4", "denoise")
    # print(len(commands))
    commands = run.test_large_set_parsing_commands(
        attr_arch="conv4", specific_type="denoise"
    )
