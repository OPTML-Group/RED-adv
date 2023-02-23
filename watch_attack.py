import run
import os
import global_args as gargs
# commands = run.gen_commands_victim()
# print(len(commands))
# commands = run.gen_commands_old()
# print(len(commands))


def get_models(dataset, arch, robust=True, omp=2):
    _data_arch_name = f"{dataset}_{arch}"
    _model_dir = os.path.join(gargs.MODEL_DIR, _data_arch_name)
    last_epoch = 100 if dataset == "tinyimagenet" else 75

    cnt = 0
    for k in run.kernels:
        for a in run.acts:
            for r in run.ratios:
                for s in run.struct:
                    if r == 0.0 and s:
                        continue
                    model_name = "seed{}_kernel{}_act{}_prune{}".format(
                        2, k, a, r)
                    if s:
                        model_name += "_struct"
                    if robust:
                        model_name += '_robust'
                    path = os.path.join(_model_dir, f"{model_name}_omp_{omp}/checkpoint_{last_epoch}.pt")
                    # print(path)

                    if not os.path.exists(path):
                        cnt += 1
    return cnt
# print(f"omp_1: {get_models(True,1)}")
# print(f"omp_2: {get_models(True,2)}")
# commands = run.gen_commands_victim(False)
datasets = ["cifar10", "cifar100", "tinyimagenet"]
archs = ['resnet9', 'resnet20s', 'resnet18', 'vgg11', 'vgg13']

dataset = datasets[0]
for arch in archs:
    print(f"arch: {arch}", end='\t')
    n = get_models(dataset, arch, robust=False, omp=1)
    print(f"omp1: {n: 4d}", end='\t')
    n = get_models(dataset, arch, robust=False, omp=2)
    print("omp2: ", n)
    # commands = run.gen_commands_victim(dataset=dataset, arch=arch, attacks=gargs.WHITEBOX_ATTACKS, robust=False)
    # print("attack: ", len(commands), end='\t')
    # commands = run.gen_commands_old(dataset=dataset, arch=arch, setting="origin", attacks=gargs.WHITEBOX_ATTACKS)
    # print("parse: ", len(commands))

arch = archs[0]
for dataset, arch in zip(datasets[1:], ['resnet9', 'resnet18']):
    print(f"data: {dataset}", end='\t')
    n = get_models(dataset, arch, robust=False, omp=1)
    print(f"omp1: {n: 4d}", end='\t')
    n = get_models(dataset, arch, robust=False, omp=2)
    print("omp2: ", n)
    # commands = run.gen_commands_victim(dataset=dataset, arch=arch, attacks=gargs.WHITEBOX_ATTACKS, robust=False)
    # print("attack: ", len(commands), end='\t')
    # commands = run.gen_commands_old(dataset=dataset, arch=arch, setting="origin", attacks=gargs.WHITEBOX_ATTACKS)
    # print("parse: ", len(commands))

# victim training
commands = []
for exp in gargs.EXPS:
    print(exp['data'], exp['arch'], exp['setting'], end=' ')
    
    if not(exp['setting'] not in 'origin robust'):
        robust = 'robust' in exp['setting']
        cmds = run.gen_commands_victim(dataset=exp['data'], arch=exp['arch'], attacks=exp['attacks'], robust=robust)
        print(len(cmds), end=' ')
    cmds = run.gen_commands_parsing(exp)
    print(len(cmds), end=' ')
    cmds = run.gen_commands_eval_parsing(exp)
    print(len(cmds))
for exp1 in gargs.EXPS:
    for exp2 in gargs.EXPS:
        cmds = run.gen_commands_eval_parsing_cross(exp1, exp2)
        print(len(cmds), end='\t')
    print()