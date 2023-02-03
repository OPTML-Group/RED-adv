from utils import run_commands


def gen_commands():
    epss = [0.3, 0.6]
    steps = [1, 2, 4, 6, 8, 10]

    commands = []
    for eps in epss:
        for step in steps:
            dataset_dir = f"./RED/PGD{step}_{eps}"
            output_path = f"./result/RED_PGD{step}_{eps}.log"
            command = f"python main.py --dataset-dir {dataset_dir} --input-type x_adv > {output_path}"
            commands.append(command)
    return commands


if __name__ == "__main__":
    commands = gen_commands()
    run_commands([0, 1] * 2, commands, call=True,
                 dir="commands", shuffle=False, delay=0.5)
