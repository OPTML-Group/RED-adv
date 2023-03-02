from IPython import embed
from torch.utils.data import Dataset, DataLoader

import torch
from torch.cuda.amp import autocast 

import os

import attr_models
import global_args as gargs
import arg_parser
import main_parser

def main():
    args = arg_parser.parse_args_model_parsing(train=False)
    main_parser.set_seed(args.seed)

    # test data
    test_dl = main_parser.get_data(args, train=False)

    n_class = test_dl.dataset.tensors[1].max() + 1
    n_output = test_dl.dataset.tensors[1].shape[1]
    model = main_parser.get_model(args, n_class, n_output)
    print(f"class num: {n_class}, output num: {n_output}")

    assert os.path.exists(os.path.join(args.save_folder, "final.pt"))
    model.load_state_dict(torch.load(os.path.join(args.save_folder, "best.pt")), strict=False)

    test_acc = main_parser.validate(model, test_dl)

    name1 = os.path.basename(args.input_folder)
    name2 = os.path.basename(args.save_folder)
    name3 = os.path.basename(os.path.dirname(args.save_folder))
    file_name = f"data_{name1}___model_{name3}__{name2}.log"
    log_path = os.path.join(args.log_dir, file_name)
    os.makedirs(args.log_dir, exist_ok=True)
    fout = open(log_path, 'w')

    for i in test_acc:
        print(i, file = fout)

if __name__ == "__main__":
    main()