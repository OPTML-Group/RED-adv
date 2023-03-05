import torch

import os

import attr_models
import arg_parser
import main_parser

def main():
    args = arg_parser.parse_args_model_parsing(train=False)
    main_parser.set_seed(args.seed)

    # test data
    test_dl = main_parser.get_data(args, train=False)

    n_class = test_dl.dataset.tensors[-1].max() + 1
    n_output = test_dl.dataset.tensors[-1].shape[1]
    print(f"class num: {n_class}, output num: {n_output}")
    model = main_parser.get_model(args, n_class, n_output)

    assert os.path.exists(os.path.join(args.save_folder, "final.pt"))

    suffix = "best"
    denoiser = None

    if args.input_type == "denoise":
        suffix = "final"
        denoiser = attr_models.DnCNN(
            image_channels=3, depth=17, n_channels=64).cuda()
        denoiser_path = os.path.join(args.save_folder, f"denoiser_{suffix}.pt")
        assert os.path.exists(denoiser_path)
        denoiser.load_state_dict(torch.load(denoiser_path))
        
    model.load_state_dict(torch.load(os.path.join(args.save_folder, f"{suffix}.pt")))

    test_acc = main_parser.validate(model, test_dl, denoiser)

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