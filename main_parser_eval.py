from IPython import embed
from torch.utils.data import Dataset, DataLoader

import torch
from torch.cuda.amp import autocast 

from tqdm import tqdm
import argparse
import os

import attr_models
import global_args as gargs

parser = argparse.ArgumentParser(description='train clf')

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--lr', type=float, default=1e-1)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--epochs', type=int, default=50)

parser.add_argument('--input_folder', type=str, default="attack")
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--attack', type=str, default='PGD_eps8_alpha1_steps10')
parser.add_argument('--input-type', type=str, default='delta')
parser.add_argument('--save_folder', type=str, default="PGD_eps8_alpha1_steps10")
parser.add_argument('--log_dir', type=str)
parser.add_argument('--attr-arch', type=str, choices=gargs.VALID_ATTR_ARCHS)

args = parser.parse_args()

seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# test data
test_data = torch.load(os.path.join(args.input_folder, f"{args.input_type}_test.pt")).detach().cuda()
test_label = torch.load(os.path.join(args.input_folder, f"attr_labels_test.pt")).detach().cuda()
test_size = test_data.shape[0]
test_set = torch.utils.data.TensorDataset(test_data, test_label)
test_dl = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False)
print(test_data.shape, test_label.shape)


n_class = test_label.max() + 1
n_output = test_label.shape[1]
dataset = args.dataset

print(f"class num: {n_class}, output num: {n_output}")

model = attr_models.get_model(
    args.attr_arch,
    num_channel=gargs.DATASET_NUM_CHANNEL[dataset],
    num_class=n_class,
    num_output=n_output,
    img_size=gargs.DATASET_INPUT_SIZE[dataset]
).cuda()

assert os.path.exists(os.path.join(args.save_folder, "final.pt"))
model.load_state_dict(torch.load(os.path.join(args.save_folder, "best.pt")), strict=False)

model.cuda()
model.eval()
correct_test = torch.zeros([3]).cuda()
for x,  y in test_dl:
    with autocast():
        x = x
        y = y.cuda().long()
        outputs = model(x)
        correct_test = correct_test + (outputs.argmax(-2) == y).sum(0)

test_acc_list = correct_test/test_size*100
name1 = os.path.basename(args.input_folder)
name2 = os.path.basename(args.save_folder)
name3 = os.path.basename(os.path.dirname(args.save_folder))
file_name = f"data_{name1}___model_{name3}__{name2}.log"
os.makedirs(args.log_dir, exist_ok=True)
fout = open(os.path.join(args.log_dir, file_name), 'w')
[print(round(test_acc_list[i].item(),2), file = fout) for i in range(3)]