from IPython import embed
from torch.utils.data import Dataset, DataLoader

import torch as ch
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
ch.manual_seed(seed)
ch.cuda.manual_seed(seed)
ch.cuda.manual_seed_all(seed)
ch.backends.cudnn.benchmark = False
ch.backends.cudnn.deterministic = True

data = ch.load(os.path.join(args.input_folder, f"{args.input_type}.pt"))
data = data.detach().cuda()
print(data.shape)
label = ch.load(os.path.join(args.input_folder, f"attr_labels.pt"))
label = label.detach().cuda()
print(label.shape)
# embed()

class M(Dataset):
    def __init__(self, data, label):
        super().__init__()
        self.data = data
        self.label = label

    def __getitem__(self, i):
        return self.data[i], self.label[i]

    def __len__(self):
        return len(self.data)


ds = M(data, label)

train_size = int(0.8 * len(ds))
test_size = len(ds) - train_size
train_dataset, test_dataset = ch.utils.data.random_split(
    ds, [train_size, test_size])

train_dl = DataLoader(train_dataset, batch_size=args.batch_size)
test_dl = DataLoader(test_dataset, batch_size=args.batch_size)


use_cuda = ch.cuda.is_available()
device = ch.device("cuda" if use_cuda else "cpu")

dataset = args.dataset
model = attr_models.get_model(
    args.attr_arch,
    num_channel=gargs.DATASET_NUM_CHANNEL[dataset],
    num_class=5,
    num_output=3,
    img_size=gargs.DATASET_INPUT_SIZE[dataset]
).to(device)

model.load_state_dict(ch.load(os.path.join(args.save_folder, "final.pt")), strict=False)

model.cuda()
model.eval()
correct_test = ch.zeros([3]).cuda()
for x,  y in tqdm(test_dl):
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