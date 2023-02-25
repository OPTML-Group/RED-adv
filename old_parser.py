# from IPython import embed
from torch.utils.data import Dataset, DataLoader

import torch as ch
from torch.cuda.amp import autocast 
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from tqdm import tqdm
import argparse
import os

from models.dncnn import DnCNN

parser = argparse.ArgumentParser(description='train clf')

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--epochs', type=int, default=100)

parser.add_argument('--input_folder', type=str)
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--attack', type=str, default='PGD_eps8_alpha1_steps10')
parser.add_argument('--input-type', type=str, default='delta')
parser.add_argument('--save_folder', type=str)

# dncnn args
parser.add_argument('--enable_denoiser', action="store_true", default=False, help='whether enable dncnn training in the pipeline')
parser.add_argument('--ims_adv_data', type=str, default='x_adv')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')     
parser.add_argument('--denoiser_lr', type=float, default=1e-4)    
parser.add_argument('--lambda1',type=float, default=15, help='the coefficient of attribution loss')
parser.add_argument('--gamma1',type=float, default=1, help='the coefficient of mae loss')
parser.add_argument('--denoiser', type=str, help='Path to a denoiser ')
parser.add_argument('--pretrained_attr', type=str, help='Path to a pretrained attr net')


args = parser.parse_args()

seed = args.seed
ch.manual_seed(seed)
ch.cuda.manual_seed(seed)
ch.cuda.manual_seed_all(seed)
ch.backends.cudnn.benchmark = False
ch.backends.cudnn.deterministic = True

data = ch.load(os.path.join(args.input_folder, f"{args.input_type}.pt"))
# train path
data = data.detach().cuda()
print(data.shape)
label = ch.load(os.path.join(args.input_folder, f"attr_labels.pt"))
# label path
label = label.detach().cuda()
print(label.shape)

# adv img path
ims_adv_data = ch.load(os.path.join(args.input_folder, f"{args.ims_adv_data}.pt"))
ims_adv_data.cuda()

class M(Dataset):
    def __init__(self, data, ims_adv_data, label):
        super().__init__()
        self.data = data
        self.ims_adv_data = ims_adv_data
        self.label = label
        
    def __getitem__(self, i):
        return self.data[i], self.ims_adv_data[i], self.label[i]

    def __len__(self):
        return len(self.data)


ds = M(data, ims_adv_data, label)

train_size = int(0.8 * len(ds))
test_size = len(ds) - train_size
train_dataset, test_dataset = ch.utils.data.random_split(
    ds, [train_size, test_size])
# n = 10000
# train_dataset, test_dataset, _ = ch.utils.data.random_split(
#     ds, [n, test_size, len(ds) - n - test_size])
train_dl = DataLoader(train_dataset, batch_size=args.batch_size)
test_dl = DataLoader(test_dataset, batch_size=args.batch_size)


class AttrNet(ch.nn.Module):
    def __init__(self, num_channel=3, num_class=5, num_output=3):
        super(AttrNet, self).__init__()
        self.conv1 = ch.nn.Conv2d(num_channel, 32, 3, 1)
        self.conv2 = ch.nn.Conv2d(32, 64, 3, 1)
        # self.bn1 = ch.nn.BatchNorm2d(3)
        # self.bn2 = ch.nn.BatchNorm2d(32)
        # self.bn3 = ch.nn.BatchNorm2d(64)
        self.dropout1 = ch.nn.Dropout(0.25)
        self.dropout2 = ch.nn.Dropout(0.25)
        self.fc1 = ch.nn.Linear(12544, 256)
        self.fc2 = ch.nn.Linear(256, num_class * num_output)
        self.num_class = num_class
        self.num_output = num_output

    def forward(self, x):
        # x = self.bn1(x)
        x = self.conv1(x)
        # x = self.bn2(x)
        x = F.relu(x)
        x = self.conv2(x)
        # x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = ch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        outputs = x.view([-1, self.num_class, self.num_output])
        return outputs


use_cuda = ch.cuda.is_available()
device = ch.device("cuda" if use_cuda else "cpu")



attr_model = AttrNet().to(device)
if args.enable_denoiser:
    # denoiser setup with pretrained attr
    attr_checkpoint = ch.load(args.pretrained_attr)
    attr_model.load_state_dict(attr_checkpoint['state_dict'])
    denoiser = DnCNN(image_channels=3, depth=17, n_channels=64).cuda()
    checkpoint = ch.load(args.denoiser)
    denoiser = denoiser.cuda()
    denoiser.load_state_dict(checkpoint['state_dict'])
    optimizer = optim.SGD([{'params': denoiser.parameters()}, {'params': attr_model.parameters(), 'lr': args.lr}], lr=args.denoiser_lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    MAE = ch.nn.L1Loss(size_average=None, reduce=None, reduction='mean').cuda()
else:
    optimizer = optim.SGD(attr_model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

criterion = ch.nn.CrossEntropyLoss()
attr_model.cuda()
attr_model.train()

os.makedirs(args.save_folder, exist_ok=True)

fout = open(os.path.join(args.save_folder, "output.log"), 'w')

for i in range(args.epochs):
    attr_model.train()
    if args.enable_denoiser:
        denoiser.train()
    correct_train = ch.zeros([3]).cuda()
    for x, x_ims_adv, y in tqdm(train_dl):
        with autocast():
            assert not x.max().isnan().any()
            x = x.cuda()
            x_ims_adv = x_ims_adv.cuda()
            cleans = x_ims_adv - x
            y = y.cuda().long()

            if args.enable_denoiser:
                after_denoise = denoiser(x_ims_adv)
                delta_estimate = x_ims_adv - after_denoise
                outputs = attr_model(delta_estimate)
            else:
                outputs = attr_model(x)
                
            correct_train = correct_train + (outputs.argmax(-2) == y).sum(0)

            
            loss1 = criterion(outputs, y)
            
            if args.enable_denoiser:
                loss2 = MAE(after_denoise, cleans)
                loss = args.lambda1 * loss1 + args.gamma1 * loss2
            else:
                loss = loss1
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            
    print(f"Epoch: {i+1}")
    print(f"Train acc:{correct_train/train_size*100}")
    print(f"Train loss: {loss.item()}")
    print(f"Epoch: {i+1}", file = fout)
    print(f"Train acc:{correct_train/train_size*100}", file = fout)
    print(f"Train loss: {loss.item()}", file = fout)


    scheduler.step()
    attr_model.eval()
    if args.enable_denoiser:
        denoiser.eval()
    correct_test = ch.zeros([3]).cuda()
    for x, x_ims_adv, y in tqdm(test_dl):
        with autocast():
            assert not x.max().isnan().any()
            x = x.cuda()
            x_ims_adv = x_ims_adv.cuda()
            y = y.cuda().long()
            if args.enable_denoiser:
                after_denoise = denoiser(x_ims_adv)
                delta_estimate = x_ims_adv - after_denoise
                outputs = attr_model(delta_estimate)
            else:
                outputs = attr_model(x)
            correct_test = correct_test + (outputs.argmax(-2) == y).sum(0)

    print(f"Test acc:{correct_test/test_size*100}")
    print(f"Test loss: {loss.item()}")
    print(f"Test acc:{correct_test/test_size*100}", file = fout)
    print(f"Test loss: {loss.item()}", file = fout)

ch.save(attr_model.state_dict(), os.path.join(args.save_folder, "final.pt"))
if args.enable_denoiser:
    ch.save(denoiser.state_dict(), os.path.join(args.save_folder, "denoiser.pt"))
# save path
