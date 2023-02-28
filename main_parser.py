from torch.utils.data import Dataset, DataLoader

import torch
from torch.cuda.amp import autocast
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from tqdm import tqdm
import argparse
import os
import attr_models
import global_args as gargs

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
parser.add_argument('--attr-arch', type=str, choices=gargs.VALID_ATTR_ARCHS)


args = parser.parse_args()

seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# train data
train_data = torch.load(os.path.join(
    args.input_folder, f"{args.input_type}_train.pt")).detach().cuda()
train_label = torch.load(os.path.join(
    args.input_folder, f"attr_labels_train.pt")).detach().cuda()
train_set = torch.utils.data.TensorDataset(train_data, train_label)
train_dl = torch.utils.data.DataLoader(
    train_set, batch_size=args.batch_size, shuffle=True)
train_size = train_data.shape[0]
print(train_data.shape, train_label.shape)

# test data
test_data = torch.load(os.path.join(
    args.input_folder, f"{args.input_type}_test.pt")).detach().cuda()
test_label = torch.load(os.path.join(
    args.input_folder, f"attr_labels_test.pt")).detach().cuda()
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

optimizer = optim.SGD(model.parameters(), lr=args.lr)
scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
criterion = torch.nn.CrossEntropyLoss()


os.makedirs(args.save_folder, exist_ok=True)

fout = open(os.path.join(args.save_folder, "output.log"), 'w')

best_acc = 0
best_epoch = 0

for i in range(args.epochs):
    model.train()

    # print(f"Epoch: {i}")
    print(f"Epoch: {i}", file=fout)

    correct_train = torch.zeros([3]).cuda()
    for x,  y in train_dl:
        with autocast():
            assert not x.max().isnan().any()
            x = x.cuda()
            y = y.cuda().long()
            outputs = model(x)
            correct_train = correct_train + (outputs.argmax(-2) == y).sum(0)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    train_acc = list((correct_train/train_size*100).cpu().numpy())
    train_print = ", ".join("{:.2f}".format(x) for x in train_acc)
    print(f"Epoch: {i}, Train acc: {train_print}")
    print(f"Train acc: {train_print}", file=fout)

    scheduler.step()
    model.eval()
    correct_test = torch.zeros([3]).cuda()
    for x,  y in test_dl:
        with autocast():
            assert not x.max().isnan().any()
            x = x.cuda()
            y = y.cuda().long()
            outputs = model(x)
            correct_test = correct_test + (outputs.argmax(-2) == y).sum(0)

    test_acc = correct_test/test_size*100
    if test_acc.mean() > best_acc:
        best_acc = test_acc.mean()
        best_epoch = i
        torch.save(model.state_dict(), os.path.join(
            args.save_folder, "best.pt"))
    
    test_acc = list(test_acc.cpu().numpy())
    test_print = ", ".join("{:.2f}".format(x) for x in test_acc)

    print(
        f"Epoch: {i}, Test acc: {test_print}, best acc: {best_acc:.2f}, best epoch: {best_epoch}")
    print(
        f"Test acc: {test_print}, best acc: {best_acc:.2f}, best epoch: {best_epoch}", file=fout)

torch.save(model.state_dict(), os.path.join(args.save_folder, "final.pt"))
# save path
