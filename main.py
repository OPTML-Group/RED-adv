import os
import time

import torch
import torch.nn as nn


import utils
import models
import trainer
import arg_parser

adv_name = "xs.pt"
delta_name = "data.pt"
label_name = "label.pt"
files = ['data.pt', 'label.pt']

def main():
    args = arg_parser.parse_args()
    dir = args.dataset_dir

    if args.input_type == "x_adv":
        suffix = adv_name
    elif args.input_type == "delta":
        suffix = delta_name
    else:
        exit(0)
    data = torch.load(os.path.join(dir, suffix)).float()
    label = torch.load(os.path.join(dir, label_name)).long()
    n_channels = data.shape[1]
    n_outputs = label.shape[1]

    cnt = utils.count_element_vector(label)
    train_set, test_set = utils.get_datasets_from_tensor_with_cnt(
        data, label, cnt, cuda=True)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False)

    model = models.Net(num_channels=n_channels, num_classes=3, num_outputs=n_outputs).cuda()

    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer, scheduler = trainer.get_optimizer_and_scheduler(model, args)


    for epoch in range(args.epochs):
        start_time = time.time()
        print(optimizer.state_dict()['param_groups'][0]['lr'])

        trainer.train_epoch(train_loader, model,
              criterion, optimizer, epoch, args)

        scheduler.step()
        print("one epoch duration:{}".format(time.time()-start_time))

        acc = trainer.validate(test_loader, model, criterion, args)

    acc = trainer.validate(test_loader, model, criterion, args)

if __name__ == "__main__":
    main()