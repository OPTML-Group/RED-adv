import os
import torch
import global_args as gargs
import training_utils
import attr_models
import matplotlib.pyplot as plt
import numpy as np

def get_img(path):
    datas = training_utils.load_datas(path, gargs.FULL_RESULT_NAMES)
    x_adv, delta, adv_pred, ori_pred, target = datas
    idx = 8000
    while idx < 10000:
        while idx < 10000 and (adv_pred[idx] == target[idx] or ori_pred[idx] != target[idx]):
            idx += 1
        if idx == 10000:
            break
        yield idx, x_adv[idx], delta[idx]
        idx += 1

def get_layer_output(model, input, layer):
    x = input
    for i in range(layer):
        x = model.encoder[i](x)
    for i in range(3):
        x = model.encoder[layer][i](x)
    return x

def visualize_image(img, save_dir, save_name):
    pth = os.path.join(save_dir, save_name)
    import cv2
    img = cv2.resize(img, [128, 128], interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(pth, img)

def visualize_features(save_dir):
    criterion = torch.nn.CrossEntropyLoss()
    dataset = "cifar10"
    arch = "resnet9"
    attr_arch = "conv4"
    setting = "origin"
    attack = gargs.PGD_ATTACKS[2]
    atk = training_utils.get_attack_name(attack)
    da_name = f"{dataset}_{arch}"
    tp = "delta"
    model_path = os.path.join(gargs.PARSING_DIR, attr_arch, da_name, setting, atk, tp, "best.pt")

    model = attr_models.ConvNet4(
        num_channel=gargs.DATASET_NUM_CHANNEL[dataset],
        num_class=3,
        num_output=3,
        img_size=gargs.DATASET_INPUT_SIZE[dataset]
    )
    print(f"Load from {model_path}")
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()

    ks = gargs.KERNEL_SIZES
    acts = gargs.ACTIVATION_FUNCTIONS
    prunes = gargs.PRUNING_RATIOS
    for idx_k, k in enumerate(ks):
        for idx_a, act in enumerate(acts):
            for idx_p, prune in enumerate(prunes):
                dir_name = training_utils.get_model_name(2, k, act, prune)
                victim = dir_name

                atk_img_dir = os.path.join(gargs.ATK_DIR, da_name, atk, victim)
                cnt = 0
                print(f"Load from {atk_img_dir}")
                for idx, x_adv, delta in get_img(atk_img_dir):
                    if tp == "delta":
                        raw_input = delta.reshape([1, 3, 32, 32])
                    else:
                        raw_input = x_adv.reshape([1, 3, 32, 32])
                    gt = torch.LongTensor([(idx_k, idx_a, idx_p)])
                    raw_input.requires_grad = True
                    with torch.no_grad():
                        output = model(raw_input)
                        pred = output.argmax(-2)
                        if (pred != gt).any():
                            continue
                        origin = x_adv - delta
                        origin = (origin.permute([1, 2, 0]).cpu().numpy() * gargs.DATASET_STD[dataset] + gargs.DATASET_MEAN[dataset]) * 255
                        x_adv = (x_adv.permute([1, 2, 0]).cpu().numpy() * gargs.DATASET_STD[dataset] + gargs.DATASET_MEAN[dataset]) * 255
                        delta = (x_adv - origin) + 128
                        save_path = os.path.join(save_dir, str(idx))
                        os.makedirs(save_path, exist_ok=True)
                        visualize_image(origin, save_path, f"origin_{dir_name}.png")
                        visualize_image(x_adv, save_path, f"adv_{dir_name}.png")
                        visualize_image(delta, save_path, f"delta_{dir_name}.png")
                        for layer in range(4):
                            layer_i = get_layer_output(model, raw_input, layer)
                            layer_i = layer_i[0].cpu().detach().numpy().mean(axis=0)
                            layer_i -= layer_i.min()
                            layer_i /= layer_i.max()
                            layer_i *= 255
                            visualize_image(layer_i, save_path, f"layer_{layer}_{dir_name}.png")
                        cnt += 1
                        if cnt > 9:
                            break
                    for i in range(3):
                        model.zero_grad()
                        raw_input.grad = None
                        output = model(raw_input)[:, :, i]
                        loss = criterion(output, gt[:, i])
                        loss.backward()
                        grad = raw_input.grad[0].permute([1, 2, 0]).cpu().numpy()
                        grad = (grad ** 2).sum(axis = -1) ** 0.5
                        grad = grad / grad.max() * 255
                        # grad = grad / np.abs(grad).max() * 128 + 128
                        visualize_image(grad, save_path, f"img_grad_{i}_{dir_name}.png")

    

if __name__ == "__main__":
    import shutil
    shutil.rmtree("visualize", ignore_errors=True)
    visualize_features("./visualize")