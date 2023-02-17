import torch
import torch.nn as nn


def _zosignsgd(X, y, model, steps=500, alpha=0.0005,
               mean=[125.307/255, 122.961/255, 113.8575/255], std=[51.5865/255, 50.847/255, 51.255/255],
               q=10, eta=0.01):

    # configure normalization
    n_channels = len(mean)
    mean = torch.tensor(mean).reshape(1, n_channels, 1, 1).cuda()
    std = torch.tensor(std).reshape(1, n_channels, 1, 1).cuda()

    # inverse normalize to [0,1]
    X = X * std + mean
    X_adv = X.clone().detach()

    _shape = list(X_adv.shape)
    num_axes = len(_shape[1:])
    gs_t = torch.zeros_like(X_adv)

    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for i in range(steps):
            # estimate grad
            for _ in range(q):
                exp_noise = torch.randn_like(X_adv)
                fxs_t = X_adv + eta * exp_noise
                bxs_t = X_adv

                est_deriv = (loss_fn(model((fxs_t - mean)/std), y) -
                             loss_fn(model((bxs_t - mean)/std), y)) / eta
                gs_t += est_deriv.reshape(-1, *[1] * num_axes) * exp_noise

            X_adv = X_adv.detach() + alpha * gs_t.sign()
            X_adv = torch.clamp(X_adv, min=0, max=1).detach()
            adv_out = model((X_adv - mean)/std)
            idx_adv = adv_out.argmax(1).ne(y)
            print(
                f"at step {i}: atk rate: {sum(idx_adv).item()/len(idx_adv)*100:.2f} %")

    X_adv = (X_adv - mean) / std
    return X_adv


class ZoSignSgd:
    def __init__(self, model):
        self.model = model
        self.mean = [0, 0, 0]
        self.std = [1, 1, 1]

    def set_normalization_used(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        return _zosignsgd(image, target, self.model, mean=self.mean, std=self.std)
