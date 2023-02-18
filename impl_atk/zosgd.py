import torch
import torch.nn as nn


def _zosgd(X, y, model, steps=500, eta=0.0005, eps=8/255,
           mean=[125.307/255, 122.961/255, 113.8575/255], std=[51.5865/255, 50.847/255, 51.255/255],
           q=10, mu=0.01, sign=False, est_time=1):
    n_channels = len(mean)
    mean = torch.tensor(mean).reshape(1, n_channels, 1, 1).cuda()
    std = torch.tensor(std).reshape(1, n_channels, 1, 1).cuda()

    X = X * std + mean  # inverse normalize to [0,1]
    assert (X < 1.01).all() and (X > -0.01).all()
    X_adv = X.clone().detach()

    _shape = list(X_adv.shape)
    num_axes = len(_shape[1:])

    loss_fn = nn.CrossEntropyLoss(reduction="none")

    with torch.no_grad():
        for i in range(steps):
            gs_t = torch.zeros_like(X_adv)  # be careful

            if est_time == 1:
                X_adv_loss = loss_fn(model((X_adv - mean)/std), y)

            for _ in range(q // est_time):
                z = torch.randn_like(X_adv)
                exp_noise = z / z.norm(dim=[1, 2, 3], p=2, keepdim=True)

                if est_time == 1:
                    x_1 = X_adv + mu * exp_noise

                    est_deriv = (loss_fn(model((x_1 - mean)/std), y) -
                                 X_adv_loss) / mu
                elif est_time == 2:
                    x_1 = X_adv + mu * exp_noise
                    x_2 = X_adv - mu * exp_noise
                    est_deriv = (loss_fn(model((x_1 - mean)/std), y) -
                                 loss_fn(model((x_2 - mean)/std), y)) / 2 / mu
                else:
                    raise NotImplementedError(
                        f"est_time {est_time} not implemented")

                gs_t += est_deriv.reshape(-1, *[1] * num_axes) * exp_noise
            gs_t = gs_t / q
            if sign:
                X_adv = X_adv.detach() + eta * gs_t.sign()
            else:
                gs_t = gs_t / gs_t.norm(dim=[1, 2, 3], p=2, keepdim=True)
                gs_t *= (3 * 32 * 32) ** 0.5
                X_adv = X_adv.detach() + eta * gs_t
            delta = torch.clamp(X_adv - X, min=-eps, max=eps)
            X_adv = torch.clamp(X + delta, min=0, max=1).detach()

            # adv_out = model((X_adv - mean)/std)
            # clean_out = model((X - mean)/std)
            # idx_adv = adv_out.argmax(1).ne(y)
            # idx_clean = clean_out.argmax(1).eq(y)
            # print(
            #     f"at step {i}: clf right & atk succ: {sum(idx_adv * idx_clean).item()/len(idx_adv)*100:.2f} %")

    X_adv = (X_adv - mean) / std
    return X_adv


class _Impl:
    def __init__(self, model, eps, norm, sign, est_time):
        self.model = model
        self.norm = norm
        # TODO: impl L2 proj
        self.eps = eps
        self.mean = [0, 0, 0]
        self.std = [1, 1, 1]

        self.sign = sign
        self.est_time = est_time

    def set_normalization_used(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        return _zosgd(image, target, self.model, mean=self.mean, std=self.std, eps=self.eps, sign=self.sign, est_time=self.est_time)


class ZoSignSgd(_Impl):
    def __init__(self, model, eps, norm):
        super().__init__(model, eps, norm, sign=True, est_time=1)


class ZoSgd(_Impl):
    def __init__(self, model, eps, norm):
        super().__init__(model, eps, norm, sign=False, est_time=1)


class Bandit(_Impl):
    def __init__(self, model, eps, norm):
        super().__init__(model, eps, norm, sign=False, est_time=2)
