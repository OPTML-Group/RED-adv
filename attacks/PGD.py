import torch


# apply infinity-norm clamp when ord is None
def PGD(X, y, model, loss_fn, scaler, steps, eps, ord=None):
    max_X, min_X = torch.max(X), torch.min(X)
    eps_X = float(eps * (max_X - min_X))

    eps_iter = 1.25 * eps_X / steps
    delta = torch.zeros_like(X).cuda()
    delta.requires_grad = True
    ori_X = X.data
    for _ in range(steps):
        output = model(X + delta)
        loss = loss_fn(output, y)
        scaler.scale(loss).backward()
        grad = delta.grad.detach()
        with torch.no_grad():
            if ord is None:
                delta.data += eps_iter * torch.sign(grad)
                delta.data = torch.clamp(delta.data, -eps_X, eps_X)
            else:
                delta.data += batch_multiply(eps_iter, grad)
                delta.data = clamp_by_pnorm(delta.data, ord, eps_X)
            X = torch.clamp(ori_X + delta, min_X, max_X)
    return X

def PGD_l2(X, y, model, loss_fn, scaler, steps, eps):
    return PGD(X, y, model, loss_fn, scaler, steps, eps, 2)


def _get_norm_batch(x, p):
    batch_size = x.size(0)
    return x.abs().pow(p).reshape(batch_size, -1).sum(dim=1).pow(1. / p)


def batch_multiply(float_or_vector, tensor):
    if isinstance(float_or_vector, torch.Tensor):
        tensor = _batch_multiply_tensor_by_vector(float_or_vector, tensor)
    elif isinstance(float_or_vector, float):
        tensor *= float_or_vector
    else:
        raise TypeError("Value has to be float or torch.Tensor")
    return tensor


def _batch_multiply_tensor_by_vector(vector, batch_tensor):
    """Equivalent to the following
    for ii in range(len(vector)):
        batch_tensor.data[ii] *= vector[ii]
    return batch_tensor
    """
    return (
        batch_tensor.transpose(0, -1) * vector).transpose(0, -1).contiguous()


def clamp_by_pnorm(x, p, r):
    assert isinstance(p, float) or isinstance(p, int)
    norm = _get_norm_batch(x, p)
    if isinstance(r, torch.Tensor):
        assert norm.size() == r.size()
    else:
        assert isinstance(r, float)
    factor = torch.min(r / norm, torch.ones_like(norm))
    return batch_multiply(factor, x)
