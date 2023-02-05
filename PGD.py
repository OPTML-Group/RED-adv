import torch

def PGD(X, y, model, loss_fn, scaler, steps, eps):
    max_X, min_X = torch.max(X), torch.min(X)
    eps_X = eps * (max_X - min_X)
    # delta = torch.zeros_like(X).uniform_(-eps_X, eps_X).cuda()
    delta = torch.zeros_like(X).cuda()
    delta.requires_grad = True
    ori_X = X.data
    for _ in range(steps):
        output = model(X + delta)
        loss = loss_fn(output, y)
        scaler.scale(loss).backward()
        grad = delta.grad.detach()
        delta.data = torch.clamp(delta + 1.25 * eps_X / steps * torch.sign(grad), -eps_X, eps_X)
        X = torch.clamp(ori_X + delta, min_X, max_X)
    return X

def PGD_l2(X, y, model, loss_fn, scaler, steps, eps):
    ord = 2
    max_X, min_X = torch.max(X), torch.min(X)
    eps_X = float(eps * (max_X - min_X))
    
    eps_iter = 1.25 * eps_X / steps
    # delta = torch.zeros_like(X).uniform_(-eps_X, eps_X).cuda()
    delta = torch.zeros_like(X).cuda()
    delta.requires_grad = True
    ori_X = X.data
    for _ in range(steps):
        output = model(X + delta)
        loss = loss_fn(output, y)
        scaler.scale(loss).backward()
        grad = delta.grad.detach()
        grad = normalize_by_pnorm(grad)
        delta.data = delta.data + batch_multiply(eps_iter, grad)
        # delta.data = clamp(xvar.data + delta.data, min_X, clip_max
        #                 ) - xvar.data
        
        delta.data = clamp_by_pnorm(delta.data, ord, eps_X)
        X = torch.clamp(ori_X + delta, min_X, max_X)
    return X

def clamp_by_pnorm(x, p, r):
    assert isinstance(p, float) or isinstance(p, int)
    norm = _get_norm_batch(x, p)
    if isinstance(r, torch.Tensor):
        assert norm.size() == r.size()
    else:
        assert isinstance(r, float)
    factor = torch.min(r / norm, torch.ones_like(norm))
    return batch_multiply(factor, x)

def _get_norm_batch(x, p):
    batch_size = x.size(0)
    return x.abs().pow(p).reshape(batch_size, -1).sum(dim=1).pow(1. / p)

def batch_multiply(float_or_vector, tensor):
    if isinstance(float_or_vector, torch.Tensor):
        # assert len(float_or_vector) == len(tensor)
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

def normalize_by_pnorm(x, p=2, small_constant=1e-6):
    """
    Normalize gradients for gradient (not gradient sign) attacks.
    # TODO: move this function to utils
    :param x: tensor containing the gradients on the input.
    :param p: (optional) order of the norm for the normalization (1 or 2).
    :param small_constant: (optional float) to avoid dividing by zero.
    :return: normalized gradients.
    """
    # loss is averaged over the batch so need to multiply the batch
    # size to find the actual gradient of each input sample

    assert isinstance(p, float) or isinstance(p, int)
    norm = _get_norm_batch(x, p)
    norm = torch.max(norm, torch.ones_like(norm) * small_constant)
    return batch_multiply(1. / norm, x)

def clamp(input, min=None, max=None):
    ndim = input.ndimension()
    if min is None:
        pass
    elif isinstance(min, (float, int)):
        input = torch.clamp(input, min=min)
    elif isinstance(min, torch.Tensor):
        if min.ndimension() == ndim - 1 and min.shape == input.shape[1:]:
            input = torch.max(input, min.view(1, *min.shape))
        else:
            assert min.shape == input.shape
            input = torch.max(input, min)
    else:
        raise ValueError("min can only be None | float | torch.Tensor")

    if max is None:
        pass
    elif isinstance(max, (float, int)):
        input = torch.clamp(input, max=max)
    elif isinstance(max, torch.Tensor):
        if max.ndimension() == ndim - 1 and max.shape == input.shape[1:]:
            input = torch.min(input, max.view(1, *max.shape))
        else:
            assert max.shape == input.shape
            input = torch.min(input, max)
    else:
        raise ValueError("max can only be None | float | torch.Tensor")
    return input