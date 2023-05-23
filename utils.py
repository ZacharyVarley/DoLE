import torch


def normalize(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())


def checkerboard(A, B, k=20):
    """
        shape: dimensions of output tensor
        k: edge size of small checker squares
    """
    b, c, h, w = A.shape
    assert b == 1
    assert ((c == 1) or (c == 3))

    coords_div = torch.div(torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w))), k, rounding_mode='trunc')
    coords_div = coords_div.to(A.device)
    checker = (coords_div.sum(0) % 2)[None, None]

    return A * checker + B * (1 - checker)