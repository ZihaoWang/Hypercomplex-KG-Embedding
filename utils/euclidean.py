"""Euclidean operations utils functions."""

import torch


def euc_sqdistance(x, y, eval_mode=False):
    """Compute euclidean squared distance between tensors.

    Args:
        x: torch.Tensor of shape (N1 x d)
        y: torch.Tensor of shape (N2 x d)
        eval_mode: boolean

    Returns:
        torch.Tensor of shape N1 x 1 with pairwise squared distances if eval_mode is false
        else torch.Tensor of shape N1 x N2 with all-pairs distances

    """
    #print(x)
    #exit()
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)
    #print(x2.size())
    #print(y2.size())
    #exit()
    if eval_mode:
        y2 = y2.t()
        xy = x @ y.t()
    else:
        assert x.shape[0] == y.shape[0]
        xy = torch.sum(x * y, dim=-1, keepdim=True)

    #print(x2.size())
    #print(y2.size())
    #print(xy.size())
    #exit()

    return x2 + y2 - 2 * xy

def givens_incomplete_DE_mult(r, x):
    """Givens rotations.
    Args:
        r: torch.Tensor of shape (N x d), rotation parameters
        x: torch.Tensor of shape (N x d), points to rotate

    Returns:    
        torch.Tensor os shape (N x d) representing rotation of x by r
    """

    s_h = x[0]
    #x_h = x[1]
    y_h = x[1]
    z_h = x[2]

    s_rot = r[0]
    x_rot = r[1]
    y_rot = r[2]
    z_rot = r[3]


    #denominator_b = torch.sqrt(s_rot ** 2 + x_rot ** 2 + y_rot ** 2 + z_rot ** 2)
    #s_rot = s_rot / denominator_b
    #x_rot = x_rot / denominator_b
    #y_rot = y_rot / denominator_b
    #z_rot = z_rot / denominator_b

    #print(y_h.size())
    #print(y_rot.size())
    #exit()
    A = s_h * s_rot + y_h * y_rot + z_h * z_rot
    B = s_h * x_rot - y_h * z_rot + y_rot * z_h
    C = s_h * y_rot + s_rot * y_h + z_h * x_rot
    D = s_h * z_rot + s_rot * z_h - x_rot * y_h


    return A, B, C, D


def givens_DE_rotations(r, x):
    """Givens rotations.
    Args:
        r: torch.Tensor of shape (N x d), rotation parameters
        x: torch.Tensor of shape (N x d), points to rotate

    Returns:    
        torch.Tensor os shape (N x d) representing rotation of x by r
    """

    s_h = x[0]
    x_h = x[1]
    y_h = x[2]
    z_h = x[3]

    s_rot = r[0]
    x_rot = r[1]
    y_rot = r[2]
    z_rot = r[3]


    denominator_b = torch.sqrt(s_rot ** 2 + x_rot ** 2 + y_rot ** 2 + z_rot ** 2)
    s_rot = s_rot / denominator_b
    x_rot = x_rot / denominator_b
    y_rot = y_rot / denominator_b
    z_rot = z_rot / denominator_b


    A = s_h * s_rot - x_h * x_rot + y_h * y_rot + z_h * z_rot
    B = s_h * x_rot + s_rot * x_h - y_h * z_rot + y_rot * z_h
    C = s_h * y_rot + s_rot * y_h + z_h * x_rot - z_rot * x_h
    D = s_h * z_rot + s_rot * z_h + x_h * y_rot - x_rot * y_h


    return A, B, C, D

def givens_DE_rotations_m(r, x):
    """Givens rotations.
    Args:
        r: torch.Tensor of shape (N x d), rotation parameters
        x: torch.Tensor of shape (N x d), points to rotate

    Returns:    
        torch.Tensor os shape (N x d) representing rotation of x by r
    """

    s_h = x[0]
    x_h = x[1]
    y_h = x[2]
    z_h = x[3]

    s_rot = r[0]
    x_rot = r[1]
    y_rot = r[2]
    z_rot = r[3]


    denominator_b = torch.sqrt(s_rot ** 2 + x_rot ** 2 + y_rot ** 2 + z_rot ** 2)
    s_rot = s_rot / denominator_b
    x_rot = x_rot / denominator_b
    y_rot = y_rot / denominator_b
    z_rot = z_rot / denominator_b

    A = s_h * s_rot - x_h * x_rot + y_h * y_rot + z_h * z_rot
    B = s_h * x_rot + x_h * s_rot - y_h * z_rot + z_h * y_rot
    C = s_h * y_rot - x_h * z_rot + y_h * s_rot + z_h * x_rot
    D = s_h * z_rot + x_h * y_rot - y_h * x_rot + z_h * s_rot
    
    return A, B, C, D

def givens_DE_product(r, x, eval_mode):
    """Givens rotations.
    Args:
        r: torch.Tensor of shape (N x d), rotation parameters
        x: torch.Tensor of shape (N x d), points to rotate

    Returns:    
        torch.Tensor os shape (N x d) representing rotation of x by r
    """

    s_h = x[0]
    x_h = x[1]
    y_h = x[2]
    z_h = x[3]

    s_rot = r[0]
    x_rot = r[1]
    y_rot = r[2]
    z_rot = r[3]


    #denominator_b = torch.sqrt(s_rot ** 2 + x_rot ** 2 + y_rot ** 2 + z_rot ** 2)
    #s_rot = s_rot / denominator_b
    #x_rot = x_rot / denominator_b
    #y_rot = y_rot / denominator_b
    #z_rot = z_rot / denominator_b

    if eval_mode:

        s_rot = r[0].transpose(0,1)
        x_rot = r[1].transpose(0,1)
        y_rot = r[2].transpose(0,1)
        z_rot = r[3].transpose(0,1)


        A = s_h @ s_rot - x_h @ x_rot + y_h @ y_rot + z_h @ z_rot
        B = s_h @ x_rot + x_h @ s_rot - y_h @ z_rot + z_h @ y_rot
        C = s_h @ y_rot + y_h @ s_rot + z_h @ x_rot - x_h @ z_rot
        D = s_h @ z_rot + z_h @ s_rot + x_h @ y_rot - y_h @ x_rot
    else:
        A = s_h * s_rot - x_h * x_rot + y_h * y_rot + z_h * z_rot
        B = s_h * x_rot + s_rot * x_h - y_h * z_rot + y_rot * z_h
        C = s_h * y_rot + s_rot * y_h + z_h * x_rot - z_rot * x_h
        D = s_h * z_rot + s_rot * z_h + x_h * y_rot - x_rot * y_h


    return A, B, C, D


def givens_complex_product(r, x, eval_mode):
    """Givens rotations.
    Args:
        r: torch.Tensor of shape (N x d), rotation parameters
        x: torch.Tensor of shape (N x d), points to rotate

    Returns:    
        torch.Tensor os shape (N x d) representing rotation of x by r
    """

    s_h = x[0]
    x_h = x[1]

    s_rot = r[0]
    x_rot = r[1]


    #denominator_b = torch.sqrt(s_rot ** 2 + x_rot ** 2)
    #s_rot = s_rot / denominator_b
    #x_rot = x_rot / denominator_b

    if eval_mode:

        s_rot = r[0].transpose(0,1)
        x_rot = r[1].transpose(0,1)    


        A = s_h @ s_rot - x_h @ x_rot
        B = s_h @ x_rot + x_h @ s_rot
    else:
        A = s_h * s_rot - x_h * x_rot 
        B = s_h * x_rot + x_h * s_rot

    return A, B


def givens_QuatE_rotations(r, x):
    """Givens rotations.
    Args:
        r: torch.Tensor of shape (N x d), rotation parameters
        x: torch.Tensor of shape (N x d), points to rotate

    Returns:    
        torch.Tensor os shape (N x d) representing rotation of x by r
    """

    s_h = x[0]
    x_h = x[1]
    y_h = x[2]
    z_h = x[3]

    s_rot = r[0]
    x_rot = r[1]
    y_rot = r[2]
    z_rot = r[3]


    denominator_b = torch.sqrt(s_rot ** 2 + x_rot ** 2 + y_rot ** 2 + z_rot ** 2)
    s_rot = s_rot / denominator_b
    x_rot = x_rot / denominator_b
    y_rot = y_rot / denominator_b
    z_rot = z_rot / denominator_b


    A = s_h * s_rot - x_h * x_rot - y_h * y_rot - z_h * z_rot
    B = s_h * x_rot + s_rot * x_h + y_h * z_rot - y_rot * z_h
    C = s_h * y_rot + s_rot * y_h + z_h * x_rot - z_rot * x_h
    D = s_h * z_rot + s_rot * z_h + x_h * y_rot - x_rot * y_h


    return A, B, C, D



def givens_rotations(r, x):
    """Givens rotations.

    Args:
        r: torch.Tensor of shape (N x d), rotation parameters
        x: torch.Tensor of shape (N x d), points to rotate

    Returns:
        torch.Tensor os shape (N x d) representing rotation of x by r
    """
    givens = r.view((r.shape[0], -1, 2))
    givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True).clamp_min(1e-15)
    x = x.view((r.shape[0], -1, 2))
    x_rot = givens[:, :, 0:1] * x + givens[:, :, 1:] * torch.cat((-x[:, :, 1:], x[:, :, 0:1]), dim=-1)
    return x_rot.view((r.shape[0], -1))


def givens_reflection(r, x):
    """Givens reflections.

    Args:
        r: torch.Tensor of shape (N x d), rotation parameters
        x: torch.Tensor of shape (N x d), points to reflect

    Returns:
        torch.Tensor os shape (N x d) representing reflection of x by r
    """
    givens = r.view((r.shape[0], -1, 2))
    givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True).clamp_min(1e-15)
    x = x.view((r.shape[0], -1, 2))
    x_ref = givens[:, :, 0:1] * torch.cat((x[:, :, 0:1], -x[:, :, 1:]), dim=-1) + givens[:, :, 1:] * torch.cat(
        (x[:, :, 1:], x[:, :, 0:1]), dim=-1)
    return x_ref.view((r.shape[0], -1))
