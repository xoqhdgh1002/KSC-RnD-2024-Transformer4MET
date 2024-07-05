import torch
from torch import Tensor

def to_polar(met: Tensor) -> Tensor:
    """
    Args:
        met: a tensor with the shape of (N, 2)
    Returns:
        a tensor with the shape of (N, 2)
    """

    px, py = met.T
    pt = torch.hypot(px, py)
    phi = torch.atan2(py, px)
    polar_met = torch.stack([pt, phi], dim=1)
    return polar_met


def rectify_phi(phi: Tensor) -> Tensor:
    """
    adapted from https://github.com/scikit-hep/vector/blob/v1.4.1/src/vector/_compute/planar/add.py#L30-L31
    """
    return (phi + torch.pi) % (2 * torch.pi) - torch.pi


def compute_delta_phi(phi0: Tensor, phi1: Tensor) -> Tensor:
    return rectify_phi(phi0 - phi1)
