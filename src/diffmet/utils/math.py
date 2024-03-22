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
