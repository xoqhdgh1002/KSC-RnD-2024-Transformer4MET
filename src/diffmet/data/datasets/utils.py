import awkward as ak
import torch


def convert_ak_to_tensor(arr: ak.Array) -> torch.Tensor:
    return torch.from_numpy(ak.to_numpy(arr))
