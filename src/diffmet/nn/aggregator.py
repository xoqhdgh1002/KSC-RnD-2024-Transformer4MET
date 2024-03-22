from torch import Tensor
import torch.nn as nn


def masked_sum(input: Tensor,
               data_mask: Tensor,
               dim: int,
) -> Tensor:
    pad_mask = data_mask.logical_not().unsqueeze(2)
    output = input.masked_fill(mask=pad_mask, value=0)
    output = output.sum(dim=dim)
    return output


def masked_mean(input: Tensor,
                data_mask: Tensor,
                dim: int,
) -> Tensor:
    summed = masked_sum(input, data_mask, dim=dim)
    length = data_mask.sum(dim=-1, keepdim=True).to(summed.dtype)
    return summed / length


class MaskedSum(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, input: Tensor, data_mask: Tensor) -> Tensor:
        return masked_sum(input, data_mask, dim=self.dim)


class MaskedMean(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, input: Tensor, data_mask: Tensor) -> Tensor:
        return masked_mean(input, data_mask, dim=self.dim)
