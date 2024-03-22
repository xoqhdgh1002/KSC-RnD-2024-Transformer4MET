from typing import Any
import torch
from torch import Tensor
from .base import Transform


class Normalize(Transform):
    offset: Tensor
    scale: Tensor

    def __init__(self,
                 offset: Tensor,
                 scale: Tensor,

    ) -> None:
        super().__init__()

        self.register_buffer('offset', offset)
        self.register_buffer('scale', scale)

    @torch.no_grad()
    def forward(self, input: Tensor) -> Tensor: # type: ignore
        return (input - self.offset) / self.scale

    @torch.no_grad()
    def inverse(self, input: Tensor) -> Tensor: # type: ignore
        return input * self.scale + self.offset

    @classmethod
    def from_data(cls, data: dict[str, Any]) -> 'Normalize':
        offset = torch.tensor(data['offset'], dtype=torch.float)
        scale = torch.tensor(data['scale'], dtype=torch.float)
        return cls(offset=offset, scale=scale)
