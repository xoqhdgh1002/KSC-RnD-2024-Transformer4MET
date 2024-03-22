import abc
from typing import Final
from tensordict.nn.common import NestedKey
from torch import nn
from torch import Tensor
from tensordict.nn import TensorDictModule


class Model(nn.Module, metaclass=abc.ABCMeta):
    IN_KEYS: Final[list[NestedKey]] = [
        'candidates',
        'candidates_pid',
        ('masks', 'candidates')
    ]
    OUT_KEYS: Final[list[NestedKey]] = [
        'rec_met'
    ]

    def __init__(self,
    ) -> None:
        super().__init__()

    @abc.abstractmethod
    def forward(self,
                candidates: Tensor,
                candidates_pid: Tensor,
                candidates_data_mask: Tensor,
    ) -> Tensor:
        """
        Args:
            candidates:
        """
        ...


    def to_tensor_dict_module(self):
        return TensorDictModule(
            module=self,
            in_keys=self.IN_KEYS, # type: ignore
            out_keys=self.OUT_KEYS, # type: ignore
        )
