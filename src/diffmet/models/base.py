import abc
from torch import nn
from tensordict.nn import TensorDictModule


class Model(nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 projection,
                 backbone,
                 regression_head,
    ) -> None:
        super().__init__()

        self.projection = projection
        self.backbone = backbone
        self.regression_head = regression_head

    @abc.abstractmethod
    def to_tensor_dict_module(self) -> TensorDictModule:
        ...
