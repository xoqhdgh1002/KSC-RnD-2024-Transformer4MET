from typing import Any
from tensordict.tensordict import TensorDict
import torch
from torch.nn.modules import Module, ModuleDict
from .base import TensorDictTransform, Transform
from .normalize import Normalize


class Compose(Module):

    def __init__(self, data_list: list[dict[str, Any]]):
        super().__init__()
        module_dict: dict[str, TensorDictTransform] = {}
        for data in data_list:
            if data['module'] == 'Normalize':
                module = Normalize.from_data(data)
            else:
                raise NotImplementedError(f'unknown: {data["module"]}')

            module_dict[data['name']] = TensorDictTransform(
                module=module,
                in_keys=data['in_keys'],
                out_keys=data['out_keys'],
            )
        self.transform_dict = ModuleDict(module_dict)


    def __getitem__(self, key: str) -> Transform:
        return self.transform_dict[key].module

    @property
    def transform_list(self) -> list[TensorDictTransform]:
        return list(self.transform_dict.values()) # type: ignore

    @property
    def inverse_transform_list(self) -> list[TensorDictTransform]:
        return list(reversed(self.transform_dict.values())) # type: ignore

    @torch.no_grad()
    def forward(self, input: TensorDict) -> TensorDict:
        output = input
        for transform in self.transform_list:
            output = transform(output)
        return output

    @torch.no_grad()
    def inverse(self, input: TensorDict) -> TensorDict:
        output = input
        for transform in self.inverse_transform_list:
            output = transform.inverse(output) # type: ignore
        return output
    #
    # @classmethod
    # def from_data(cls, data_list: list[dict[str, Any]]) -> 'Compose':
    #     module_dict: dict[str, TensorDictTransform] = {}
    #     for data in data_list:
    #         if data['module'] == 'Normalize':
    #             module = Normalize.from_data(data)
    #         else:
    #             raise NotImplementedError(f'unknown: {data["module"]}')
    #
    #         module_dict[data['name']] = TensorDictTransform(
    #             module=module,
    #             in_keys=data['in_keys'],
    #             out_keys=data['out_keys'],
    #         )
    #     return cls(module_dict)
    #
    # @classmethod
    # def from_yaml(cls, path: str) -> 'Compose':
    #     with open(path) as stream:
    #         data_list: list[dict[str, Any]] = yaml.safe_load(stream)
    #     return cls.from_data(data_list)
