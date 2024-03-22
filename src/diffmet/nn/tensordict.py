from torch.nn.modules import Module
from tensordict.nn import TensorDictModule
from tensordict.nn import TensorDictSequential


class TensorDictNamedSequential(TensorDictSequential):

    def __init__(
        self,
        module_dict: dict[str, TensorDictModule],
        partial_tolerant: bool = False,
    ) -> None:
        modules = list(module_dict.values())
        super().__init__(
            *modules,
            partial_tolerant=partial_tolerant,
        )

        self.module_dict = module_dict

    def get_module(self, name: str) -> Module:
        return self.module_dict[name].module
