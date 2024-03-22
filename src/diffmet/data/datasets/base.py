import abc
import tensordict as td
from tensordict import TensorDict
from torch.utils.data import Dataset


class TensorDictListDataset(Dataset, metaclass=abc.ABCMeta):

    def __init__(self, example_list: list[TensorDict]) -> None:
        self.example_list = example_list

    def __len__(self) -> int:
        return len(self.example_list)

    def __getitem__(self, index) -> TensorDict:
        return self.example_list[index]

    def __add__(self, # type: ignore[override]
                other: 'TensorDictListDataset'
    ) -> 'TensorDictListDataset':
        example_list = self.example_list + other.example_list
        return self.__class__(example_list)

    def collate(self, batch: list[TensorDict]) -> TensorDict:
        return td.pad_sequence(
            list_of_tensordicts=batch,
            batch_first=True,
            pad_dim=0,
            padding_value=0,
            return_mask=True
        )
