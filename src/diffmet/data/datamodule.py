from functools import cached_property
from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule
from ..utils.lit import get_class


class DataModule(LightningDataModule):
    def __init__(self,
                 dataset_class_path: str,
                 train_files: list[str],
                 val_files: list[str],
                 test_files: list[str],
                 batch_size: int = 256,
                 eval_batch_size: int = 512,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size

        self.train_files = train_files
        self.val_files = val_files
        self.test_files = test_files

        self.dataset_cls = get_class(dataset_class_path)

    @cached_property
    def train_set(self):
        return self.dataset_cls.from_root(self.train_files)

    @cached_property
    def val_set(self):
        return self.dataset_cls.from_root(self.val_files)

    @cached_property
    def test_set(self):
        return self.dataset_cls.from_root(self.test_files)

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            self.train_set
            self.val_set
        elif stage == 'validate':
            self.val_set
        elif stage == 'test':
            self.test_set
        else:
            raise ValueError

    def teardown(self, stage: str) -> None:
        if stage == 'fit':
            delattr(self, 'train_set')
            delattr(self, 'val_set')
        elif stage == 'test':
            delattr(self, 'test_set')


    def train_dataloader(self):
        dataset = self.train_set
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=dataset.collate,
            drop_last=True,
        )

    def _eval_dataloader(self, dataset):
        return DataLoader(
            dataset=dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            collate_fn=dataset.collate,
            drop_last=False,
        )

    def val_dataloader(self):
        return self._eval_dataloader(self.val_set)

    def test_dataloader(self):
        return self._eval_dataloader(self.test_set)
