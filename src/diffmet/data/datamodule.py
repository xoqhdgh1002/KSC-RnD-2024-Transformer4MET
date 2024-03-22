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

        dataset_cls = get_class(dataset_class_path)

        self.train_set = dataset_cls.from_root(train_files)
        self.val_set = dataset_cls.from_root(val_files)
        self.test_set = dataset_cls.from_root(test_files)

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

    def predict_dataloader(self):
        return self.test_dataloader()
