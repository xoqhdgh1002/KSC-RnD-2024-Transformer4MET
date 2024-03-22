from typing import Final, Type, Any
import uproot.writing
import numpy as np
from torch import Tensor
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
from .lit import LitModel


class UprootWriter(BasePredictionWriter):

    BRANCH_NAMES: Final[list[str]] = [
        'gen_met_px',
        'gen_met_py',
        'rec_met_px',
        'rec_met_py',
    ]

    BRANCH_TYPES: Final[dict[str, Type]] = {each: np.float32
                                            for each in BRANCH_NAMES}

    def __init__(self,
                 output_path,
    ):
        super().__init__(write_interval='batch')
        self.file = uproot.writing.create(output_path)
        self.tree = self.file.mktree(name='tree',
                                     branch_types=self.BRANCH_TYPES)

    def __del__(self):
        if hasattr(self, 'file'):
            self.file.close()

    def write_on_batch_end(self, # type: ignore[override]
                           trainer: Trainer,
                           pl_module: LitModel,
                           prediction: Any,
                           batch_indices: list[int],
                           batch: Any,
                           batch_idx: int,
                           dataloader_idx: int,
    ):
        rec_met = prediction.detach().cpu().numpy()

        batch = pl_module.preprocessing.inverse(batch)
        gen_met = batch['gen_met'].cpu().numpy()

        data = {
            'rec_met_px': rec_met[:, 0],
            'rec_met_py': rec_met[:, 1],
            'gen_met_px': gen_met[:, 0],
            'gen_met_py': gen_met[:, 1],
        }

        self.tree.extend(data)
