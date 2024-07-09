from typing import Final
from lightning import LightningModule
import torch
from torch import Tensor
from torch import nn
from torch.nn.modules import ModuleDict
from tensordict import TensorDict
from torchmetrics import MetricCollection
from .data.transforms.compose import Compose
from .optim import configure_optimizers
from .metrics import Bias, Resolution
from .utils.math import rectify_phi, to_polar
from .models.base import Model


DEFAULT_PT_BINNING: Final[list[tuple[float, float]]] = [
    (0, 45),
    (45, 70),
    (70, 100),
    (100, float('inf')),
]

class LitModel(LightningModule):

    def __init__(self,
                 augmentation: Compose,
                 preprocessing: Compose,
                 model: Model,
                 criterion: nn.Module = nn.MSELoss(),
                 pt_binning: list[tuple[float, float]] = DEFAULT_PT_BINNING,
    ) -> None:
        super().__init__()
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.model = model.to_tensor_dict_module()
        self.criterion = criterion

        self.pt_binning = pt_binning
        self.val_metrics = self.build_metrics(self.pt_binning, 'val')
        self.test_metrics = self.build_metrics(self.pt_binning, 'test')

    def build_metrics(self,
                      pt_bins: list[tuple[float, float]],
                      stage: str,
    ) -> ModuleDict:
        """

        >>> eval_metrics["pt-0-45"]["pt"]
        """
        # FIXME gen met pt binning
        metrics = MetricCollection({
            'bias': Bias(),
            'res': Resolution(),
        })

        eval_metrics = ModuleDict()
        for low, up in pt_bins:
            pt_key = f'pt-{low:.0f}-{up:.0f}'

            eval_metrics[pt_key] = ModuleDict({
                key: metrics.clone(f'{stage}_{pt_key}_{key}_')
                for key in ['px', 'py', 'pt', 'phi']
            })
        return eval_metrics

    def training_step(self, # type: ignore
                      input: TensorDict,
    ) -> Tensor:
        output = self.augmentation(input)
        output = self.preprocessing(output)
        output = self.model(output)
        loss = self.criterion(input=output['rec_met'], target=output['gen_met'])
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def _eval_step(self, # type: ignore
                  input: TensorDict,
                  metrics: ModuleDict,
                  stage: str,
    ) -> None:
        # eval step doesn't require data augmentation
        output = self.preprocessing(input)
        output = self.model(output)
        loss = self.criterion(input=output['rec_met'], target=output['gen_met'])

        gen_met = output['gen_met']
        rec_met = output['rec_met']
        if 'gen_met_norm' in self.preprocessing.keys():
            gen_met_norm = self.preprocessing['gen_met_norm']
            # undo normalisation
            gen_met: Tensor = gen_met_norm.inverse(gen_met) # type: ignore
            rec_met: Tensor = gen_met_norm.inverse(rec_met) # type: ignore

        # (px, py) to (pt, phi)
        gen_met_polar = to_polar(gen_met)
        rec_met_polar = to_polar(rec_met)

        gen_met_pt = gen_met_polar[:, 0]

        residual = rec_met - gen_met
        residual_polar = rec_met_polar - gen_met_polar
        residual_polar[:, 1] = rectify_phi(residual_polar[:, 1])

        for low, up in self.pt_binning:
            pt_key = f'pt-{low:.0f}-{up:.0f}'

            pt_mask = torch.logical_and(gen_met_pt > low, gen_met_pt < up)

            masked_residual = residual[pt_mask]
            masked_residual_polar = residual_polar[pt_mask]

            metrics[pt_key]['px'].update(masked_residual[:, 0]) # type: ignore
            metrics[pt_key]['py'].update(masked_residual[:, 1]) # type: ignore
            metrics[pt_key]['pt'].update(masked_residual_polar[:, 0]) # type: ignore
            metrics[pt_key]['phi'].update(masked_residual_polar[:, 1]) # type: ignore

        self.log(f'{stage}_loss', loss, prog_bar=True)

        return output

    def _on_eval_epoch_end(self, metrics: ModuleDict):
        log_dict = {}
        for component_dict in metrics.values():
            for each in component_dict.values():
                log_dict |= each.compute() # type: ignore
        self.log_dict(log_dict, prog_bar=True)


    def validation_step(self, # type: ignore
                        input: TensorDict,
    ) -> None:
        return self._eval_step(input=input, metrics=self.val_metrics,
                               stage='val')

    def on_validation_epoch_end(self):
        return self._on_eval_epoch_end(metrics=self.val_metrics)

    def test_step(self, # type: ignore
                  input: TensorDict,
    ) -> None:
        return self._eval_step(input=input, metrics=self.test_metrics,
                               stage='test')

    def on_test_epoch_end(self):
        return self._on_eval_epoch_end(metrics=self.test_metrics)

    def predict_step(self, input): # type: ignore[override]
        output = self.preprocessing(input)
        output = self.model(output)
        rec_met = output['rec_met']
        if 'gen_met_norm' in self.preprocessing.keys():
            rec_met: Tensor = self.preprocessing['gen_met_norm'].inverse(rec_met) # type: ignore
        return rec_met
