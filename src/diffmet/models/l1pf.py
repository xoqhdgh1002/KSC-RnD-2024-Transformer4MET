from tensordict.nn.common import NestedKey
from torch import Tensor
from tensordict.nn import TensorDictModule
from .base import Model


class L1PFModel(Model):

    def forward(self,
                candidates: Tensor,
                candidates_pid: Tensor,
                candidates_data_mask: Tensor,
    ) -> Tensor:
        x = self.projection(
            candidates,
            candidates_pid,
            candidates_data_mask
        )
        z = self.backbone(x, candidates_data_mask)
        y = self.regression_head(z)
        return y


    def to_tensor_dict_module(self):
        in_keys: list[NestedKey] = [
            'candidates',
            'candidates_pid',
            ('masks', 'candidates')
        ]

        out_keys: list[NestedKey] = [
            'rec_met'
        ]

        return TensorDictModule(
            module=self,
            in_keys=in_keys,
            out_keys=out_keys,
        )
