from torch import Tensor
from torch import nn
from ..nn.candidate import AdditiveCandidateEmbedding
from ..nn.transformer import TransformerEncoder
from ..nn.aggregator import MaskedMean
from .base import Model


class TransformerBackbone(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 activation: str,
                 widening_factor: int,
                 dropout_p: float,
                 num_layers: int
    ):
        super().__init__()

        self.encoder = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            activation=activation,
            widening_factor=widening_factor,
            dropout_p=dropout_p,
            num_layers=num_layers
        )

        self.aggregator = MaskedMean(dim=1)

    def forward(self,
                input: Tensor,
                data_mask: Tensor,
    ) -> Tensor:
        """
        Args:
            input:
            data_mask:
        Returns:
            a tensor of the shape of (N, L, E)
        """
        output = self.encoder(input=input, data_mask=data_mask)
        output = self.aggregator(input=output, data_mask=data_mask)
        return output


class Transformer(Model):

    def __init__(self,
                 embed_dim: int = 32,
                 num_heads: int = 2,
                 activation: str = 'ReLU',
                 widening_factor: int = 4,
                 dropout_p: float = 0.0,
                 num_layers: int = 2
    ) -> None:
        super().__init__()

        self.cand_proj = AdditiveCandidateEmbedding(
            input_dim=3,
            num_pids=6,
            embed_dim=embed_dim
        )

        self.backbone = TransformerBackbone(
            embed_dim=embed_dim,
            num_heads=num_heads,
            activation=activation,
            widening_factor=widening_factor,
            dropout_p=dropout_p,
            num_layers=num_layers
        )

        self.regression_head = nn.Linear(
            in_features=embed_dim,
            out_features=2
        )

    def forward(self,
                candidates: Tensor,
                candidates_pid: Tensor,
                candidates_data_mask: Tensor,
    ) -> Tensor:
        x = self.cand_proj(candidates, candidates_pid, candidates_data_mask)
        z = self.backbone(x, candidates_data_mask)
        y = self.regression_head(z)
        return y
