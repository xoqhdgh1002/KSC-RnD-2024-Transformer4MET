from torch import Tensor
import torch.nn as nn
from .attention import SelfAttentionModule
from .aggregator import MaskedMean


class TransformerEncoder(nn.Module):
    latent: nn.Parameter

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 activation: str,
                 widening_factor: int,
                 dropout_p: float,
                 num_layers: int
    ) -> None:
        """
        """
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = SelfAttentionModule(
                embed_dim=embed_dim,
                num_heads=num_heads,
                activation=activation,
                widening_factor=widening_factor,
                dropout_p=dropout_p
            )
            self.layers.append(layer)

    def forward(self,
                input: Tensor,
                data_mask: Tensor | None,
    ) -> Tensor:
        """
        """
        output = input
        for layer in self.layers:
            output = layer(input=output, data_mask=data_mask)
        return output


class Transformer(nn.Module):

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

