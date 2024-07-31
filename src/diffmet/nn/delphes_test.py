import torch
from torch import Tensor
from torch import nn

class PFMerger(nn.Module):

    def __init__(self,
                 track_dim: int,
                 embed_dim: int,
    ) -> None:
        super().__init__()

        self.track_projection = nn.Linear(track_dim, embed_dim)


    def forward(self,
                track: Tensor,
                track_data_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        z_track = self.track_projection(track)

        # concatenate two tensors along
        z = torch.cat([z_track], dim=1)
        data_mask = torch.cat([track_data_mask], dim=1)

        z.masked_fill_(
            mask=data_mask.logical_not().unsqueeze(-1),
            value=0
        )

        return z, data_mask
