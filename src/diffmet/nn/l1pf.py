import abc
from torch import nn
from torch import Tensor



class CandidateEmbedding(nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 num_pids: int,
                 pid_embedding_dim: int,
    ):
        super().__init__()

        self.cont_embed = nn.Linear(
            in_features=in_features,
            out_features=out_features,
        )
        self.pid_embed = nn.Embedding(
            num_embeddings=(num_pids + 1),
            embedding_dim=pid_embedding_dim,
            padding_idx=0
        )


    @abc.abstractmethod
    def forward(self,
                input: Tensor,
                pid: Tensor,
                data_mask: Tensor
    ) -> Tensor:
        ...


class AdditiveCandidateEmbedding(CandidateEmbedding):

    def __init__(self,
                 input_dim: int,
                 num_pids: int,
                 embed_dim: int
    ) -> None:
        """
        Args:
            input_dim: the number of continuous features of a candidate
            pid_num_embeddings: the number of PDGIDs for candidates
            embed_dim:
        """

        super().__init__(
            in_features=input_dim,
            out_features=embed_dim,
            num_pids=num_pids,
            pid_embedding_dim=embed_dim,
        )

    def forward(self,
                input: Tensor,
                pid: Tensor,
                data_mask: Tensor
    ) -> Tensor:
        """
        Args:
            input: continuous features of candidates
            pid: a long tensor with the shape of (N, L)
            data_mask:
        Returns:
            a tensor
        """
        x = self.cont_embed(input)
        pad_mask = data_mask.logical_not().unsqueeze(-1)
        x.masked_fill_(pad_mask, 0)

        x_pid = self.pid_embed(pid)

        x = x + x_pid

        return x
