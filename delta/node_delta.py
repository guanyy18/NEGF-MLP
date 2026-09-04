import torch
import torch.nn as nn
from torch import Tensor
from e3nn import o3

class ConditionalAtomEmbed(nn.Module):
    """
    Uniform embedding shared by all atoms:
    Z -> n*0e  (+)  vbias -> 1*1o  (direct sum)
    Output irreps: n*0e + 1*1o  (n = n_scalar)
    """
    def __init__(self, n_scalar: int = 16, num_atom_types: int = 100):
        super().__init__()
        self.n_scalar = n_scalar
        self.irreps_out = o3.Irreps(f"{n_scalar}x0e + 1x1o")


        self.embed_z = nn.Embedding(num_atom_types, n_scalar)
        self.proj_v   = nn.Linear(3, 3)

    def forward(self, Z: Tensor, vbias: Tensor) -> Tensor:
        """
        Z:         [N]  LongTensor
        vbias:     [N, 3]  bias vector; zeros are fine when no bias is applied
        """

        h_scalar =  self.embed_z(Z)
        h_vector = self.proj_v(vbias)


        out = torch.cat([h_scalar, h_vector], dim=1)
        return out
