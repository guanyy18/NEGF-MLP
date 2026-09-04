import torch
import torch.nn as nn
from torch import Tensor
from e3nn import o3

class ConditionalAtomEmbed(nn.Module):
    """
    Selects the embedding path based on is_center:
    Center atoms:   Z -> n*0e
    Environment:    Z -> n*0e  (+)  vbias -> 1*1o  (direct sum)
    Output irreps: n*0e + 1*1o  (n = n_scalar)
    """
    def __init__(self, n_scalar: int = 16, num_atom_types: int = 100):
        super().__init__()
        self.n_scalar = n_scalar
        self.irreps_out = o3.Irreps(f"{n_scalar}x0e + 1x1o")


        self.center_embed = nn.Embedding(num_atom_types, n_scalar)


        self.env_embed_z = nn.Embedding(num_atom_types, n_scalar)
        self.env_proj_v   = nn.Linear(3, 3)

    def forward(self, Z: Tensor, vbias: Tensor, is_center: Tensor) -> Tensor:
        """
        Z:         [N]  LongTensor
        vbias:     [N, 3]  vbias is given for environment atoms; zeros are fine for center atoms
        is_center: [N]  BoolTensor
        """

        h_center = self.center_embed(Z)


        h_env_z = self.env_embed_z(Z)
        h_env_v = self.env_proj_v(vbias)


        mask_c = is_center.unsqueeze(1)
        mask_e = ~mask_c

        h_scalar = torch.where(mask_c, h_center, h_env_z)

        h_vector = torch.where(mask_c, torch.zeros_like(h_env_v), h_env_v)


        out = torch.cat([h_scalar, h_vector], dim=1)
        return out
