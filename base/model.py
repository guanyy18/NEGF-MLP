import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import global_add_pool
from torch_geometric.data import Data
from e3nn import o3
from typing import Union, Optional
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from module.interaction import MaceInteractionBlock, scatter_sum
from node import ConditionalAtomEmbed

class AtomicEnergiesBlock(nn.Module):
    def __init__(self, atomic_energies: Union[np.ndarray, torch.Tensor, None], num_atom_types: int = 100):
        super().__init__()

        if atomic_energies is None:

            energies = torch.zeros(num_atom_types, 1)
        else:
            energies = torch.tensor(atomic_energies, dtype=torch.get_default_dtype())


        if energies.dim() == 1:
            energies = energies.unsqueeze(-1)

        self.register_buffer("atomic_energies", energies)

    def forward(self, x: torch.Tensor) -> torch.Tensor:


        if x.dim() == 1 and x.dtype in [torch.long, torch.int]:


            return torch.nn.functional.embedding(x, self.atomic_energies).squeeze(-1)
        else:

            return torch.matmul(x, self.atomic_energies).squeeze(-1)

    def __repr__(self):
        formatted_energies = ", ".join(
            [
                "[" + ", ".join([f"{x:.4f}" for x in group]) + "]"
                for group in torch.atleast_2d(self.atomic_energies)
            ]
        )
        return f"{self.__class__.__name__}(energies=[{formatted_energies}])"

class ScaleShiftBlock(nn.Module):
    def __init__(self, scale: float, shift: float):
        super().__init__()
        self.register_buffer("scale", torch.tensor(scale))
        self.register_buffer("shift", torch.tensor(shift))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x + self.shift

    def __repr__(self):
        return f"{self.__class__.__name__}(scale={self.scale:.6f}, shift={self.shift:.6f})"


class FeatureNorm(nn.Module):
    def __init__(self, irreps_in):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.instructions = []
        start = 0
        self.num_output = 0
        for mul, ir in self.irreps_in:
            dim = mul * ir.dim
            self.instructions.append((start, start + dim, mul, ir.dim))
            start += dim
            self.num_output += mul

    def forward(self, x):
        norms_list = []
        for start, stop, mul, dim_ir in self.instructions:
            feature = x[:, start:stop]
            feature = feature.reshape(x.shape[0], mul, dim_ir)
            n = feature.pow(2).sum(dim=-1).add(1e-8).sqrt()
            norms_list.append(n)
        return torch.cat(norms_list, dim=-1)


class MaceReadout(nn.Module):
    def __init__(self, node_irreps: str, hidden_dim: int = 128, last_layer: bool = False):
        super().__init__()
        self.node_irreps = o3.Irreps(node_irreps)
        self.last_layer = last_layer

        self.scalars_irreps = o3.Irreps([(mul, ir) for mul, ir in self.node_irreps if ir.l == 0 and ir.p == 1]).simplify()

        if not last_layer:
            self.energy_projector = o3.Linear(self.node_irreps, o3.Irreps("1x0e"))
        else:
            self.scalar_extractor = o3.Linear(self.node_irreps, self.scalars_irreps)


            irreps_non_scalar = o3.Irreps([(mul, ir) for mul, ir in self.node_irreps if ir.l == 1 and ir.p == -1])
            if len(irreps_non_scalar) > 0:
                self.norm = FeatureNorm(irreps_non_scalar)
                num_scalar_channels = self.scalars_irreps.num_irreps
                num_norm_channels = self.norm.num_output
                mlp_input_dim = num_scalar_channels + num_norm_channels
            else:
                self.norm = None
                mlp_input_dim = self.scalars_irreps.num_irreps

            self.energy_projector = nn.Sequential(
                nn.Linear(mlp_input_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1)
            )

    def forward(self, x: Tensor) -> Tensor:
        if not self.last_layer:
            atomic_energy = self.energy_projector(x)
        else:
            scalars = self.scalar_extractor(x)
            if self.norm is not None:
                dim_scalar = self.scalars_irreps.dim
                x_non_scalar = x[..., dim_scalar:]
                norms = self.norm(x_non_scalar)
                scalars = torch.cat([scalars, norms], dim=-1)

            atomic_energy = self.energy_projector(scalars)

        return atomic_energy


class MaceModel(nn.Module):
    def __init__(self,
                 n_scalar: int = 128,
                 num_atom_types: int = 4,
                 num_interactions: int = 2,
                 correlation: int = 3,
                 feature_irreps_hidden: str = "128x0e + 128x1o + 128x2e",

                 radial_dim: int = 8,
                 radial_width: int = 64,
                 edge_attr_lmax: int = 2,
                 avg_num_neighbors: float = 15.0,
                 rbf_cutoff: float = 5.0,
                 lmax_center: int = 2,
                 lmax_env: int = 2,
                 atomic_energies_mean: Optional[torch.Tensor] = None,
                 atomic_inter_scale: float = 1.0,
                 atomic_inter_shift: float = 0.0,
                 ):
        super().__init__()


        self.embed = ConditionalAtomEmbed(
            n_scalar=n_scalar,
            num_atom_types=num_atom_types
        )
        self.irreps_embedding = self.embed.irreps_out
        current_irreps = self.irreps_embedding
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies_mean, num_atom_types)


        self.scale_shift = ScaleShiftBlock(scale=atomic_inter_scale, shift=atomic_inter_shift)


        self.interactions = nn.ModuleList()
        self.readouts = nn.ModuleList()
        self.hidden_irreps_obj = o3.Irreps(feature_irreps_hidden)

        for i in range(num_interactions):
            if i == 0:
                irreps_in = current_irreps
            else:
                irreps_in = self.hidden_irreps_obj

            irreps_out = self.hidden_irreps_obj
            node_attrs_irreps = o3.Irreps(f"{num_atom_types}x0e")
            edge_attrs_irreps = o3.Irreps.spherical_harmonics(edge_attr_lmax)
            edge_feats_irreps = o3.Irreps(f"{radial_dim}x0e")


            block = MaceInteractionBlock(
                num_elements=num_atom_types,
                node_attrs_irreps=node_attrs_irreps,
                node_feats_irreps=irreps_in,
                edge_attrs_irreps=edge_attrs_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=irreps_out,
                hidden_irreps=self.hidden_irreps_obj,
                edge_irreps=self.hidden_irreps_obj,
                correlation=correlation,
                radial_dim=radial_dim,
                radial_width=radial_width,
                radial_depth=2,
                edge_attr_lmax=edge_attr_lmax,
                avg_num_neighbors=avg_num_neighbors,
                use_gate=True,
                lmax_center=lmax_center,
                lmax_env=lmax_env,
                cutoff=rbf_cutoff
            )
            self.interactions.append(block)

            is_last = (i == num_interactions - 1)

            readout = MaceReadout(
                node_irreps=str(irreps_out),
                hidden_dim=128,
                last_layer=is_last
            )
            self.readouts.append(readout)

    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:

        data.pos.requires_grad_(True)


        edge_index = data.edge_index
        diff = data.pos[edge_index[1]] - data.pos[edge_index[0]]
        if hasattr(data, 'edge_shift'):
            diff = diff + data.edge_shift
        data.edge_vec = diff


        data.x = self.embed(data.atom_type, data.vbias, data.is_center)


        node_e0 = self.atomic_energies_fn(data.atom_type)


        node_inter_es = torch.zeros_like(node_e0)


        for i, (interaction, readout) in enumerate(zip(self.interactions, self.readouts)):

            data = interaction(data)

            delta_e = readout(data.x)


            node_inter_es = node_inter_es + delta_e.squeeze(-1)


        node_inter_es = self.scale_shift(node_inter_es)


        node_energies = node_e0 + node_inter_es


        total_energy = global_add_pool(node_energies, data.batch)
        forces = -torch.autograd.grad(
            outputs=total_energy.sum(),
            inputs=data.pos,
            create_graph=True
        )[0]

        return total_energy, forces
