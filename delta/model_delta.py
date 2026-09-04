import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import global_add_pool
from torch_geometric.data import Data
from torch_scatter import scatter
from e3nn import o3
from typing import Union, Optional
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from module.interaction import MaceInteractionBlock

from node_delta import ConditionalAtomEmbed


def solve_and_compute_E_z(pos: Tensor, edge_index: Tensor, batch: Optional[Tensor] = None, ptr: Optional[Tensor] = None, num_graphs: Optional[int] = None, sgn_v: float = 1.0, num_electrode: int = 52, atom_type: Optional[Tensor] = None, electrolyte_types: tuple = (1, 2)) -> Tensor:
    num_nodes = pos.shape[0]


    if batch is None:
        batch = torch.zeros(num_nodes, dtype=torch.long, device=pos.device)
    if ptr is None:
        ptr = torch.tensor([0, num_nodes], device=pos.device)
    if num_graphs is None:
        num_graphs = 1

    orig_dtype = pos.dtype
    device_type = pos.device.type


    with torch.amp.autocast(device_type=device_type, enabled=False):
        pos_fp32 = pos.float()


        src, dst = edge_index[0], edge_index[1]
        r_ij = torch.norm(pos_fp32[src] - pos_fp32[dst], dim=-1)
        weights = 1.0 / (r_ij + 1e-8)

        A = torch.zeros((num_nodes, num_nodes), device=pos.device, dtype=torch.float32)
        A[src, dst] = weights
        D = torch.diag(A.sum(dim=1))
        L = D - A


        boundary_mask = torch.zeros(num_nodes, dtype=torch.bool, device=pos.device)
        Phi_B_list = []

        p_or_s_mask = torch.isin(atom_type, torch.tensor(electrolyte_types, device=pos.device)) if atom_type is not None else torch.zeros(num_nodes, dtype=torch.bool, device=pos.device)

        for g in range(num_graphs):
            start = ptr[g].item()
            end = ptr[g+1].item()

            pos_g = pos_fp32[start:end]
            p_or_s_g = p_or_s_mask[start:end]

            if p_or_s_g.any():
                z_g = pos_g[:, 2]
                z_electrolyte_min = z_g[p_or_s_g].min()
                z_electrolyte_max = z_g[p_or_s_g].max()


                left_metal_mask_g = z_g < z_electrolyte_min

                right_metal_mask_g = z_g > z_electrolyte_max

                boundary_mask[start:end] = left_metal_mask_g | right_metal_mask_g

                num_left_metal = left_metal_mask_g.sum().item()
                num_right_metal = right_metal_mask_g.sum().item()
                phi_b_g = torch.zeros((num_left_metal + num_right_metal, 1), device=pos.device, dtype=torch.float32)
                phi_b_g[:num_left_metal] = 1.0 * sgn_v
                phi_b_g[num_left_metal:] = -1.0 * sgn_v
                Phi_B_list.append(phi_b_g)
            else:
                boundary_mask[start : start + num_electrode] = True
                boundary_mask[end - num_electrode : end] = True
                phi_b_g = torch.zeros((2 * num_electrode, 1), device=pos.device, dtype=torch.float32)
                phi_b_g[:num_electrode] = 1.0 * sgn_v
                phi_b_g[num_electrode:] = -1.0 * sgn_v
                Phi_B_list.append(phi_b_g)

        Phi_B = torch.cat(Phi_B_list, dim=0)

        interior_indices = torch.where(~boundary_mask)[0]
        boundary_indices = torch.where(boundary_mask)[0]


        L_II = L[interior_indices][:, interior_indices]
        L_IB = L[interior_indices][:, boundary_indices]
        L_II_reg = L_II + torch.eye(L_II.shape[0], device=pos.device, dtype=torch.float32) * 1e-6
        Phi_I = torch.linalg.solve(L_II_reg, -L_IB @ Phi_B)


        Phi_full = torch.zeros((num_nodes, 1), device=pos.device, dtype=torch.float32)
        Phi_full[boundary_indices] = Phi_B
        Phi_full[interior_indices] = Phi_I


        diff_z = pos_fp32[dst, 2] - pos_fp32[src, 2]
        d_phi = Phi_full[dst].squeeze(-1) - Phi_full[src].squeeze(-1)

        edge_grad_z = d_phi * diff_z / (r_ij.pow(2) + 1e-8)
        edge_weight = 1.0 / (r_ij + 1e-8)

        sum_weights = scatter(edge_weight, dst, dim=0, dim_size=num_nodes)
        sum_grads = scatter(edge_grad_z, dst, dim=0, dim_size=num_nodes)

        grad_z = sum_grads / (sum_weights + 1e-8)
        E_z = -grad_z


        new_vbias = torch.zeros((num_nodes, 3), device=pos.device, dtype=torch.float32)
        new_vbias[:, 2] = E_z

    return new_vbias.to(dtype=orig_dtype)


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
    def __init__(self, node_irreps: str, hidden_dim: int = 64, last_layer: bool = False):
        super().__init__()
        self.node_irreps = o3.Irreps(node_irreps)
        self.last_layer = last_layer
        self.scalars_irreps = o3.Irreps([(mul, ir) for mul, ir in self.node_irreps if ir.l == 0 and ir.p == 1]).simplify()
        self.scalar_extractor = o3.Linear(self.node_irreps, self.scalars_irreps)
        irreps_non_scalar = o3.Irreps([(mul, ir) for mul, ir in self.node_irreps if not (ir.l == 0 and ir.p == 1)])

        if len(irreps_non_scalar) > 0:
            self.norm = FeatureNorm(irreps_non_scalar)
            mlp_input_dim = self.scalars_irreps.num_irreps + self.norm.num_output
        else:
            self.norm = None
            mlp_input_dim = self.scalars_irreps.num_irreps

        self.energy_projector = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.polarization_projector = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: Tensor, v_scalar: Optional[Tensor] = None) -> tuple[Tensor, Optional[Tensor]]:
        scalars = self.scalar_extractor(x)
        features_to_concat = [scalars]

        if self.norm is not None:
            dim_scalar = self.scalars_irreps.dim
            x_non_scalar = x[..., dim_scalar:]
            norms = self.norm(x_non_scalar)
            features_to_concat.append(norms)

        combined_features = torch.cat(features_to_concat, dim=-1)
        if v_scalar is not None:
            combined_features = combined_features * v_scalar

        base_energy = self.energy_projector(combined_features)
        pol_response = self.polarization_projector(combined_features)
        return base_energy, pol_response


class MaceModel(nn.Module):
    def __init__(self,
                 n_scalar: int = 32,
                 num_atom_types: int = 3,
                 num_interactions: int = 2,
                 correlation: int = 3,
                 feature_irreps_hidden: str = "128x0e + 128x1o + 128x2e",
                 radial_dim: int = 8,
                 radial_width: int = 64,
                 edge_attr_lmax: int = 2,
                 avg_num_neighbors: float = 15.0,
                 rbf_cutoff: float = 5.0,
                 lmax_center: int = 2,
                 lmax_env: int = 1,
                 is_discharge: bool = False,
                 num_electrode: int = 52,
                 electrolyte_types: tuple = (1, 2)
                 ):
        super().__init__()
        self.is_discharge = is_discharge
        self.num_electrode = num_electrode
        self.electrolyte_types = electrolyte_types

        self.embed = ConditionalAtomEmbed(
            n_scalar=n_scalar,
            num_atom_types=num_atom_types
        )
        self.irreps_embedding = self.embed.irreps_out
        current_irreps = self.irreps_embedding

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

        batch = data.batch if hasattr(data, 'batch') else None
        ptr = data.ptr if hasattr(data, 'ptr') else None
        num_graphs = data.num_graphs if hasattr(data, 'num_graphs') else None

        if hasattr(data, 'voltage'):
            v_per_atom = data.voltage[batch] if batch is not None else data.voltage
        elif hasattr(data, 'vbias_scalar'):
            v_per_atom = data.vbias_scalar[batch] if batch is not None else data.vbias_scalar
        else:
            num_nodes = data.pos.shape[0]
            v_per_atom = torch.ones((num_nodes, 1), device=data.pos.device, dtype=data.pos.dtype)


        v_val_first = v_per_atom[0].item()
        sgn_v = 1.0 if v_val_first >= 0.0 else -1.0
        v_per_atom_abs = torch.abs(v_per_atom)

        num_elec_attr = self.num_electrode if hasattr(self, 'num_electrode') else 52


        vbias_dynamic = solve_and_compute_E_z(
            data.pos,
            data.edge_index,
            batch,
            ptr,
            num_graphs,
            sgn_v=sgn_v,
            num_electrode=num_elec_attr,
            atom_type=data.atom_type,
            electrolyte_types=self.electrolyte_types
        )
        data.vbias = vbias_dynamic.detach()

        data.x = self.embed(data.atom_type, data.vbias)

        num_atoms = data.x.shape[0]
        node_total_energies = torch.zeros(num_atoms, device=data.x.device, dtype=data.x.dtype)

        for i, (interaction, readout) in enumerate(zip(self.interactions, self.readouts)):
            data = interaction(data)
            delta_e, _ = readout(data.x, v_scalar=v_per_atom_abs)
            node_total_energies = node_total_energies + delta_e.squeeze(-1)

        total_energy = global_add_pool(node_total_energies, batch if batch is not None else torch.zeros(num_atoms, dtype=torch.long, device=data.pos.device))


        forces = -torch.autograd.grad(
            outputs=total_energy.sum(),
            inputs=data.pos,
            create_graph=self.training
        )[0]

        return total_energy, forces
