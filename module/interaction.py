import torch
import torch.nn as nn
from torch import Tensor
from torch_scatter import scatter
from math import sqrt, pi
import e3nn
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct, TensorProduct
from e3nn.nn import Gate
from torch_geometric.data import Data
from typing import Any, Callable, List, Optional, Tuple, Union

from mace.modules.symmetric_contraction import SymmetricContraction
from typing import Any, Callable, Dict, List, Optional, Type, Union


def mask_irreps(x: Tensor, irreps: o3.Irreps, is_center: Tensor, lmax: dict):
    start = 0
    output = x.clone()
    for mul, (l, p) in irreps:
        end = start + mul * (2 * l + 1)
        if l > lmax["center"]:
            output[:, start:end] *= (~is_center).unsqueeze(1)
        if l > lmax["env"]:
            output[:, start:end] *= is_center.unsqueeze(1)
        start = end
    return output

def _broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src

def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> torch.Tensor:
    assert reduce == "sum"
    index = _broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


def tp_out_irreps_with_instructions(
    irreps1: o3.Irreps, irreps2: o3.Irreps, target_irreps: o3.Irreps
) -> Tuple[o3.Irreps, List]:
    trainable = True


    irreps_out_list: List[Tuple[int, o3.Irreps]] = []
    instructions = []
    for i, (mul, ir_in) in enumerate(irreps1):
        for j, (_, ir_edge) in enumerate(irreps2):
            for ir_out in ir_in * ir_edge:
                if ir_out in target_irreps:
                    k = len(irreps_out_list)
                    irreps_out_list.append((mul, ir_out))
                    instructions.append((i, j, k, "uvu", trainable))


    irreps_out = o3.Irreps(irreps_out_list)
    irreps_out, permut, _ = irreps_out.sort()


    instructions = [
        (i_in1, i_in2, permut[i_out], mode, train)
        for i_in1, i_in2, i_out, mode, train in instructions
    ]

    instructions = sorted(instructions, key=lambda x: x[2])

    return irreps_out, instructions

class reshape_irreps(torch.nn.Module):
    def __init__(
        self, irreps: o3.Irreps
    ) -> None:
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.dims = []
        self.muls = []
        for mul, ir in self.irreps:
            d = ir.dim
            self.dims.append(d)
            self.muls.append(mul)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        ix = 0
        out = []
        batch, _ = tensor.shape
        for mul, d in zip(self.muls, self.dims):
            field = tensor[:, ix : ix + mul * d]
            ix += mul * d
            if hasattr(self, "cueq_config"):
                if self.cueq_config is not None:
                    if self.cueq_config.layout_str == "mul_ir":
                        field = field.reshape(batch, mul, d)
                    else:
                        field = field.reshape(batch, d, mul)
                else:
                    field = field.reshape(batch, mul, d)
            else:
                field = field.reshape(batch, mul, d)
            out.append(field)
        return torch.cat(out, dim=-1)


class Linear:
    """Returns either a cuet.Linear or o3.Linear based on config"""

    def __new__(
        cls,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        shared_weights: bool = True,
        internal_weights: bool = True,
    ):
        return o3.Linear(
            irreps_in,
            irreps_out,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
        )

class PolynomialCutoff(nn.Module):
    def __init__(self, p: float = 6.0):
        super().__init__()
        assert p >= 2.0
        self.p = p
    def forward(self, x: Tensor) -> Tensor:
        c1 = (self.p + 1.0) * (self.p + 2.0) / 2.0
        c2 = self.p * (self.p + 2.0)
        c3 = self.p * (self.p + 1.0) / 2.0
        x_p = x.pow(self.p); x_p1 = x_p * x; x_p2 = x_p1 * x
        return (1.0 - c1 * x_p + c2 * x_p1 - c3 * x_p2) * (x < 1.0).to(x.dtype)

class Envelope(torch.nn.Module):
    def __init__(self, exponent: int):
        super().__init__()
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = int(self.p)
        a, b, c = self.a, self.b, self.c


        x_pow_p0 = x
        for _ in range(p - 2):
            x_pow_p0 = x_pow_p0 * x

        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x

        return (1.0 / x + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p2) * (x < 1.0).to(x.dtype)

class BesselRBF1(torch.nn.Module):
    def __init__(self, num_radial: int, cutoff: float = 5.0, envelope_exponent: int = 5):
        super().__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)
        self.freq = torch.nn.Parameter(torch.empty(num_radial))
        self.reset_parameters()
    def reset_parameters(self):
        with torch.no_grad():
            torch.arange(1, self.freq.numel() + 1, out=self.freq).mul_(pi)
        self.freq.requires_grad_()
    def forward(self, dist: Tensor) -> Tensor:
        if dist.dim() == 1: dist = dist.unsqueeze(-1)
        dist_norm = dist / self.cutoff
        return self.envelope(dist_norm) * (self.freq * dist_norm).sin()

class SphericalHarmonicsEdgeAttrs(torch.nn.Module):
    def __init__(self, irreps_out: o3.Irreps):
        super().__init__()
        self.irreps_out = irreps_out
    def forward(self, edge_vec: Tensor) -> Tensor:
        return o3.spherical_harmonics(self.irreps_out, edge_vec, normalize=True, normalization='component')


class EquivariantProductBasisBlock(nn.Module):
    def __init__(
        self,
        node_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        correlation: int,
        num_elements: int,
        use_sc: bool = True,
        use_agnostic_product: bool = True
    ):
        super().__init__()
        self.use_sc = use_sc
        self.use_agnostic_product = use_agnostic_product


        self.num_contraction_elements = 1 if use_agnostic_product else num_elements


        self.symmetric_contractions = SymmetricContraction(
            irreps_in=node_feats_irreps,
            irreps_out=target_irreps,
            correlation=correlation,
            num_elements=self.num_contraction_elements,
            internal_weights=True,
            shared_weights=True,
        )


        self.linear = o3.Linear(target_irreps, target_irreps)

    def forward(
        self,
        node_feats: torch.Tensor,
        sc: torch.Tensor,
        node_attrs: torch.Tensor,
    ) -> torch.Tensor:


        if self.use_agnostic_product:


            batch_size = node_feats.shape[0]
            contraction_attrs = torch.ones(
                (batch_size, 1),
                dtype=node_feats.dtype,
                device=node_feats.device,
            )
        else:

            contraction_attrs = node_attrs


        out = self.symmetric_contractions(node_feats, contraction_attrs)


        out = self.linear(out)


        if self.use_sc and sc is not None:
            out = out + sc

        return out


class RealAgnosticResidualInteractionBlock(nn.Module):
    def __init__(
        self,
        node_attrs_irreps: o3.Irreps,
        node_feats_irreps: o3.Irreps,
        edge_attrs_irreps: o3.Irreps,
        edge_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        hidden_irreps: o3.Irreps,
        avg_num_neighbors: float,
        edge_irreps: Optional[o3.Irreps] = None,
        radial_MLP: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.node_attrs_irreps = node_attrs_irreps
        self.node_feats_irreps = node_feats_irreps
        self.edge_attrs_irreps = edge_attrs_irreps
        self.edge_feats_irreps = edge_feats_irreps
        self.target_irreps = target_irreps
        self.hidden_irreps = hidden_irreps
        self.avg_num_neighbors = avg_num_neighbors
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        if edge_irreps is None:
            edge_irreps = self.node_feats_irreps
        self.radial_MLP = radial_MLP
        self.edge_irreps = edge_irreps


        self.linear_up = Linear(
            self.node_feats_irreps,
            self.edge_irreps,
            internal_weights=True,
            shared_weights=True,
        )

        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.edge_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = TensorProduct(
            self.edge_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )


        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = e3nn.nn.FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )


        self.irreps_out = self.target_irreps
        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
        )


        self.skip_tp = FullyConnectedTensorProduct(
            self.node_feats_irreps,
            self.node_attrs_irreps,
            self.hidden_irreps,
        )

        self.reshape = reshape_irreps(self.irreps_out)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: Optional[torch.Tensor] = None,
        first_layer: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sc = self.skip_tp(node_feats, node_attrs)
        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(edge_feats)
        if cutoff is not None:
            tp_weights = tp_weights * cutoff
        message = None
        if hasattr(self, "conv_fusion"):
            message = self.conv_tp(node_feats, edge_attrs, tp_weights, edge_index)
        else:
            mji = self.conv_tp(
                node_feats[edge_index[0]], edge_attrs, tp_weights
            )
            message = scatter_sum(
                src=mji, index=edge_index[1], dim=0, dim_size=node_feats.shape[0]
            )
        message = self.linear(message) / self.avg_num_neighbors
        return (
            self.reshape(message),
            sc,
        )

class MaceInteractionBlock(nn.Module):
    def __init__(self,
                 num_elements: int,
                 node_attrs_irreps: o3.Irreps,
                 node_feats_irreps: o3.Irreps,
                 edge_attrs_irreps: o3.Irreps,
                 edge_feats_irreps: o3.Irreps,
                 target_irreps: o3.Irreps,
                 hidden_irreps: o3.Irreps,
                 edge_irreps: Optional[o3.Irreps] = None,
                 correlation: int = 3,
                 radial_dim: int = 8,
                 radial_width: int = 64,
                 radial_depth: int = 2,
                 edge_attr_lmax: int = 2,
                 avg_num_neighbors: float = 10.0,
                 use_gate: bool = True,
                 lmax_center: int = 2,
                 lmax_env: int = 1,
                 cutoff: float = 5.0
                 ):
        super().__init__()

        self.num_elements = num_elements
        self.hidden_irreps = hidden_irreps
        self.irreps_out = target_irreps
        self.lmax = {"center": lmax_center, "env": lmax_env}


        self.rbf = BesselRBF1(num_radial=radial_dim, cutoff=cutoff)
        self.sh = SphericalHarmonicsEdgeAttrs(o3.Irreps.spherical_harmonics(lmax=edge_attr_lmax))


        self.interaction = RealAgnosticResidualInteractionBlock(
            node_attrs_irreps=node_attrs_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=edge_attrs_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=target_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            edge_irreps=edge_irreps,
            radial_MLP=[radial_width] * radial_depth,
        )


        self.product = EquivariantProductBasisBlock(
            node_feats_irreps=self.hidden_irreps,
            target_irreps=self.irreps_out,
            correlation=correlation,
            num_elements=num_elements,
            use_sc=True,
            use_agnostic_product=True
        )

    def forward(self, data: Data) -> Data:
        x = data.x
        edge_index = data.edge_index


        if hasattr(data, 'edge_vec'):
            edge_vec = data.edge_vec
        else:
            edge_vec = data.pos[edge_index[1]] - data.pos[edge_index[0]]
            if hasattr(data, 'edge_shift'):
                edge_vec = edge_vec + data.edge_shift


        edge_len = edge_vec.norm(dim=-1, keepdim=True)
        edge_attr = self.sh(edge_vec)
        edge_feats = self.rbf(edge_len)


        node_attrs = torch.nn.functional.one_hot(
            data.atom_type.squeeze().long(), num_classes=self.num_elements
        ).to(x.dtype)


        message, sc = self.interaction(
            node_feats=x,
            node_attrs=node_attrs,
            edge_attrs=edge_attr,
            edge_feats=edge_feats,
            edge_index=edge_index
        )


        out = self.product(
            node_feats=message,
            sc=sc,
            node_attrs=node_attrs
        )


        out = mask_irreps(out, self.irreps_out, data.is_center, self.lmax)
        data.x = out
        return data
