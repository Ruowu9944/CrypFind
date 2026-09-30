"""
PockMon model backbone.

Equivariant graph neural network with AF2 pair representation fusion
for residue-level cryptic pocket prediction.

Key design decisions:
- Whole-protein processing paradigm preserved (output [B, L])
- Internal dense <-> sparse conversion (transparent to train.py)
- Cutoff graph replaces KNN graph
- Geometric priors (dihedral, pseudo-Cb, fwd/rev) preserved
- GATA + EQFF message passing
- AF2 pair representation as auxiliary global interaction prior
- Coordinate unit: nm (consistent with MDTraj)
"""

import math
import os
from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import constant_, xavier_uniform_

import e3nn.o3
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter, softmax
from torch_cluster import radius_graph

zeros_initializer = partial(constant_, val=0.0)


# ============================================================================
#  Section 1: Utility layers
# ============================================================================

class CosineCutoff(nn.Module):
    def __init__(self, cutoff):
        super().__init__()
        if isinstance(cutoff, torch.Tensor):
            cutoff = cutoff.item()
        self.cutoff = cutoff

    def forward(self, distances):
        cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff) + 1.0)
        cutoffs = cutoffs * (distances < self.cutoff).float()
        return cutoffs


class ExpNormalSmearing(nn.Module):
    def __init__(self, cutoff=5.0, n_rbf=50, trainable=False):
        super().__init__()
        if isinstance(cutoff, torch.Tensor):
            cutoff = cutoff.item()
        self.cutoff = cutoff
        self.n_rbf = n_rbf
        self.cutoff_fn = CosineCutoff(cutoff)
        self.alpha = 5.0 / cutoff
        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        start_value = torch.exp(torch.scalar_tensor(-self.cutoff))
        means = torch.linspace(start_value, 1, self.n_rbf)
        betas = torch.tensor(
            [(2 / self.n_rbf * (1 - start_value)) ** -2] * self.n_rbf
        )
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(
            -self.betas * (torch.exp(self.alpha * (-dist)) - self.means) ** 2
        )


class Dense(nn.Linear):
    def __init__(
        self, in_features, out_features, bias=True, activation=None,
        weight_init=xavier_uniform_, bias_init=zeros_initializer,
        norm=None, gain=None,
    ):
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.gain = gain
        super().__init__(in_features, out_features, bias)
        self.activation = activation
        if norm == 'layer':
            self.norm = nn.LayerNorm(out_features)
        elif norm == 'batch':
            self.norm = nn.BatchNorm1d(out_features)
        else:
            self.norm = None

    def reset_parameters(self):
        if self.gain:
            self.weight_init(self.weight, gain=self.gain)
        else:
            self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, inputs):
        y = super().forward(inputs)
        if self.norm is not None:
            y = self.norm(y)
        if self.activation:
            y = self.activation(y)
        return y


class GotenMLP(nn.Module):
    def __init__(
        self, hidden_dims: List[int], bias=True, activation=None,
        last_activation=None, weight_init=xavier_uniform_,
        bias_init=zeros_initializer, norm='',
    ):
        super().__init__()
        dims = hidden_dims
        n_layers = len(dims)
        DenseMLP = partial(
            Dense, bias=bias, weight_init=weight_init, bias_init=bias_init
        )
        self.dense_layers = nn.ModuleList(
            [DenseMLP(dims[i], dims[i + 1], activation=activation, norm=norm)
             for i in range(n_layers - 2)]
            + [DenseMLP(dims[-2], dims[-1], activation=last_activation)]
        )
        self.layers = nn.Sequential(*self.dense_layers)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.dense_layers:
            m.reset_parameters()

    def forward(self, x):
        return self.layers(x)


def get_split_sizes_from_lmax(lmax, start=1):
    return [2 * l + 1 for l in range(start, lmax + 1)]


class TensorLayerNorm(nn.Module):
    def __init__(self, hidden_channels, trainable=False, lmax=1):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.eps = 1e-12
        self.lmax = lmax
        weight = torch.ones(self.hidden_channels)
        if trainable:
            self.register_parameter("weight", nn.Parameter(weight))
        else:
            self.register_buffer("weight", weight)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.fill_(1.0)

    def max_min_norm(self, tensor):
        dist = torch.norm(tensor, dim=1, keepdim=True)
        if (dist == 0).all():
            return torch.zeros_like(tensor)
        dist = dist.clamp(min=self.eps)
        direct = tensor / dist
        max_val, _ = torch.max(dist, dim=-1)
        min_val, _ = torch.min(dist, dim=-1)
        delta = (max_val - min_val).view(-1)
        delta = torch.where(delta == 0, torch.ones_like(delta), delta)
        dist = (dist - min_val.view(-1, 1, 1)) / delta.view(-1, 1, 1)
        return F.relu(dist) * direct

    def forward(self, tensor):
        split_sizes = get_split_sizes_from_lmax(self.lmax)
        vec_parts = torch.split(tensor, split_sizes, dim=1)
        normalized_parts = [self.max_min_norm(part) for part in vec_parts]
        normalized_vec = torch.cat(normalized_parts, dim=1)
        return normalized_vec * self.weight.unsqueeze(0).unsqueeze(0)


# ============================================================================
#  Section 2: Spherical harmonics helpers
# ============================================================================

def split_to_components(tensor, lmax, start=1, dim=-1):
    split_sizes = get_split_sizes_from_lmax(lmax, start=start)
    return torch.split(tensor, split_sizes, dim=dim)


# ============================================================================
#  Section 3: Format conversion (dense <-> sparse)
# ============================================================================

def dense_to_sparse(X_ca, S, mask):
    """Convert PocketMiner dense padded tensors to PyG sparse format.

    Args:
        X_ca: [B, L, 3] Ca coordinates (nm, padding=0)
        S:    [B, L]     amino acid types (padding=0)
        mask: [B, L]     valid residue mask (1=valid, 0=pad)

    Returns:
        pos:       [N_total, 3]
        seq:       [N_total]
        batch:     [N_total]
        local_idx: [N_total]  per-protein sequence index (0, 1, ..., Li-1)
    """
    valid = mask.bool()
    pos = X_ca[valid]
    seq = S[valid]
    B, L = mask.shape
    batch_idx = torch.arange(B, device=mask.device).unsqueeze(1).expand(B, L)
    batch = batch_idx[valid]
    lengths = mask.sum(dim=1).long()
    local_idx = torch.cat(
        [torch.arange(l.item(), device=mask.device) for l in lengths]
    )
    return pos, seq, batch, local_idx


def sparse_to_dense(h, mask):
    """Convert sparse per-residue values back to dense [B, L] format.

    Padding positions are filled with 0.
    """
    valid = mask.bool()
    if h.dim() == 1:
        h_dense = torch.zeros(mask.shape, device=h.device, dtype=h.dtype)
    else:
        h_dense = torch.zeros(
            *mask.shape, h.shape[-1], device=h.device, dtype=h.dtype
        )
    h_dense[valid] = h
    return h_dense


# ============================================================================
#  Section 4: Graph construction
# ============================================================================

def build_cutoff_graph(pos, batch, cutoff=1.5, loop=True, max_num_neighbors=128):
    """Build cutoff residue graph using PyG radius_graph.

    Args:
        pos:   [N_total, 3]  Ca coordinates (nm)
        batch: [N_total]     batch assignment
        cutoff: float        cutoff distance in nm (default 1.5 nm = 15 A)
        loop:  bool          include self-loops
        max_num_neighbors: int

    Returns:
        edge_index: [2, E_total]
        edge_dist:  [E_total]     scalar distances
        edge_vec:   [E_total, 3]  direction vectors (source - target)
    """
    edge_index = radius_graph(
        pos, r=cutoff, batch=batch, loop=loop,
        max_num_neighbors=max_num_neighbors,
    )
    edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
    if loop:
        nonself = edge_index[0] != edge_index[1]
        edge_dist = torch.zeros(edge_vec.size(0), device=edge_vec.device)
        edge_dist[nonself] = torch.norm(edge_vec[nonself], dim=-1)
    else:
        edge_dist = torch.norm(edge_vec, dim=-1)
    return edge_index, edge_dist, edge_vec


def build_dilated_sequence_edges(pos, lengths, dilations):
    """Build sparse long-range sequence edges inside each protein.

    The sparse residue order produced by dense-to-sparse conversion is batch-major,
    so cumulative sequence lengths map local residue indices to global node ids.
    Edges are directed in both directions for every requested dilation.
    """
    if not dilations or pos.numel() == 0:
        empty = torch.empty(2, 0, dtype=torch.long, device=pos.device)
        return empty

    edge_parts = []
    offset = 0
    for length_t in lengths:
        length = int(length_t.item())
        if length <= 1:
            offset += length
            continue
        for dilation in dilations:
            dilation = int(dilation)
            if dilation <= 0 or dilation >= length:
                continue
            src = torch.arange(0, length - dilation, device=pos.device) + offset
            dst = src + dilation
            edge_parts.append(torch.stack([src, dst], dim=0))
            edge_parts.append(torch.stack([dst, src], dim=0))
        offset += length

    if not edge_parts:
        return torch.empty(2, 0, dtype=torch.long, device=pos.device)
    return torch.cat(edge_parts, dim=1)


def augment_with_dilated_sequence_edges(
    edge_index, edge_dist, edge_vec, pos, lengths, dilations, edge_type
):
    """Append non-duplicate dilated sequence edges to an existing graph."""
    seq_edge_index = build_dilated_sequence_edges(pos, lengths, dilations)
    if seq_edge_index.numel() == 0:
        return edge_index, edge_dist, edge_vec, edge_type

    n_nodes = pos.size(0)
    local_key = edge_index[0] * n_nodes + edge_index[1]
    seq_key = seq_edge_index[0] * n_nodes + seq_edge_index[1]
    keep = ~torch.isin(seq_key, local_key)
    seq_edge_index = seq_edge_index[:, keep]
    if seq_edge_index.numel() == 0:
        return edge_index, edge_dist, edge_vec, edge_type

    seq_edge_vec = pos[seq_edge_index[0]] - pos[seq_edge_index[1]]
    seq_edge_dist = torch.norm(seq_edge_vec, dim=-1)
    seq_edge_type = torch.ones(
        seq_edge_index.size(1), dtype=torch.long, device=pos.device,
    )

    return (
        torch.cat([edge_index, seq_edge_index], dim=1),
        torch.cat([edge_dist, seq_edge_dist], dim=0),
        torch.cat([edge_vec, seq_edge_vec], dim=0),
        torch.cat([edge_type, seq_edge_type], dim=0),
    )


# ============================================================================
#  Section 5: Geometric feature computation (from PocketMiner, dense format)
# ============================================================================

def _normalize_vec(tensor, axis=-1, eps=1e-8):
    norms = torch.norm(tensor, dim=axis, keepdim=True).clamp(min=eps)
    return tensor / norms


def compute_dihedrals_dense(X, eps=1e-7):
    """Compute backbone dihedrals in dense format (reused from PocketMiner).

    Args:
        X: [B, L, A, 3]  full-atom coordinates (A >= 3)
    Returns:
        [B, L, 6]  cos/sin of 3 dihedral angles per residue
    """
    X3 = X[:, :, :3, :].reshape(X.shape[0], 3 * X.shape[1], 3)
    dX = X3[:, 1:, :] - X3[:, :-1, :]
    U = _normalize_vec(dX, axis=-1)
    u_2 = U[:, :-2, :]
    u_1 = U[:, 1:-1, :]
    u_0 = U[:, 2:, :]
    n_2 = _normalize_vec(torch.linalg.cross(u_2, u_1), axis=-1)
    n_1 = _normalize_vec(torch.linalg.cross(u_1, u_0), axis=-1)
    cosD = torch.sum(n_2 * n_1, dim=-1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign(torch.sum(u_2 * n_1, dim=-1)) * torch.acos(cosD)
    D = F.pad(D, (1, 2))
    D = D.reshape(D.shape[0], D.shape[1] // 3, 3)
    return torch.cat([torch.cos(D), torch.sin(D)], dim=2)


def compute_sidechains_dense(X):
    """Compute pseudo-Cb direction vectors (dense format, from PocketMiner).

    Uses only N, CA, C backbone atoms. NOT actual sidechain atoms.

    Args:
        X: [B, L, A, 3]
    Returns:
        [B, L, 3]  pseudo-Cb direction vector per residue
    """
    n, origin, c = X[:, :, 0, :], X[:, :, 1, :], X[:, :, 2, :]
    c_norm = _normalize_vec(c - origin)
    n_norm = _normalize_vec(n - origin)
    bisector = _normalize_vec(c_norm + n_norm)
    perp = _normalize_vec(torch.linalg.cross(c_norm, n_norm))
    return -bisector * (1.0 / 3.0) ** 0.5 - perp * (2.0 / 3.0) ** 0.5


def compute_orientations_dense(X_ca):
    """Compute forward/reverse backbone unit vectors (dense format).

    Args:
        X_ca: [B, L, 3]
    Returns:
        fwd: [B, L, 3]  forward direction
        rev: [B, L, 3]  reverse direction
    """
    forward = _normalize_vec(X_ca[:, 1:] - X_ca[:, :-1])
    backward = _normalize_vec(X_ca[:, :-1] - X_ca[:, 1:])
    forward = F.pad(forward, (0, 0, 0, 1))
    backward = F.pad(backward, (0, 0, 1, 0))
    return forward, backward


# ============================================================================
#  Section 6: Positional encoding for edges (sparse format)
# ============================================================================

class SparsePositionalEncodings(nn.Module):
    """Sinusoidal positional encoding based on sequence index difference."""

    def __init__(self, num_embeddings=16):
        super().__init__()
        self.num_embeddings = num_embeddings

    def forward(self, seq_diff):
        """
        Args:
            seq_diff: [E_total]  sequence index difference per edge
        Returns:
            [E_total, num_embeddings]
        """
        d = seq_diff.float().unsqueeze(-1)
        frequency = torch.exp(
            torch.arange(
                0, self.num_embeddings, 2,
                dtype=torch.float32, device=d.device,
            ) * -(math.log(10000.0) / self.num_embeddings)
        )
        angles = d * frequency.unsqueeze(0)
        return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)


# ============================================================================
#  Section 7: Residue-adapted NodeInit / EdgeInit
# ============================================================================

class ResidueNodeInit(MessagePassing):
    """NodeInit adapted for residue-level graph (max_z=21: 20 AA + 1 unknown)."""

    def __init__(
        self, hidden_channels, num_rbf, cutoff, max_z=21,
        activation=F.silu, proj_ln='',
        weight_init=nn.init.xavier_uniform_, bias_init=nn.init.zeros_,
    ):
        super().__init__(aggr="add")
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels]
        last_channel = hidden_channels[-1]
        self.A_nbr = nn.Embedding(max_z, last_channel)
        self.W_ndp = GotenMLP(
            [num_rbf] + [last_channel], activation=None, norm='',
            weight_init=weight_init, bias_init=bias_init,
            last_activation=None,
        )
        self.W_nrd_nru = GotenMLP(
            [2 * last_channel] + hidden_channels, activation=activation,
            norm=proj_ln, weight_init=weight_init, bias_init=bias_init,
            last_activation=None,
        )
        self.cutoff_fn = CosineCutoff(cutoff)
        self.reset_parameters()

    def reset_parameters(self):
        self.A_nbr.reset_parameters()
        self.W_ndp.reset_parameters()
        self.W_nrd_nru.reset_parameters()

    def forward(self, z, h, edge_index, r0_ij, varphi_r0_ij):
        mask = edge_index[0] != edge_index[1]
        if not mask.all():
            edge_index = edge_index[:, mask]
            r0_ij = r0_ij[mask]
            varphi_r0_ij = varphi_r0_ij[mask]
        h_src = self.A_nbr(z)
        phi_r0_ij = self.cutoff_fn(r0_ij)
        r0_ij_feat = self.W_ndp(varphi_r0_ij) * phi_r0_ij.view(-1, 1)
        m_i = self.propagate(
            edge_index, h_src=h_src, r0_ij_feat=r0_ij_feat, size=None,
        )
        return self.W_nrd_nru(torch.cat([h, m_i], dim=1))

    def message(self, h_src_j, r0_ij_feat):
        return h_src_j * r0_ij_feat


class ResidueEdgeInit(MessagePassing):
    """EdgeInit with expanded input (RBF + positional encoding)."""

    def __init__(self, num_edge_basis, hidden_channels):
        super().__init__(aggr=None)
        self.W_erp = nn.Linear(num_edge_basis, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_erp.weight)
        self.W_erp.bias.data.fill_(0)

    def forward(self, edge_index, edge_basis, h):
        return self.propagate(edge_index, h=h, edge_basis=edge_basis)

    def message(self, h_i, h_j, edge_basis):
        return (h_i + h_j) * self.W_erp(edge_basis)

    def aggregate(self, features, index):
        return features


# ============================================================================
#  Section 8: GATA
# ============================================================================

class GATA(MessagePassing):
    """Geometry-Aware Tensor Attention layer."""

    def __init__(
        self, n_atom_basis, activation, weight_init=nn.init.xavier_uniform_,
        bias_init=nn.init.zeros_, aggr="add", node_dim=0, epsilon=1e-7,
        layer_norm="", steerable_norm="", cutoff=5.0, num_heads=8,
        dropout=0.0, edge_updates=False, last_layer=False, scale_edge=True,
        evec_dim=None, emlp_dim=None, sep_htr=True, sep_dir=True,
        sep_tensor=True, lmax=2, edge_ln="",
    ):
        super().__init__(aggr=aggr, node_dim=node_dim)
        self.sep_htr = sep_htr
        self.epsilon = epsilon
        self.last_layer = last_layer
        self.edge_updates = edge_updates
        self.scale_edge = scale_edge
        self.activation = activation
        self.sep_dir = sep_dir
        self.sep_tensor = sep_tensor

        update_info = {
            "gated": False, "rej": True, "mlp": False, "mlpa": False,
            "lin_w": 0, "lin_ln": 0,
        }
        update_parts = (
            edge_updates.split("_") if isinstance(edge_updates, str) else []
        )
        if "gated" in update_parts:
            update_info["gated"] = "gated"
        if "gatedt" in update_parts:
            update_info["gated"] = "gatedt"
        if "act" in update_parts:
            update_info["gated"] = "act"
        if "norej" in update_parts:
            update_info["rej"] = False
        if "mlp" in update_parts:
            update_info["mlp"] = True
        if "mlpa" in update_parts:
            update_info["mlpa"] = True
        if "linw" in update_parts:
            update_info["lin_w"] = 1
        if "linwa" in update_parts:
            update_info["lin_w"] = 2
        if "ln" in update_parts:
            update_info["lin_ln"] = 1
        if "postln" in update_parts:
            update_info["lin_ln"] = 2
        self.update_info = update_info

        self.dropout = dropout
        self.n_atom_basis = n_atom_basis
        self.lmax = lmax

        multiplier = 3
        if self.sep_dir:
            multiplier += lmax - 1
        if self.sep_tensor:
            multiplier += lmax - 1
        self.multiplier = multiplier

        InitDense = partial(Dense, weight_init=weight_init, bias_init=bias_init)

        self.gamma_s = nn.Sequential(
            InitDense(n_atom_basis, n_atom_basis, activation=activation),
            InitDense(n_atom_basis, multiplier * n_atom_basis, activation=None),
        )
        self.num_heads = num_heads
        self.W_q = InitDense(n_atom_basis, n_atom_basis, activation=None)
        self.W_k = InitDense(n_atom_basis, n_atom_basis, activation=None)
        self.gamma_v = nn.Sequential(
            InitDense(n_atom_basis, n_atom_basis, activation=activation),
            InitDense(n_atom_basis, multiplier * n_atom_basis, activation=None),
        )
        self.W_re = InitDense(n_atom_basis, n_atom_basis, activation=activation)

        self.edge_vec_dim = n_atom_basis if evec_dim is None else evec_dim
        self.edge_mlp_dim = n_atom_basis if emlp_dim is None else emlp_dim

        if not self.last_layer and self.edge_updates:
            if self.update_info["mlp"] or self.update_info["mlpa"]:
                dims = [n_atom_basis, self.edge_mlp_dim, n_atom_basis]
            else:
                dims = [n_atom_basis, n_atom_basis]
            self.gamma_t = GotenMLP(
                dims, activation=activation,
                last_activation=None if self.update_info["mlp"] else self.activation,
                norm=edge_ln,
            )
            self.W_vq = InitDense(
                n_atom_basis, self.edge_vec_dim, activation=None, bias=False,
            )
            if self.sep_htr:
                self.W_vk = nn.ModuleList([
                    InitDense(n_atom_basis, self.edge_vec_dim, activation=None, bias=False)
                    for _ in range(self.lmax)
                ])
            else:
                self.W_vk = InitDense(
                    n_atom_basis, self.edge_vec_dim, activation=None, bias=False,
                )
            modules = []
            if self.update_info["lin_w"] > 0:
                if self.update_info["lin_ln"] == 1:
                    modules.append(nn.LayerNorm(self.edge_vec_dim))
                if self.update_info["lin_w"] % 10 == 2:
                    modules.append(self.activation)
                self.W_edp = InitDense(
                    self.edge_vec_dim, n_atom_basis, activation=None,
                    norm="layer" if self.update_info["lin_ln"] == 2 else "",
                )
                modules.append(self.W_edp)
            if self.update_info["gated"] == "gatedt":
                modules.append(nn.Tanh())
            elif self.update_info["gated"] == "gated":
                modules.append(nn.Sigmoid())
            elif self.update_info["gated"] == "act":
                modules.append(nn.SiLU())
            self.gamma_w = nn.Sequential(*modules)

        self.cutoff = CosineCutoff(cutoff)
        self._alpha = None
        self.W_rs = InitDense(
            n_atom_basis, n_atom_basis * self.multiplier, activation=None,
        )
        self.layernorm_ = layer_norm
        self.steerable_norm_ = steerable_norm
        self.layernorm = (
            nn.LayerNorm(n_atom_basis) if layer_norm != "" else nn.Identity()
        )
        self.tensor_layernorm = (
            TensorLayerNorm(n_atom_basis, trainable=False, lmax=self.lmax)
            if steerable_norm != "" else nn.Identity()
        )
        self.reset_parameters()

    def reset_parameters(self):
        if self.layernorm_:
            self.layernorm.reset_parameters()
        if self.steerable_norm_:
            self.tensor_layernorm.reset_parameters()
        for l in self.gamma_s:
            l.reset_parameters()
        self.W_q.reset_parameters()
        self.W_k.reset_parameters()
        for l in self.gamma_v:
            l.reset_parameters()
        self.W_rs.reset_parameters()
        if not self.last_layer and self.edge_updates:
            self.gamma_t.reset_parameters()
            self.W_vq.reset_parameters()
            if self.sep_htr:
                for w in self.W_vk:
                    w.reset_parameters()
            else:
                self.W_vk.reset_parameters()
            if self.update_info["lin_w"] > 0:
                self.W_edp.reset_parameters()

    @staticmethod
    def vector_rejection(rep, rl_ij):
        vec_proj = (rep * rl_ij.unsqueeze(2)).sum(dim=1, keepdim=True)
        return rep - vec_proj * rl_ij.unsqueeze(2)

    def forward(self, edge_index, h, X, rl_ij, t_ij, r_ij, n_edges):
        h = self.layernorm(h)
        X = self.tensor_layernorm(X)
        q = self.W_q(h).reshape(
            -1, self.num_heads, self.n_atom_basis // self.num_heads
        )
        k = self.W_k(h).reshape(
            -1, self.num_heads, self.n_atom_basis // self.num_heads
        )
        x = self.gamma_s(h)
        v = self.gamma_v(h)
        t_ij_attn = self.W_re(t_ij)
        t_ij_filter = self.W_rs(t_ij)

        d_h, d_X = self.propagate(
            edge_index=edge_index, x=x, q=q, k=k, v=v, X=X,
            t_ij_filter=t_ij_filter, t_ij_attn=t_ij_attn,
            r_ij=r_ij, rl_ij=rl_ij, n_edges=n_edges,
        )
        h = h + d_h
        X = X + d_X

        if not self.last_layer and self.edge_updates:
            X_htr = X
            EQ = self.W_vq(X_htr)
            if self.sep_htr:
                X_split = torch.split(
                    X_htr, get_split_sizes_from_lmax(self.lmax), dim=1,
                )
                EK = torch.cat(
                    [w(X_split[i]) for i, w in enumerate(self.W_vk)], dim=1,
                )
            else:
                EK = self.W_vk(X_htr)
            dt_ij = self.edge_updater(
                edge_index, EQ=EQ, EK=EK, rl_ij=rl_ij, t_ij=t_ij,
            )
            t_ij = t_ij + dt_ij

        self._alpha = None
        return h, X, t_ij

    def message(
        self, edge_index, x_j, q_i, k_j, v_j, X_j,
        t_ij_filter, t_ij_attn, r_ij, rl_ij, n_edges,
        index, ptr, dim_size,
    ):
        t_ij_attn = t_ij_attn.reshape(
            -1, self.num_heads, self.n_atom_basis // self.num_heads,
        )
        attn = (q_i * k_j * t_ij_attn).sum(dim=-1, keepdim=True)
        attn = softmax(attn, index, ptr, dim_size)
        if self.scale_edge:
            norm = (
                torch.sqrt(n_edges.reshape(-1, 1, 1))
                / np.sqrt(self.n_atom_basis)
            )
        else:
            norm = 1.0 / np.sqrt(self.n_atom_basis)
        attn = attn * norm
        self._alpha = attn
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        sea_ij = attn * v_j.reshape(
            -1, self.num_heads,
            (self.n_atom_basis * self.multiplier) // self.num_heads,
        )
        sea_ij = sea_ij.reshape(-1, 1, self.n_atom_basis * self.multiplier)

        spatial_attn = (
            t_ij_filter.unsqueeze(1)
            * x_j
            * self.cutoff(r_ij.unsqueeze(-1).unsqueeze(-1))
        )
        outputs = spatial_attn + sea_ij
        components = torch.split(outputs, self.n_atom_basis, dim=-1)
        o_s_ij = components[0]
        components = components[1:]

        if self.sep_dir:
            o_d_l_ij, components = (
                components[: self.lmax], components[self.lmax :],
            )
            rl_ij_split = split_to_components(rl_ij[..., None], self.lmax, dim=1)
            dir_comps = [rl_ij_split[i] * o_d_l_ij[i] for i in range(self.lmax)]
            dX_R = torch.cat(dir_comps, dim=1)
        else:
            o_d_ij, components = components[0], components[1:]
            dX_R = o_d_ij * rl_ij[..., None]

        if self.sep_tensor:
            o_t_l_ij = components[: self.lmax]
            X_j_split = split_to_components(X_j, self.lmax, dim=1)
            tensor_comps = [
                X_j_split[i] * o_t_l_ij[i] for i in range(self.lmax)
            ]
            dX_X = torch.cat(tensor_comps, dim=1)
        else:
            o_t_ij = components[0]
            dX_X = o_t_ij * X_j

        dX = dX_R + dX_X
        return o_s_ij, dX

    def edge_update(self, EQ_i, EK_j, rl_ij, t_ij):
        if self.sep_htr:
            EQ_i_split = split_to_components(EQ_i, self.lmax, dim=1)
            EK_j_split = split_to_components(EK_j, self.lmax, dim=1)
            rl_ij_split = split_to_components(rl_ij, self.lmax, dim=1)
            pairs = []
            for l_idx in range(len(EQ_i_split)):
                if self.update_info["rej"]:
                    EQ_i_l = self.vector_rejection(
                        EQ_i_split[l_idx], rl_ij_split[l_idx],
                    )
                    EK_j_l = self.vector_rejection(
                        EK_j_split[l_idx], -rl_ij_split[l_idx],
                    )
                else:
                    EQ_i_l = EQ_i_split[l_idx]
                    EK_j_l = EK_j_split[l_idx]
                pairs.append((EQ_i_l, EK_j_l))
        elif not self.update_info["rej"]:
            pairs = [(EQ_i, EK_j)]
        else:
            EQr_i = self.vector_rejection(EQ_i, rl_ij)
            EKr_j = self.vector_rejection(EK_j, -rl_ij)
            pairs = [(EQr_i, EKr_j)]

        w_ij = None
        for el in pairs:
            EQ_l, EK_l = el
            w_l = (EQ_l * EK_l).sum(dim=1)
            w_ij = w_l if w_ij is None else w_ij + w_l
        return self.gamma_t(t_ij) * self.gamma_w(w_ij)

    def aggregate(self, features, index, ptr, dim_size):
        h, X = features
        h = scatter(h, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        X = scatter(X, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        return h, X

    def update(self, inputs):
        return inputs


# ============================================================================
#  Section 9: EQFF
# ============================================================================

class EQFF(nn.Module):
    """Equivariant Feed-Forward mixing for scalar and steerable features."""

    def __init__(
        self, n_atom_basis, activation, lmax, epsilon=1e-8,
        weight_init=nn.init.xavier_uniform_, bias_init=nn.init.zeros_,
    ):
        super().__init__()
        self.lmax = lmax
        self.n_atom_basis = n_atom_basis
        self.epsilon = epsilon
        InitDense = partial(Dense, weight_init=weight_init, bias_init=bias_init)
        context_dim = 2 * n_atom_basis
        self.gamma_m = nn.Sequential(
            InitDense(context_dim, n_atom_basis, activation=activation),
            InitDense(n_atom_basis, 2 * n_atom_basis, activation=None),
        )
        self.W_vu = InitDense(
            n_atom_basis, n_atom_basis, activation=None, bias=False,
        )

    def reset_parameters(self):
        self.W_vu.reset_parameters()
        for l in self.gamma_m:
            l.reset_parameters()

    def forward(self, h, X):
        X_p = self.W_vu(X)
        X_pn = torch.sqrt(
            torch.sum(X_p ** 2, dim=-2, keepdim=True) + self.epsilon
        )
        ctx = torch.cat([h, X_pn], dim=-1)
        x = self.gamma_m(ctx)
        m1, m2 = torch.split(x, self.n_atom_basis, dim=-1)
        h = h + m1
        X = X + m2 * X_p
        return h, X


# ============================================================================
#  Section 10: PockMonEncoder
# ============================================================================

class PockMonEncoder(nn.Module):
    """Equivariant encoder for residue-level protein graph."""

    def __init__(
        self, n_atom_basis=128, n_interactions=4, lmax=2,
        num_heads=4, dropout=0.0, cutoff=1.5,
        activation=F.silu, aggr='add',
        layernorm='layer', steerable_norm='layer',
        edge_updates=False, scale_edge=True,
        sep_htr=True, sep_dir=True, sep_tensor=True,
        weight_init=nn.init.xavier_uniform_,
        bias_init=nn.init.zeros_,
    ):
        super().__init__()
        self.n_interactions = n_interactions
        self.n_atom_basis = n_atom_basis
        self.lmax = lmax

        self.gata_list = nn.ModuleList([
            GATA(
                n_atom_basis=n_atom_basis, activation=activation, aggr=aggr,
                weight_init=weight_init, bias_init=bias_init,
                layer_norm=layernorm, steerable_norm=steerable_norm,
                cutoff=cutoff, epsilon=1e-7, num_heads=num_heads,
                dropout=dropout, edge_updates=edge_updates,
                last_layer=(i == n_interactions - 1),
                scale_edge=scale_edge, sep_htr=sep_htr,
                sep_dir=sep_dir, sep_tensor=sep_tensor, lmax=lmax,
            )
            for i in range(n_interactions)
        ])
        self.eqff_list = nn.ModuleList([
            EQFF(
                n_atom_basis=n_atom_basis, activation=activation,
                lmax=lmax, epsilon=1e-8,
                weight_init=weight_init, bias_init=bias_init,
            )
            for _ in range(n_interactions)
        ])

    def forward(self, h, X, t_ij, rl_ij, edge_index, edge_dist, n_edges):
        """
        Args:
            h:          [N_total, 1, H]
            X:          [N_total, equi_dim, H]
            t_ij:       [E_total, H]
            rl_ij:      [E_total, equi_dim]
            edge_index: [2, E_total]
            edge_dist:  [E_total]
            n_edges:    [E_total]
        Returns:
            h: [N_total, H]   (squeezed)
            X: [N_total, equi_dim, H]
        """
        for gata, eqff in zip(self.gata_list, self.eqff_list):
            h, X, t_ij = gata(
                edge_index, h, X, rl_ij=rl_ij, t_ij=t_ij,
                r_ij=edge_dist, n_edges=n_edges,
            )
            h, X = eqff(h, X)
        h = h.squeeze(1)
        return h, X


# ============================================================================
#  Section 11: AF2 Pair modules
# ============================================================================

class AF2PairProvider:
    """Loads and caches AF2 pair representations, indexed by amino acid sequence."""

    def __init__(self, pair_dir):
        self.pairs = {}
        if pair_dir and os.path.isdir(pair_dir):
            self._load_all(pair_dir)

    def _load_all(self, pair_dir):
        count = 0
        for fname in sorted(os.listdir(pair_dir)):
            if not fname.endswith('_pair.npz'):
                continue
            fpath = os.path.join(pair_dir, fname)
            data = np.load(fpath)
            if 'pair_repr' not in data or 'aatype' not in data:
                continue
            seq_key = tuple(data['aatype'].tolist())
            self.pairs[seq_key] = data['pair_repr']
            count += 1
        print(f"AF2PairProvider: loaded {count} pair representations "
              f"from {pair_dir}")

    def get_pair_repr(self, S, mask, device):
        """Look up pair representations by matching amino acid sequences.

        Args:
            S:    [B, L]  amino acid type IDs
            mask: [B, L]  valid residue mask
            device: target device

        Returns:
            list of (Tensor [Ni, Ni, C_pair] or None) for each protein
        """
        B = S.shape[0]
        results = []
        for b in range(B):
            valid = mask[b].bool()
            seq_key = tuple(S[b, valid].cpu().tolist())
            pair_np = self.pairs.get(seq_key)
            if pair_np is not None:
                results.append(
                    torch.from_numpy(pair_np.astype(np.float32)).to(device)
                )
            else:
                results.append(None)
        return results


class AF2BinderPairProvider:
    """Loads AF2 binder pair features [L, 5120].

    Supports two lookup modes:
      1. By structureId (preferred): exact file-based lookup with automatic
         length alignment when mdtraj and AF2 disagree on residue count.
      2. By aatype tuple (legacy fallback): used when meta/structureId is
         not available (e.g. PocketMiner evaluation path).

    Each .npz contains:
      - features : float32/16, [L, 5120]   (pair_A + pair_B concatenated)
      - aatype   : int32,      [L]
    """

    def __init__(self, binder_dir):
        self.by_id = {}
        self.by_seq = {}
        if binder_dir and os.path.isdir(binder_dir):
            self._load_all(binder_dir)

    def _load_all(self, binder_dir):
        count = 0
        for fname in sorted(os.listdir(binder_dir)):
            if not fname.endswith('.npz'):
                continue
            fpath = os.path.join(binder_dir, fname)
            data = np.load(fpath)
            if 'features' not in data:
                continue

            sid = fname.replace('.npz', '')
            aatype = data['aatype'] if 'aatype' in data else None
            feat = data['features']

            self.by_id[sid] = (feat, aatype)

            if aatype is not None:
                seq_key = tuple(aatype.tolist())
                self.by_seq[seq_key] = feat

            count += 1
        print(f"AF2BinderPairProvider: loaded {count} binder features "
              f"from {binder_dir} ({len(self.by_id)} by id, "
              f"{len(self.by_seq)} unique seq keys)")

    # ------------------------------------------------------------------
    #  Alignment for length-mismatched binder features
    # ------------------------------------------------------------------

    @staticmethod
    def _subsequence_align(binder_feat, binder_aatype, cache_seq):
        """Align via greedy subsequence matching (handles scattered insertions).

        Finds the binder sequence as a subsequence of cache_seq (skipping
        extra residues such as CYM/ACE/NME from MD simulations or merged
        multi-chain PDBs). Each matched binder residue's features are placed
        at the corresponding cache position; unmatched cache positions get
        zeros.

        Returns:
            [cache_len, F] float32 array, or None if fewer than 90% of
            binder residues could be matched.
        """
        cache_len = len(cache_seq)
        binder_len = binder_feat.shape[0]
        feat_dim = binder_feat.shape[1]

        bi = 0
        mapping = {}
        for ci in range(cache_len):
            if bi < binder_len and binder_aatype[bi] == cache_seq[ci]:
                mapping[bi] = ci
                bi += 1
        if bi < int(binder_len * 0.9):
            return None

        aligned = np.zeros((cache_len, feat_dim), dtype=np.float32)
        for b_idx, c_idx in mapping.items():
            aligned[c_idx] = binder_feat[b_idx].astype(np.float32)
        return aligned

    @staticmethod
    def _align_features(binder_feat, binder_aatype, cache_seq):
        """Align binder features to cache sequence.

        Tries sliding-window alignment first; falls back to subsequence
        matching when the sliding window yields a poor match rate (< 90%).
        This handles both simple prefix/suffix truncations and scattered
        insertions (e.g. CYM/ACE/NME from MD simulations).

        Args:
            binder_feat:   [binder_len, F]  float array
            binder_aatype: [binder_len]     int array (or None)
            cache_seq:     [cache_len]      int array

        Returns:
            [cache_len, F] float32 array
        """
        cache_len = len(cache_seq)
        binder_len = binder_feat.shape[0]
        feat_dim = binder_feat.shape[1]

        if binder_len == cache_len:
            return binder_feat.astype(np.float32)

        aligned = np.zeros((cache_len, feat_dim), dtype=np.float32)

        if binder_aatype is None:
            n = min(binder_len, cache_len)
            aligned[:n] = binder_feat[:n].astype(np.float32)
            return aligned

        ref_len = min(binder_len, cache_len)
        best_start, best_matches = 0, -1

        if binder_len > cache_len:
            for start in range(binder_len - cache_len + 1):
                matches = int(np.sum(
                    binder_aatype[start:start + cache_len] == cache_seq
                ))
                if matches > best_matches:
                    best_matches = matches
                    best_start = start

            if best_matches >= int(ref_len * 0.9):
                aligned[:] = binder_feat[
                    best_start:best_start + cache_len
                ].astype(np.float32)
                return aligned
        else:
            for start in range(cache_len - binder_len + 1):
                matches = int(np.sum(
                    cache_seq[start:start + binder_len] == binder_aatype
                ))
                if matches > best_matches:
                    best_matches = matches
                    best_start = start

            if best_matches >= int(ref_len * 0.9):
                aligned[
                    best_start:best_start + binder_len
                ] = binder_feat.astype(np.float32)
                return aligned

        subseq = AF2BinderPairProvider._subsequence_align(
            binder_feat, binder_aatype, cache_seq,
        )
        if subseq is not None:
            return subseq

        aligned = np.zeros((cache_len, feat_dim), dtype=np.float32)
        if binder_len > cache_len:
            aligned[:] = binder_feat[
                best_start:best_start + cache_len
            ].astype(np.float32)
        else:
            aligned[
                best_start:best_start + binder_len
            ] = binder_feat.astype(np.float32)
        return aligned

    # ------------------------------------------------------------------
    #  Primary lookup: by structureId (with alignment)
    # ------------------------------------------------------------------

    def get_binder_features(self, S, mask, device, meta=None):
        """Look up binder pair features, preferring structureId when available.

        Args:
            S:      [B, L]  amino acid types
            mask:   [B, L]  valid residue mask
            device: torch device
            meta:   list[str] or None -- structureId per protein

        Returns:
            list of (Tensor [Ni, 5120] or None) for each protein
        """
        B = S.shape[0]
        results = []
        for b in range(B):
            valid = mask[b].bool()
            cache_seq = S[b, valid].cpu().numpy()
            n_valid = int(valid.sum().item())

            feat_np = None

            if meta is not None and b < len(meta):
                sid = meta[b]
                entry = self.by_id.get(sid)
                if entry is not None:
                    raw_feat, raw_aatype = entry
                    if raw_feat.shape[0] == n_valid:
                        feat_np = raw_feat.astype(np.float32)
                    else:
                        feat_np = self._align_features(
                            raw_feat, raw_aatype, cache_seq,
                        )

            if feat_np is None:
                seq_key = tuple(cache_seq.tolist())
                raw = self.by_seq.get(seq_key)
                if raw is not None and raw.shape[0] == n_valid:
                    feat_np = raw.astype(np.float32)

            if feat_np is not None:
                results.append(torch.from_numpy(feat_np).to(device))
            else:
                results.append(None)
        return results


class AF2CombinedPairProvider:
    """Loads combined pair features: binder fingerprint [L, 5120] +
    target-target global pair [L, L, 128], both extracted from a single
    binder-protocol forward pass.

    Supports two lookup modes (mirrors AF2BinderPairProvider):
      1. By structureId (preferred): exact file-based lookup with automatic
         length alignment when mdtraj and AF2 disagree on residue count.
      2. By aatype tuple (legacy fallback): used when meta/structureId is
         not available (e.g. PocketMiner evaluation path).

    Each npz file must contain:
      - features  : float16/32, [L, 5120]
      - pair_repr : float16/32, [L, L, 128]
      - aatype    : int32,      [L]
    """

    def __init__(self, combined_dir):
        self.by_id = {}
        self.by_seq = {}
        if combined_dir and os.path.isdir(combined_dir):
            self._load_all(combined_dir)

    def _load_all(self, combined_dir):
        count = 0
        for fname in sorted(os.listdir(combined_dir)):
            if not fname.endswith('.npz'):
                continue
            fpath = os.path.join(combined_dir, fname)
            try:
                data = np.load(fpath)
            except Exception:
                continue
            if 'features' not in data or 'pair_repr' not in data:
                continue
            aatype = data['aatype'] if 'aatype' in data else None
            binder_feat = data['features']
            pair_repr = data['pair_repr']

            sid = fname.replace('.npz', '')
            self.by_id[sid] = (binder_feat, pair_repr, aatype)

            if aatype is not None:
                seq_key = tuple(aatype.tolist())
                self.by_seq[seq_key] = (binder_feat, pair_repr)

            count += 1
        print(f"AF2CombinedPairProvider: loaded {count} combined features "
              f"from {combined_dir} ({len(self.by_id)} by id, "
              f"{len(self.by_seq)} unique seq keys)")

    # ------------------------------------------------------------------
    #  Alignment helpers (same three-level strategy as BinderPairProvider
    #  but extended to simultaneously align [L, F] and [L, L, C])
    # ------------------------------------------------------------------

    @staticmethod
    def _subsequence_align(binder_feat, pair_repr, source_aatype, cache_seq):
        """Greedy subsequence matching for combined features.

        Finds source_aatype as a subsequence of cache_seq, placing matched
        residue features at the corresponding cache positions. Unmatched
        cache positions get zeros. pair_repr is indexed on both axes.

        Returns (aligned_feat, aligned_pair) or None if < 90% matched.
        """
        cache_len = len(cache_seq)
        source_len = len(source_aatype)

        bi = 0
        mapping = {}
        for ci in range(cache_len):
            if bi < source_len and source_aatype[bi] == cache_seq[ci]:
                mapping[bi] = ci
                bi += 1
        if bi < int(source_len * 0.9):
            return None

        feat_dim = binder_feat.shape[1]
        pair_dim = pair_repr.shape[2]
        aligned_feat = np.zeros((cache_len, feat_dim), dtype=np.float32)
        aligned_pair = np.zeros(
            (cache_len, cache_len, pair_dim), dtype=np.float32,
        )

        src_idx = np.asarray(sorted(mapping.keys()), dtype=np.int64)
        dst_idx = np.asarray([mapping[s] for s in src_idx], dtype=np.int64)
        aligned_feat[dst_idx] = binder_feat[src_idx].astype(np.float32)
        aligned_pair[np.ix_(dst_idx, dst_idx)] = pair_repr[
            np.ix_(src_idx, src_idx)
        ].astype(np.float32)
        return aligned_feat, aligned_pair

    @staticmethod
    def _align_combined(binder_feat, pair_repr, source_aatype, cache_seq):
        """Align combined features to cache sequence.

        Three-level strategy identical to AF2BinderPairProvider._align_features:
          1. Sliding-window alignment (≥ 90% match rate)
          2. Greedy subsequence matching (≥ 90% matched)
          3. Fallback to sliding-window best position

        Args:
            binder_feat:   [source_len, F]           float array
            pair_repr:     [source_len, source_len, C] float array
            source_aatype: [source_len]              int array (or None)
            cache_seq:     [cache_len]               int array

        Returns:
            (aligned_feat [cache_len, F],
             aligned_pair [cache_len, cache_len, C])
        """
        cache_len = len(cache_seq)
        source_len = binder_feat.shape[0]
        feat_dim = binder_feat.shape[1]
        pair_dim = pair_repr.shape[2]

        if source_len == cache_len:
            return (
                binder_feat.astype(np.float32),
                pair_repr.astype(np.float32),
            )

        aligned_feat = np.zeros((cache_len, feat_dim), dtype=np.float32)
        aligned_pair = np.zeros(
            (cache_len, cache_len, pair_dim), dtype=np.float32,
        )

        if source_aatype is None:
            n = min(source_len, cache_len)
            aligned_feat[:n] = binder_feat[:n].astype(np.float32)
            aligned_pair[:n, :n, :] = pair_repr[:n, :n, :].astype(np.float32)
            return aligned_feat, aligned_pair

        ref_len = min(source_len, cache_len)
        best_start, best_matches = 0, -1

        if source_len > cache_len:
            for start in range(source_len - cache_len + 1):
                matches = int(np.sum(
                    source_aatype[start:start + cache_len] == cache_seq
                ))
                if matches > best_matches:
                    best_matches = matches
                    best_start = start

            if best_matches >= int(ref_len * 0.9):
                src = slice(best_start, best_start + cache_len)
                return (
                    binder_feat[src].astype(np.float32),
                    pair_repr[src, src, :].astype(np.float32),
                )
        else:
            for start in range(cache_len - source_len + 1):
                matches = int(np.sum(
                    cache_seq[start:start + source_len] == source_aatype
                ))
                if matches > best_matches:
                    best_matches = matches
                    best_start = start

            if best_matches >= int(ref_len * 0.9):
                dst = slice(best_start, best_start + source_len)
                aligned_feat[dst] = binder_feat.astype(np.float32)
                aligned_pair[dst, dst, :] = pair_repr.astype(np.float32)
                return aligned_feat, aligned_pair

        subseq = AF2CombinedPairProvider._subsequence_align(
            binder_feat, pair_repr, source_aatype, cache_seq,
        )
        if subseq is not None:
            return subseq

        aligned_feat = np.zeros((cache_len, feat_dim), dtype=np.float32)
        aligned_pair = np.zeros(
            (cache_len, cache_len, pair_dim), dtype=np.float32,
        )
        if source_len > cache_len:
            src = slice(best_start, best_start + cache_len)
            aligned_feat[:] = binder_feat[src].astype(np.float32)
            aligned_pair[:, :, :] = pair_repr[src, src, :].astype(np.float32)
        else:
            dst = slice(best_start, best_start + source_len)
            aligned_feat[dst] = binder_feat.astype(np.float32)
            aligned_pair[dst, dst, :] = pair_repr.astype(np.float32)
        return aligned_feat, aligned_pair

    # ------------------------------------------------------------------
    #  Primary lookup: by structureId (with alignment), fallback by_seq
    # ------------------------------------------------------------------

    def get_combined_features(self, S, mask, device, meta=None):
        """Look up both binder fingerprint and global pair for each protein.

        Lookup order (mirrors AF2BinderPairProvider.get_binder_features):
          1. meta/structureId -> by_id (O(1)), align if lengths differ
          2. aatype tuple     -> by_seq (O(1)), exact length match only

        Args:
            S:      [B, L]  amino acid types
            mask:   [B, L]  valid residue mask
            device: torch device
            meta:   list[str] or None -- structureId per protein

        Returns:
            (binder_list, pair_list)
            binder_list: list of (Tensor [Ni, 5120] or None)
            pair_list:   list of (Tensor [Ni, Ni, 128] or None)
        """
        B = S.shape[0]
        binder_list = []
        pair_list = []
        for b in range(B):
            valid = mask[b].bool()
            n_valid = int(valid.sum().item())
            cache_seq = S[b, valid].cpu().numpy()

            feat_result = None

            if meta is not None and b < len(meta):
                sid = meta[b]
                entry = self.by_id.get(sid)
                if entry is not None:
                    raw_feat, raw_pair, raw_aatype = entry
                    if raw_feat.shape[0] == n_valid:
                        feat_result = (
                            raw_feat.astype(np.float32),
                            raw_pair.astype(np.float32),
                        )
                    else:
                        feat_result = self._align_combined(
                            raw_feat, raw_pair, raw_aatype, cache_seq,
                        )

            if feat_result is None:
                seq_key = tuple(cache_seq.tolist())
                entry = self.by_seq.get(seq_key)
                if entry is not None:
                    bf, pr = entry
                    if bf.shape[0] == n_valid:
                        feat_result = (
                            bf.astype(np.float32),
                            pr.astype(np.float32),
                        )

            if feat_result is not None:
                bf_np, pr_np = feat_result
                binder_list.append(torch.from_numpy(bf_np).to(device))
                pair_list.append(torch.from_numpy(pr_np).to(device))
            else:
                binder_list.append(None)
                pair_list.append(None)
        return binder_list, pair_list


class BinderPairProjection(nn.Module):
    """Project binder pair features [L, 5120] -> [L, c_proj] per residue."""

    def __init__(self, c_in=5120, c_proj=64, dropout=0.1):
        super().__init__()
        self.c_proj = c_proj
        self.proj = nn.Sequential(
            nn.Linear(c_in, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, c_proj),
        )

    def forward(self, binder_features, batch):
        """
        Args:
            binder_features: list of [Ni, 5120] or None, one per protein
            batch:           [N_total] batch assignment
        Returns:
            [N_total, c_proj]
        """
        device = batch.device
        N_total = batch.shape[0]
        summary = torch.zeros(N_total, self.c_proj, device=device)
        B = batch.max().item() + 1

        for b in range(B):
            if b >= len(binder_features) or binder_features[b] is None:
                continue
            node_mask = (batch == b)
            n_b = node_mask.sum().item()
            feat = binder_features[b]
            if feat.shape[0] != n_b:
                continue
            summary[node_mask] = self.proj(feat)

        return summary


class BinderPairLogitHead(nn.Module):
    """Direct per-residue logit head for AF2BIND binder-pair features."""

    def __init__(self, c_in=5120, dropout=0.0, layer_norm=True):
        super().__init__()
        self.norm = nn.LayerNorm(c_in) if layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(c_in, 1)

    def forward(self, binder_features, batch):
        device = batch.device
        logits = torch.zeros(batch.shape[0], device=device)
        B = batch.max().item() + 1

        for b in range(B):
            if b >= len(binder_features) or binder_features[b] is None:
                continue
            node_mask = (batch == b)
            n_b = node_mask.sum().item()
            feat = binder_features[b]
            if feat.shape[0] != n_b:
                continue
            logits[node_mask] = self.linear(
                self.dropout(self.norm(feat))
            ).squeeze(-1)

        return logits


class GlobalPairFusion(nn.Module):
    """Attention-pooled AF2 pair row -> per-residue global summary."""

    def __init__(self, c_pair=128, c_proj=64):
        super().__init__()
        self.c_proj = c_proj
        self.pair_proj = nn.Linear(c_pair, c_proj)
        self.attn_linear = nn.Linear(c_proj, 1)

    def forward(self, pair_reprs, batch):
        """
        Args:
            pair_reprs: list of [Ni, Ni, C_pair] or None, one per protein
            batch:      [N_total] batch assignment
        Returns:
            [N_total, C_proj]
        """
        device = batch.device
        N_total = batch.shape[0]
        summary = torch.zeros(N_total, self.c_proj, device=device)
        B = batch.max().item() + 1

        for b in range(B):
            if b >= len(pair_reprs) or pair_reprs[b] is None:
                continue
            node_mask = (batch == b)
            n_b = node_mask.sum().item()
            pair = pair_reprs[b]
            if pair.shape[0] != n_b:
                continue

            proj = self.pair_proj(pair)
            attn_scores = self.attn_linear(proj).squeeze(-1)
            attn_weights = F.softmax(attn_scores, dim=-1)
            row_summary = torch.einsum('ij,ijk->ik', attn_weights, proj)
            summary[node_mask] = row_summary

        return summary


class EdgeConditioning(nn.Module):
    """Gated residual conditioning of edge features with AF2 pair info."""

    def __init__(self, c_pair=128, n_atom_basis=128):
        super().__init__()
        self.n_atom_basis = n_atom_basis
        self.pair_proj = nn.Linear(c_pair, n_atom_basis)
        self.gate_mlp = nn.Sequential(
            nn.Linear(n_atom_basis + 1, n_atom_basis),
            nn.SiLU(),
            nn.Linear(n_atom_basis, n_atom_basis),
            nn.Sigmoid(),
        )
        nn.init.xavier_uniform_(self.pair_proj.weight, gain=0.1)
        nn.init.zeros_(self.pair_proj.bias)
        nn.init.constant_(self.gate_mlp[2].bias, -2.0)

    def forward(self, t_ij, pair_reprs, edge_index, batch, local_idx, edge_dist):
        """
        Args:
            t_ij:       [E_total, H]
            pair_reprs: list of [Ni, Ni, C_pair] or None
            edge_index: [2, E_total]
            batch:      [N_total]
            local_idx:  [N_total]
            edge_dist:  [E_total]
        Returns:
            [E_total, H]  conditioned edge features
        """
        af2_edge = torch.zeros_like(t_ij)
        local_i = local_idx[edge_index[0]]
        local_j = local_idx[edge_index[1]]
        batch_edge = batch[edge_index[0]]
        B = batch.max().item() + 1

        for b in range(B):
            if b >= len(pair_reprs) or pair_reprs[b] is None:
                continue
            pair = pair_reprs[b]
            edge_mask = (batch_edge == b)
            if not edge_mask.any():
                continue
            li = local_i[edge_mask]
            lj = local_j[edge_mask]
            if li.max() >= pair.shape[0] or lj.max() >= pair.shape[1]:
                continue
            pair_ij = pair[li, lj]
            af2_edge[edge_mask] = self.pair_proj(pair_ij)

        gate_input = torch.cat([t_ij, edge_dist.unsqueeze(-1)], dim=-1)
        gate = self.gate_mlp(gate_input)
        return t_ij + gate * af2_edge


# ============================================================================
#  Section 12: Classifier
# ============================================================================

class ResidueClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ResidueLogitClassifier(nn.Module):
    """Classifier variant that returns logits for residual fusion."""

    def __init__(self, in_dim, hidden_dim=256, dropout=0.1, zero_init=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        if zero_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ============================================================================
#  Section 13: Main Model
# ============================================================================

class PockMonModel(nn.Module):
    """
    PockMon: equivariant backbone with AF2 pair representation fusion.

    External interface: identical to MQAModel (input [B,L,...], output [B,L]).
    Internal computation: PyG sparse format for efficiency.
    Coordinate unit: nm (consistent with MDTraj).
    """

    def __init__(
        self,
        cutoff: float = 1.5,
        n_atom_basis: int = 128,
        n_rbf: int = 32,
        n_pos_enc: int = 16,
        n_interactions: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        lmax: int = 2,
        max_z: int = 21,
        c_pair: int = 128,
        c_pair_proj: int = 64,
        pair_repr_dir: Optional[str] = None,
        binder_pair_dir: Optional[str] = None,
        combined_pair_dir: Optional[str] = None,
        c_binder_in: int = 5120,
        init_steerable_from_backbone: bool = True,
        use_pair_fusion: bool = True,
        use_edge_conditioning: bool = False,
        max_num_neighbors: int = 128,
        aggr: str = 'add',
        layernorm: str = 'layer',
        steerable_norm: str = 'layer',
        edge_updates: Union[bool, str] = False,
        scale_edge: bool = True,
        sep_htr: bool = True,
        sep_dir: bool = True,
        sep_tensor: bool = True,
        cls_hidden_dim: int = 256,
        fusion_strategy: str = "late",
        pair_logit_dropout: float = 0.0,
        pair_logit_layernorm: bool = True,
        residual_init_zero: bool = True,
        use_dihedral: bool = True,
        use_fusion_multiply: bool = True,
        dilated_sequence_edges: Optional[Sequence[int]] = None,
        edge_type_dim: int = 0,
    ):
        super().__init__()

        if fusion_strategy not in {"late", "residual", "gated_residual"}:
            raise ValueError(
                "fusion_strategy must be 'late', 'residual', or 'gated_residual'"
            )

        self.cutoff = cutoff
        self.n_atom_basis = n_atom_basis
        self.lmax = lmax
        self.init_steerable_from_backbone = init_steerable_from_backbone
        self.use_pair_fusion = use_pair_fusion
        self.use_edge_conditioning = use_edge_conditioning
        self.max_num_neighbors = max_num_neighbors
        self.c_pair_proj = c_pair_proj
        self.use_binder_pair = binder_pair_dir is not None
        self.use_combined_pair = combined_pair_dir is not None
        self.fusion_strategy = fusion_strategy
        self.use_dihedral = use_dihedral
        self.use_fusion_multiply = use_fusion_multiply
        self.dilated_sequence_edges = tuple(
            int(d) for d in (dilated_sequence_edges or []) if int(d) > 0
        )
        self.edge_type_dim = int(edge_type_dim) if self.dilated_sequence_edges else 0
        self.edge_type_embedding = (
            nn.Embedding(2, self.edge_type_dim)
            if self.edge_type_dim > 0 else None
        )
        self.use_residual_pair = (
            fusion_strategy in {"residual", "gated_residual"}
            and binder_pair_dir is not None
            and not combined_pair_dir
            and use_pair_fusion
        )

        equi_dim = (lmax + 1) ** 2 - 1
        self.equi_dim = equi_dim
        activation = F.silu

        # --- Node embedding ---
        self.A_na = nn.Embedding(max_z, n_atom_basis)
        self.dihedral_proj = nn.Linear(6, n_atom_basis)

        # --- Steerable initialization from backbone vectors ---
        self.vec_proj = nn.Linear(3, n_atom_basis, bias=False)
        nn.init.xavier_uniform_(self.vec_proj.weight, gain=0.1)

        # --- Radial basis ---
        self.radial_basis = ExpNormalSmearing(cutoff=cutoff, n_rbf=n_rbf)

        # --- Positional encoding ---
        self.n_pos_enc = n_pos_enc
        self.pos_enc = (
            SparsePositionalEncodings(n_pos_enc) if n_pos_enc > 0 else None
        )

        # --- NodeInit / EdgeInit (residue adapted) ---
        self.node_init = ResidueNodeInit(
            [n_atom_basis, n_atom_basis], n_rbf, cutoff, max_z=max_z,
            activation=activation, proj_ln="layer",
        )
        edge_basis_dim = n_rbf + n_pos_enc if n_pos_enc > 0 else n_rbf
        edge_basis_dim += self.edge_type_dim
        self.edge_init = ResidueEdgeInit(edge_basis_dim, n_atom_basis)

        # --- Spherical harmonics ---
        sh_irreps = e3nn.o3.Irreps.spherical_harmonics(lmax)
        self.sphere = e3nn.o3.SphericalHarmonics(
            sh_irreps, normalize=False, normalization="norm",
        )

        # --- Encoder ---
        self.encoder = PockMonEncoder(
            n_atom_basis=n_atom_basis, n_interactions=n_interactions,
            lmax=lmax, num_heads=num_heads, dropout=dropout,
            cutoff=cutoff, activation=activation, aggr=aggr,
            layernorm=layernorm, steerable_norm=steerable_norm,
            edge_updates=edge_updates, scale_edge=scale_edge,
            sep_htr=sep_htr, sep_dir=sep_dir, sep_tensor=sep_tensor,
        )

        # --- AF2 pair modules (fixbb, binder, or combined) ---
        self.pair_provider = None
        self.binder_provider = None
        self.combined_provider = None

        if combined_pair_dir:
            self.combined_provider = AF2CombinedPairProvider(combined_pair_dir)
        elif binder_pair_dir:
            self.binder_provider = AF2BinderPairProvider(binder_pair_dir)
        elif pair_repr_dir:
            self.pair_provider = AF2PairProvider(pair_repr_dir)

        self.global_pair_fusion = None
        self.binder_projection = None
        self.binder_logit_head = None

        if use_pair_fusion and combined_pair_dir:
            self.binder_projection = BinderPairProjection(
                c_in=c_binder_in, c_proj=c_pair_proj, dropout=dropout,
            )
            self.global_pair_fusion = GlobalPairFusion(
                c_pair=c_pair, c_proj=c_pair_proj,
            )
            self.local_to_pair_proj = nn.Linear(
                2 * n_atom_basis, 2 * c_pair_proj,
            )
        elif use_pair_fusion and binder_pair_dir:
            self.binder_projection = BinderPairProjection(
                c_in=c_binder_in, c_proj=c_pair_proj, dropout=dropout,
            )
            self.local_to_pair_proj = nn.Linear(2 * n_atom_basis, c_pair_proj)
            if self.use_residual_pair:
                self.binder_logit_head = BinderPairLogitHead(
                    c_in=c_binder_in,
                    dropout=pair_logit_dropout,
                    layer_norm=pair_logit_layernorm,
                )
        elif use_pair_fusion:
            self.global_pair_fusion = GlobalPairFusion(
                c_pair=c_pair, c_proj=c_pair_proj,
            )
            self.local_to_pair_proj = nn.Linear(2 * n_atom_basis, c_pair_proj)

        self.edge_conditioning = None
        if use_edge_conditioning and not binder_pair_dir and not combined_pair_dir:
            self.edge_conditioning = EdgeConditioning(
                c_pair=c_pair, n_atom_basis=n_atom_basis,
            )

        # --- Classifier ---
        if use_pair_fusion and combined_pair_dir:
            if use_fusion_multiply:
                cls_in_dim = 2 * n_atom_basis + 4 * c_pair_proj
            else:
                cls_in_dim = 2 * n_atom_basis + 2 * c_pair_proj
        elif use_pair_fusion:
            if use_fusion_multiply:
                cls_in_dim = 2 * n_atom_basis + 2 * c_pair_proj
            else:
                cls_in_dim = 2 * n_atom_basis + c_pair_proj
        else:
            cls_in_dim = 2 * n_atom_basis
        if self.use_residual_pair:
            self.classifier = ResidueLogitClassifier(
                cls_in_dim,
                hidden_dim=cls_hidden_dim,
                dropout=dropout,
                zero_init=residual_init_zero,
            )
            self.residual_gate = None
            if fusion_strategy == "gated_residual":
                self.residual_gate = nn.Sequential(
                    nn.Linear(cls_in_dim, cls_hidden_dim),
                    nn.SiLU(),
                    nn.Linear(cls_hidden_dim, 1),
                    nn.Sigmoid(),
                )
                nn.init.constant_(self.residual_gate[2].bias, -2.0)
        else:
            self.classifier = ResidueClassifier(
                cls_in_dim, hidden_dim=cls_hidden_dim, dropout=dropout,
            )
            self.residual_gate = None

    def forward(self, X, S, mask, train=False, res_level=False, meta=None):
        """
        Args:
            X:    [B, L, A, 3]  atom coordinates (nm, PocketMiner format)
            S:    [B, L]        amino acid types
            mask: [B, L]        valid residue mask (1=valid, 0=pad)
            train:     ignored (kept for interface compatibility)
            res_level: ignored (always outputs [B, L])
            meta: list[str] or None -- structureId per protein for binder
                  feature lookup; falls back to aatype matching if None.

        Returns:
            probs: [B, L]  per-residue pocket probability
        """
        B, L = mask.shape
        device = X.device

        # ====== 1. Compute geometric features in dense format ======
        X_ca = X[:, :, 1, :]
        dihedrals_dense = compute_dihedrals_dense(X)
        pseudo_cb_dense = compute_sidechains_dense(X)
        fwd_dense, rev_dense = compute_orientations_dense(X_ca)

        # ====== 2. Dense -> Sparse ======
        valid = mask.bool()
        pos = X_ca[valid]
        seq = S[valid]

        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, L)
        batch = batch_idx[valid]

        dihedrals = dihedrals_dense[valid]
        pseudo_cb = pseudo_cb_dense[valid]
        fwd = fwd_dense[valid]
        rev = rev_dense[valid]

        lengths = mask.sum(dim=1).long()
        local_idx = torch.cat(
            [torch.arange(l.item(), device=device) for l in lengths]
        )

        N_total = pos.shape[0]

        # ====== 3. Build cutoff graph ======
        edge_index, edge_dist, edge_vec = build_cutoff_graph(
            pos, batch, self.cutoff, loop=True,
            max_num_neighbors=self.max_num_neighbors,
        )
        edge_type = torch.zeros(
            edge_index.size(1), dtype=torch.long, device=device,
        )
        if self.dilated_sequence_edges:
            edge_index, edge_dist, edge_vec, edge_type = (
                augment_with_dilated_sequence_edges(
                    edge_index=edge_index,
                    edge_dist=edge_dist,
                    edge_vec=edge_vec,
                    pos=pos,
                    lengths=lengths,
                    dilations=self.dilated_sequence_edges,
                    edge_type=edge_type,
                )
            )

        # ====== 4. Feature construction & initialization ======
        h = self.A_na(seq)
        if self.use_dihedral:
            h = h + self.dihedral_proj(dihedrals)

        # RBF
        phi_r0_ij = self.radial_basis(edge_dist)

        # NodeInit
        h = self.node_init(seq, h, edge_index, edge_dist, phi_r0_ij)

        # EdgeInit: RBF + optional positional encoding
        edge_basis_parts = [phi_r0_ij]
        if self.pos_enc is not None:
            seq_diff = local_idx[edge_index[0]] - local_idx[edge_index[1]]
            edge_basis_parts.append(self.pos_enc(seq_diff))
        if self.edge_type_embedding is not None:
            edge_basis_parts.append(self.edge_type_embedding(edge_type))
        edge_basis = torch.cat(edge_basis_parts, dim=-1)
        t_ij = self.edge_init(edge_index, edge_basis, h)

        # Steerable X initialization (§7.2.2)
        X_steer = torch.zeros(
            N_total, self.equi_dim, self.n_atom_basis, device=device,
        )
        if self.init_steerable_from_backbone:
            vec_input = torch.stack([pseudo_cb, fwd, rev], dim=-1)
            X_steer[:, 0:3, :] = self.vec_proj(vec_input)

        # Edge geometric tensors (spherical harmonics)
        edge_vec_for_sh = edge_vec.clone()
        nonself = edge_index[0] != edge_index[1]
        norms = torch.norm(edge_vec_for_sh[nonself], dim=1, keepdim=True)
        edge_vec_for_sh[nonself] = edge_vec_for_sh[nonself] / (norms + 1e-8)
        rl_ij = self.sphere(edge_vec_for_sh)[:, 1:]

        # ====== 5. Optional: AF2 pair features ======
        pair_reprs = None
        binder_features = None

        if self.combined_provider is not None:
            binder_features, pair_reprs = (
                self.combined_provider.get_combined_features(
                    S, mask, device, meta=meta,
                )
            )
        elif self.binder_provider is not None:
            binder_features = self.binder_provider.get_binder_features(
                S, mask, device, meta=meta,
            )
        elif self.pair_provider is not None:
            pair_reprs = self.pair_provider.get_pair_repr(S, mask, device)

        if (pair_reprs is not None and self.edge_conditioning is not None
                and self.use_edge_conditioning):
            t_ij = self.edge_conditioning(
                t_ij, pair_reprs, edge_index, batch, local_idx, edge_dist,
            )

        # ====== 6. Equivariant message passing ======
        h = h.unsqueeze(1)
        num_edges = scatter(
            torch.ones_like(edge_dist), edge_index[0],
            dim=0, dim_size=N_total, reduce="sum",
        )
        n_edges = num_edges[edge_index[0]]

        h, X_steer = self.encoder(
            h, X_steer, t_ij, rl_ij, edge_index, edge_dist, n_edges,
        )

        # ====== 7. Readout: scalar + steerable norm pool ======
        steerable_inv = torch.sqrt(
            torch.sum(X_steer ** 2, dim=1) + 1e-8
        )
        local_emb = torch.cat([h, steerable_inv], dim=-1)

        # ====== 8. AF2 late fusion (combined, binder, or fixbb) ======
        if self.use_pair_fusion and self.use_combined_pair:
            # Combined mode: dual-branch fusion
            if binder_features is not None and any(
                f is not None for f in binder_features
            ):
                binder_summary = self.binder_projection(
                    binder_features, batch,
                )
            else:
                binder_summary = torch.zeros(
                    N_total, self.c_pair_proj, device=device,
                )
            if pair_reprs is not None and any(
                p is not None for p in pair_reprs
            ):
                pair_summary = self.global_pair_fusion(pair_reprs, batch)
            else:
                pair_summary = torch.zeros(
                    N_total, self.c_pair_proj, device=device,
                )
            combined_summary = torch.cat(
                [binder_summary, pair_summary], dim=-1,
            )
            if self.use_fusion_multiply:
                local_proj = self.local_to_pair_proj(local_emb)
                fused = torch.cat([
                    local_emb, combined_summary,
                    local_proj * combined_summary,
                ], dim=-1)
            else:
                fused = torch.cat([local_emb, combined_summary], dim=-1)
        elif self.use_pair_fusion and self.binder_projection is not None:
            if binder_features is not None and any(
                f is not None for f in binder_features
            ):
                global_summary = self.binder_projection(
                    binder_features, batch,
                )
            else:
                global_summary = torch.zeros(
                    N_total, self.c_pair_proj, device=device,
                )
            if self.use_fusion_multiply:
                local_proj = self.local_to_pair_proj(local_emb)
                fused = torch.cat([
                    local_emb, global_summary, local_proj * global_summary,
                ], dim=-1)
            else:
                fused = torch.cat([local_emb, global_summary], dim=-1)
        elif self.use_pair_fusion and self.global_pair_fusion is not None:
            if pair_reprs is not None and any(
                p is not None for p in pair_reprs
            ):
                global_summary = self.global_pair_fusion(pair_reprs, batch)
            else:
                global_summary = torch.zeros(
                    N_total, self.c_pair_proj, device=device,
                )
            if self.use_fusion_multiply:
                local_proj = self.local_to_pair_proj(local_emb)
                fused = torch.cat([
                    local_emb, global_summary, local_proj * global_summary,
                ], dim=-1)
            else:
                fused = torch.cat([local_emb, global_summary], dim=-1)
        else:
            fused = local_emb

        # ====== 9. Classification ======
        if self.use_residual_pair and self.binder_logit_head is not None:
            pair_logits = self.binder_logit_head(binder_features, batch)
            delta_logits = self.classifier(fused)
            if self.residual_gate is not None:
                gate = self.residual_gate(fused).squeeze(-1)
                delta_logits = gate * delta_logits
            probs_sparse = torch.sigmoid(pair_logits + delta_logits)
        else:
            probs_sparse = self.classifier(fused)

        # ====== 10. Sparse -> Dense ======
        probs = torch.zeros(B, L, device=device)
        probs[valid] = probs_sparse

        return probs
