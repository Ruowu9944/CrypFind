"""
PocketMon-backbone pretraining model for apo-holo pair graphs.

The pretraining targets are unchanged:
  - residue-level apo/holo InfoNCE
  - apo-edge breakage prediction

This wrapper reuses the same PockMonModel backbone class used by the downstream
PocketMon fine-tuning code. The pair-fusion modules are disabled during
pretraining because the apo-holo pair dataset does not contain AF2 pair features;
the shared geometric backbone keys still align with the fine-tuning model.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import to_dense_batch


from crypfind.backbone import PockMonModel  # noqa: E402


@dataclass
class EdgeGCLConfig:
    """Compatibility config consumed by train_pretrain.py.

    Historical argument names are kept where possible. The defaults mirror
    PocketMon's combined fine-tuning config so pretrained backbone weights can
    be loaded into that model with minimal shape mismatch.
    """

    num_aa_types: int = 21
    num_layers: int = 4
    scalar_channels: int = 128
    vector_channels: int = 16
    tensor_channels: int = 0

    edge_dim: int = 64
    radial_hidden_dim: int = 128
    edge_hidden_dim: int = 256

    dropout: float = 0.1
    temperature: float = 0.1
    contrastive_proj_dim: int = 128

    cutoff: float = 1.5
    n_rbf: int = 32
    n_pos_enc: int = 16
    num_heads: int = 4
    lmax: int = 2
    c_pair: int = 128
    c_pair_proj: int = 64
    max_num_neighbors: int = 128
    cls_hidden_dim: int = 256
    edge_updates: bool = False

    k_neighbors: int = 30
    edge_vector_channels: int = 1
    rbf_d_max: float = 20.0


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(max(1, num_layers - 1)):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class PocketMonResidueEncoder(nn.Module):
    """Expose PockMonModel's classifier input as residue embeddings."""

    def __init__(self, cfg: EdgeGCLConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.output_dim = 2 * cfg.scalar_channels
        self._last_classifier_input: Optional[Tensor] = None

        self.backbone = PockMonModel(
            cutoff=cfg.cutoff,
            n_atom_basis=cfg.scalar_channels,
            n_rbf=cfg.n_rbf,
            n_pos_enc=cfg.n_pos_enc,
            n_interactions=cfg.num_layers,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            lmax=cfg.lmax,
            max_z=cfg.num_aa_types,
            c_pair=cfg.c_pair,
            c_pair_proj=cfg.c_pair_proj,
            pair_repr_dir=None,
            binder_pair_dir=None,
            combined_pair_dir=None,
            init_steerable_from_backbone=True,
            use_pair_fusion=False,
            use_edge_conditioning=False,
            max_num_neighbors=cfg.max_num_neighbors,
            layernorm="layer",
            steerable_norm="layer",
            edge_updates=cfg.edge_updates,
            sep_htr=True,
            sep_dir=True,
            sep_tensor=True,
            cls_hidden_dim=cfg.cls_hidden_dim,
            use_fusion_multiply=False,
        )
        self.backbone.classifier.register_forward_hook(self._capture_classifier_input)

    def _capture_classifier_input(
        self,
        _module: nn.Module,
        inputs: Tuple[Tensor, ...],
        _output: Tensor,
    ) -> None:
        self._last_classifier_input = inputs[0]

    def forward(
        self,
        pos: Tensor,
        sequence: Tensor,
        batch: Optional[Tensor] = None,
        backbone: Optional[Tensor] = None,
        backbone_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if batch is None:
            batch = pos.new_zeros((pos.size(0),), dtype=torch.long)

        ca_dense, residue_mask = to_dense_batch(pos, batch=batch)
        seq_dense, _ = to_dense_batch(
            sequence.clamp(min=0, max=self.cfg.num_aa_types - 1),
            batch=batch,
            fill_value=0,
        )

        if backbone is not None:
            x, backbone_residue_mask = to_dense_batch(backbone, batch=batch)
            residue_mask = residue_mask & backbone_residue_mask
            if backbone_mask is not None:
                atom_mask, _ = to_dense_batch(
                    backbone_mask.bool(),
                    batch=batch,
                    fill_value=False,
                )
                x = x * atom_mask.unsqueeze(-1).to(x.dtype)
        else:
            x = ca_dense[:, :, None, :].expand(-1, -1, 4, -1).clone()

        # PDB/mmCIF coordinates are stored in Angstrom in the processed pair
        # files; PockMonModel expects nm and uses a 1.5 nm cutoff graph.
        x_nm = x / 10.0

        self._last_classifier_input = None
        if x_nm.is_cuda:
            with torch.amp.autocast("cuda", enabled=False):
                _ = self.backbone(
                    x_nm.float(),
                    seq_dense.long(),
                    residue_mask.float(),
                    train=False,
                    res_level=True,
                    meta=None,
                )
        else:
            _ = self.backbone(
                x_nm.float(),
                seq_dense.long(),
                residue_mask.float(),
                train=False,
                res_level=True,
                meta=None,
            )
        if self._last_classifier_input is None:
            raise RuntimeError("Failed to capture PockMonModel residue embeddings")
        return self._last_classifier_input


class ContrastiveHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int, temperature: float) -> None:
        super().__init__()
        self.projector = MLP(in_dim, in_dim, proj_dim, num_layers=2, dropout=0.0)
        self.temperature = temperature

    def forward(self, z_apo: Tensor, z_holo: Tensor) -> Tensor:
        q = F.normalize(self.projector(z_apo), dim=-1)
        k = F.normalize(self.projector(z_holo), dim=-1)
        logits = q @ k.t() / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_ab = F.cross_entropy(logits, labels)
        loss_ba = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_ab + loss_ba)


class EdgePredictionHead(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.mlp = MLP(
            in_dim=4 * node_dim + 1,
            hidden_dim=hidden_dim,
            out_dim=1,
            num_layers=3,
            dropout=dropout,
        )

    def forward(self, z: Tensor, pos: Tensor, edge_index: Tensor) -> Tensor:
        if edge_index.numel() == 0:
            return z.new_empty((0,))
        src, dst = edge_index
        dist_nm = (pos[src] - pos[dst]).norm(dim=-1, keepdim=True) / 10.0
        edge_feat = torch.cat(
            [z[src], z[dst], torch.abs(z[src] - z[dst]), z[src] * z[dst], dist_nm],
            dim=-1,
        )
        return self.mlp(edge_feat).squeeze(-1)

    @staticmethod
    def predict_proba(edge_logits: Tensor) -> Tensor:
        return torch.sigmoid(edge_logits)


class EdgeGCLModel(nn.Module):
    """Compatibility wrapper whose backbone is PocketMon PockMonModel."""

    def __init__(self, cfg: Optional[EdgeGCLConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else EdgeGCLConfig()
        self.encoder = PocketMonResidueEncoder(self.cfg)
        self.contrastive_head = ContrastiveHead(
            in_dim=self.encoder.output_dim,
            proj_dim=self.cfg.contrastive_proj_dim,
            temperature=self.cfg.temperature,
        )
        self.edge_head = EdgePredictionHead(
            node_dim=self.encoder.output_dim,
            hidden_dim=self.cfg.edge_hidden_dim,
            dropout=self.cfg.dropout,
        )
        self.pseudo_pocket_head = MLP(
            in_dim=self.encoder.output_dim,
            hidden_dim=self.cfg.edge_hidden_dim,
            out_dim=1,
            num_layers=3,
            dropout=self.cfg.dropout,
        )

    def forward(
        self,
        apo_pos: Tensor,
        holo_pos: Tensor,
        edge_index: Tensor,
        sequence: Tensor,
        batch: Optional[Tensor] = None,
        apo_backbone: Optional[Tensor] = None,
        holo_backbone: Optional[Tensor] = None,
        apo_backbone_mask: Optional[Tensor] = None,
        holo_backbone_mask: Optional[Tensor] = None,
        **_: object,
    ) -> Dict[str, Tensor]:
        z_apo = self.encoder(
            apo_pos,
            sequence,
            batch=batch,
            backbone=apo_backbone,
            backbone_mask=apo_backbone_mask,
        )
        z_holo = self.encoder(
            holo_pos,
            sequence,
            batch=batch,
            backbone=holo_backbone,
            backbone_mask=holo_backbone_mask,
        )
        edge_logits = self.edge_head(z_apo, apo_pos, edge_index)
        return {
            "z_apo": z_apo,
            "z_holo": z_holo,
            "edge_logits": edge_logits,
        }

    def info_nce_loss(self, z_apo: Tensor, z_holo: Tensor) -> Tensor:
        return self.contrastive_head(z_apo, z_holo)

    def forward_edge_only(
        self,
        apo_pos: Tensor,
        edge_index: Tensor,
        sequence: Tensor,
        batch: Optional[Tensor] = None,
        apo_backbone: Optional[Tensor] = None,
        apo_backbone_mask: Optional[Tensor] = None,
        **_: object,
    ) -> Tensor:
        z_apo = self.encoder(
            apo_pos,
            sequence,
            batch=batch,
            backbone=apo_backbone,
            backbone_mask=apo_backbone_mask,
        )
        return self.edge_head(z_apo, apo_pos, edge_index)

    def forward_pseudo(
        self,
        apo_pos: Tensor,
        sequence: Tensor,
        batch: Optional[Tensor] = None,
        apo_backbone: Optional[Tensor] = None,
        apo_backbone_mask: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
        compute_edge: bool = False,
        **_: object,
    ) -> Tuple[Tensor, Tensor]:
        z_apo = self.encoder(
            apo_pos,
            sequence,
            batch=batch,
            backbone=apo_backbone,
            backbone_mask=apo_backbone_mask,
        )
        pseudo_logits = self.pseudo_pocket_head(z_apo).squeeze(-1)
        if compute_edge:
            if edge_index is None:
                raise ValueError("edge_index is required when compute_edge=True")
            edge_logits = self.edge_head(z_apo, apo_pos, edge_index)
        else:
            edge_logits = pseudo_logits.new_empty((0,))
        return pseudo_logits, edge_logits

    def forward_train(
        self,
        apo_pos: Tensor,
        holo_pos: Tensor,
        edge_index: Tensor,
        sequence: Tensor,
        batch: Optional[Tensor] = None,
        apo_backbone: Optional[Tensor] = None,
        holo_backbone: Optional[Tensor] = None,
        apo_backbone_mask: Optional[Tensor] = None,
        holo_backbone_mask: Optional[Tensor] = None,
        compute_contrastive: bool = True,
        **kwargs: object,
    ) -> Tuple[Tensor, Tensor]:
        if not compute_contrastive:
            edge_logits = self.forward_edge_only(
                apo_pos=apo_pos,
                edge_index=edge_index,
                sequence=sequence,
                batch=batch,
                apo_backbone=apo_backbone,
                apo_backbone_mask=apo_backbone_mask,
                **kwargs,
            )
            return edge_logits.new_zeros(()), edge_logits
        out = self.forward(
            apo_pos=apo_pos,
            holo_pos=holo_pos,
            edge_index=edge_index,
            sequence=sequence,
            batch=batch,
            apo_backbone=apo_backbone,
            holo_backbone=holo_backbone,
            apo_backbone_mask=apo_backbone_mask,
            holo_backbone_mask=holo_backbone_mask,
            **kwargs,
        )
        return self.info_nce_loss(out["z_apo"], out["z_holo"]), out["edge_logits"]

    def pockmon_backbone_state_dict(self) -> Dict[str, Tensor]:
        return self.encoder.backbone.state_dict()


__all__ = [
    "EdgeGCLConfig",
    "EdgeGCLModel",
    "PocketMonResidueEncoder",
    "ContrastiveHead",
    "EdgePredictionHead",
]
