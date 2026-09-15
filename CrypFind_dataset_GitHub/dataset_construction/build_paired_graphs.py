#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import io
import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import zstandard as zstd
from Bio import Align
from Bio.PDB import MMCIFParser, PDBParser
from torch_geometric.data import Data

try:
    import gemmi  # type: ignore

    HAS_GEMMI = True
except Exception:
    HAS_GEMMI = False

try:
    from scipy.spatial import cKDTree  # type: ignore

    HAS_CKDTREE = True
except Exception:
    HAS_CKDTREE = False

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
    "SEC": "C",
    "PYL": "K",
}

AA1_TO_IDX = {
    aa: i
    for i, aa in enumerate(
        ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
    )
}

BACKBONE_ATOMS = ("N", "CA", "C", "O")
REQUIRED_BACKBONE_ATOMS = ("N", "CA", "C")


@dataclass
class ResidueRecord:
    seq_idx: int
    aa1: str
    ca: Optional[np.ndarray]
    atoms: np.ndarray
    backbone: np.ndarray
    backbone_mask: np.ndarray


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build PyG Apo-Holo interface graphs.")
    p.add_argument("--csv", default="AlphaFold_Data/refined_homo_pretrain_final.csv")
    p.add_argument("--apo-dir", default="AlphaFold_Data/apo_monomer_structures")
    p.add_argument("--holo-dir", default="AlphaFold_Data/holo_complex_structures")
    p.add_argument("--out-dir", default="processed_pairs")
    p.add_argument("--log-csv", default="build_pyg_graphs_skips.csv")
    p.add_argument("--nproc", type=int, default=max((os.cpu_count() or 2) - 1, 1))
    p.add_argument("--chunksize", type=int, default=16)
    p.add_argument("--limit", type=int, default=0, help="Debug only: process first N rows.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--interface-cutoff", type=float, default=5.0)
    p.add_argument("--apo-edge-cutoff", type=float, default=8.0)
    p.add_argument("--break-delta", type=float, default=3.0)
    p.add_argument("--min-interface", type=int, default=5)
    p.add_argument("--min-nodes", type=int, default=5)
    return p.parse_args()


def read_pairs(csv_path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"uniprotAccession", "modelEntityId"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")
        for row in reader:
            uniprot = (row.get("uniprotAccession") or "").strip()
            model_id = (row.get("modelEntityId") or "").strip()
            if uniprot and model_id:
                pairs.append((uniprot, model_id))
    return pairs


def read_zst_text(path: Path) -> str:
    dctx = zstd.ZstdDecompressor()
    with path.open("rb") as f:
        with dctx.stream_reader(f) as reader:
            raw = reader.read()
    return raw.decode("utf-8", errors="replace")


def read_plain_text(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return f.read()


def parse_structure_text(text: str, fmt: str):
    fmt = fmt.lower()
    if HAS_GEMMI:
        try:
            if fmt == "cif":
                doc = gemmi.cif.read_string(text)
                return ("gemmi", gemmi.make_structure_from_block(doc.sole_block()))
            if fmt == "pdb":
                return ("gemmi", gemmi.read_pdb_string(text))
        except Exception:
            pass

    if fmt == "cif":
        return ("biopython", MMCIFParser(QUIET=True).get_structure("structure", io.StringIO(text)))
    if fmt == "pdb":
        return ("biopython", PDBParser(QUIET=True).get_structure("structure", io.StringIO(text)))
    raise ValueError(f"Unsupported structure format: {fmt}")


def load_structure(path: Path):
    suffixes = [s.lower() for s in path.suffixes]
    is_zst = ".zst" in suffixes
    fmt = "pdb" if ".pdb" in suffixes else "cif"
    text = read_zst_text(path) if is_zst else read_plain_text(path)
    return parse_structure_text(text, fmt)


def build_chain_records(parsed) -> dict[str, list[ResidueRecord]]:
    backend, structure = parsed
    if backend == "gemmi":
        return build_chain_records_gemmi(structure)
    return build_chain_records_biopython(structure)


def build_backbone(atom_coords: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    backbone = np.zeros((len(BACKBONE_ATOMS), 3), dtype=np.float32)
    mask = np.zeros((len(BACKBONE_ATOMS),), dtype=np.bool_)
    for i, atom_name in enumerate(BACKBONE_ATOMS):
        coord = atom_coords.get(atom_name)
        if coord is None:
            continue
        backbone[i] = coord.astype(np.float32, copy=False)
        mask[i] = True
    return backbone, mask


def has_required_backbone(record: ResidueRecord) -> bool:
    required_indexes = [BACKBONE_ATOMS.index(atom_name) for atom_name in REQUIRED_BACKBONE_ATOMS]
    return bool(np.all(record.backbone_mask[required_indexes]))


def build_chain_records_gemmi(structure) -> dict[str, list[ResidueRecord]]:
    chains: dict[str, list[ResidueRecord]] = {}
    model = structure[0]
    for chain in model:
        chain_id = str(chain.name).strip() or "?"
        records: list[ResidueRecord] = []
        seq_idx = 0
        for res in chain:
            aa1 = AA3_TO_1.get(str(res.name).upper())
            if aa1 is None:
                continue

            coords: list[list[float]] = []
            atom_coords: dict[str, np.ndarray] = {}
            for atom in res:
                xyz = np.asarray([float(atom.pos.x), float(atom.pos.y), float(atom.pos.z)], dtype=np.float32)
                coords.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])
                atom_name = str(atom.name).strip().upper()
                atom_coords.setdefault(atom_name, xyz)

            if not coords:
                continue
            backbone, backbone_mask = build_backbone(atom_coords)
            records.append(
                ResidueRecord(
                    seq_idx=seq_idx,
                    aa1=aa1,
                    ca=atom_coords.get("CA"),
                    atoms=np.asarray(coords, dtype=np.float32),
                    backbone=backbone,
                    backbone_mask=backbone_mask,
                )
            )
            seq_idx += 1

        if records:
            chains[chain_id] = records
    return chains


def build_chain_records_biopython(structure) -> dict[str, list[ResidueRecord]]:
    chains: dict[str, list[ResidueRecord]] = {}
    model = next(structure.get_models())
    for chain in model:
        chain_id = str(chain.id).strip() or "?"
        records: list[ResidueRecord] = []
        seq_idx = 0
        for res in chain:
            if not isinstance(res.id, tuple) or res.id[0] != " ":
                continue
            aa1 = AA3_TO_1.get(str(res.resname).upper())
            if aa1 is None:
                continue

            coords: list[list[float]] = []
            atom_coords: dict[str, np.ndarray] = {}
            for atom in res:
                xyz = atom.coord.astype(np.float32)
                coords.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])
                atom_name = str(atom.name).strip().upper()
                atom_coords.setdefault(atom_name, np.asarray(xyz, dtype=np.float32))

            if not coords:
                continue
            backbone, backbone_mask = build_backbone(atom_coords)
            records.append(
                ResidueRecord(
                    seq_idx=seq_idx,
                    aa1=aa1,
                    ca=atom_coords.get("CA"),
                    atoms=np.asarray(coords, dtype=np.float32),
                    backbone=backbone,
                    backbone_mask=backbone_mask,
                )
            )
            seq_idx += 1

        if records:
            chains[chain_id] = records
    return chains


def sequence_from_records(records: Sequence[ResidueRecord]) -> str:
    return "".join(r.aa1 for r in records)


def interface_indices(chain_a: Sequence[ResidueRecord], chain_b: Sequence[ResidueRecord], cutoff: float) -> set[int]:
    b_atoms = [r.atoms for r in chain_b if r.atoms.size > 0]
    if not b_atoms:
        raise ValueError("chain B contains no atoms")
    b_all = np.concatenate(b_atoms, axis=0)

    out: set[int] = set()
    if HAS_CKDTREE:
        tree = cKDTree(b_all)
        for r in chain_a:
            if r.atoms.size == 0:
                continue
            dmin, _ = tree.query(r.atoms, k=1)
            if float(np.min(dmin)) < cutoff:
                out.add(r.seq_idx)
        return out

    cutoff2 = cutoff * cutoff
    for r in chain_a:
        if r.atoms.size == 0:
            continue
        diff = r.atoms[:, None, :] - b_all[None, :, :]
        if float(np.min(np.sum(diff * diff, axis=2))) < cutoff2:
            out.add(r.seq_idx)
    return out


def build_apo_to_holo_mapping(apo_seq: str, holo_seq: str) -> dict[int, int]:
    if not apo_seq or not holo_seq:
        raise ValueError("empty sequence")

    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5

    alignments = aligner.align(apo_seq, holo_seq)
    if len(alignments) == 0:
        raise ValueError("no sequence alignment")

    mapping: dict[int, int] = {}
    apo_blocks, holo_blocks = alignments[0].aligned
    for (a0, a1), (h0, h1) in zip(apo_blocks, holo_blocks):
        for offset in range(min(int(a1 - a0), int(h1 - h0))):
            mapping[int(a0 + offset)] = int(h0 + offset)
    return mapping


def find_apo_path(apo_dir: Path, uniprot: str) -> Optional[Path]:
    candidates = [apo_dir / f"AF-{uniprot}-F1-model_v{version}.cif" for version in range(6, 0, -1)]
    candidates.extend(
        [
            apo_dir / f"AF-{uniprot}-F1-model_v6.cif.zst",
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    return None


def find_holo_path(holo_dir: Path, model_id: str) -> Optional[Path]:
    model_stem = model_id if model_id.endswith("-model_v1") else f"{model_id}-model_v1"
    candidates = [
        holo_dir / f"{model_id}.cif.zst",
        holo_dir / f"{model_id}.pdb.zst",
        holo_dir / f"{model_stem}.cif.zst",
        holo_dir / f"{model_stem}.pdb.zst",
        holo_dir / f"{model_id}.cif",
        holo_dir / f"{model_id}.pdb",
        holo_dir / f"{model_stem}.cif",
        holo_dir / f"{model_stem}.pdb",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def build_edges(apo_pos: np.ndarray, holo_pos: np.ndarray, apo_cutoff: float, break_delta: float) -> tuple[torch.Tensor, torch.Tensor]:
    n = apo_pos.shape[0]
    edge_src: list[int] = []
    edge_dst: list[int] = []
    labels: list[int] = []

    for i in range(n):
        for j in range(i + 1, n):
            apo_dist = float(np.linalg.norm(apo_pos[i] - apo_pos[j]))
            if apo_dist >= apo_cutoff:
                continue
            holo_dist = float(np.linalg.norm(holo_pos[i] - holo_pos[j]))
            label = 1 if (holo_dist - apo_dist) > break_delta else 0
            edge_src.extend([i, j])
            edge_dst.extend([j, i])
            labels.extend([label, label])

    if not edge_src:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long)
    return torch.tensor([edge_src, edge_dst], dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def build_one_graph(uniprot: str, model_id: str, args: argparse.Namespace) -> tuple[str, str]:
    apo_dir = Path(args.apo_dir)
    holo_dir = Path(args.holo_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{uniprot}_{model_id}.pt"
    if out_path.exists() and not args.overwrite:
        return "skipped_existing", ""

    apo_path = find_apo_path(apo_dir, uniprot)
    if apo_path is None:
        return "missing_apo", ""
    holo_path = find_holo_path(holo_dir, model_id)
    if holo_path is None:
        return "missing_holo", ""

    apo_chains = build_chain_records(load_structure(apo_path))
    holo_chains = build_chain_records(load_structure(holo_path))
    if not apo_chains:
        return "parse_error", "no valid apo chains"
    if "A" not in holo_chains or "B" not in holo_chains:
        return "bad_holo_chains", "missing holo chain A or B"

    apo_chain_id = "A" if "A" in apo_chains else max(apo_chains, key=lambda c: len(apo_chains[c]))
    apo_records = apo_chains[apo_chain_id]
    holo_a = holo_chains["A"]
    holo_b = holo_chains["B"]

    iface = interface_indices(holo_a, holo_b, args.interface_cutoff)
    if len(iface) < args.min_interface:
        return "too_few_interface", str(len(iface))

    apo_seq = sequence_from_records(apo_records)
    holo_seq = sequence_from_records(holo_a)
    apo_to_holo = build_apo_to_holo_mapping(apo_seq, holo_seq)
    holo_to_apo = {h: a for a, h in apo_to_holo.items()}

    node_apo_idx: list[int] = []
    node_holo_idx: list[int] = []
    sequence_ids: list[int] = []
    for h_idx in sorted(iface):
        a_idx = holo_to_apo.get(h_idx)
        if a_idx is None:
            continue
        if a_idx < 0 or a_idx >= len(apo_records) or h_idx >= len(holo_a):
            continue
        if apo_records[a_idx].ca is None or holo_a[h_idx].ca is None:
            continue
        if not has_required_backbone(apo_records[a_idx]) or not has_required_backbone(holo_a[h_idx]):
            continue
        aa_idx = AA1_TO_IDX.get(apo_records[a_idx].aa1)
        if aa_idx is None:
            continue
        node_apo_idx.append(a_idx)
        node_holo_idx.append(h_idx)
        sequence_ids.append(aa_idx)

    if len(node_apo_idx) < args.min_nodes:
        return "too_few_nodes", str(len(node_apo_idx))

    apo_pos = np.stack([apo_records[i].ca for i in node_apo_idx], axis=0).astype(np.float32)
    holo_pos = np.stack([holo_a[i].ca for i in node_holo_idx], axis=0).astype(np.float32)
    apo_backbone = np.stack([apo_records[i].backbone for i in node_apo_idx], axis=0).astype(np.float32)
    holo_backbone = np.stack([holo_a[i].backbone for i in node_holo_idx], axis=0).astype(np.float32)
    apo_backbone_mask = np.stack([apo_records[i].backbone_mask for i in node_apo_idx], axis=0).astype(np.bool_)
    holo_backbone_mask = np.stack([holo_a[i].backbone_mask for i in node_holo_idx], axis=0).astype(np.bool_)
    edge_index, edge_label = build_edges(apo_pos, holo_pos, args.apo_edge_cutoff, args.break_delta)

    data = Data(
        apo_pos=torch.tensor(apo_pos, dtype=torch.float32),
        holo_pos=torch.tensor(holo_pos, dtype=torch.float32),
        apo_backbone=torch.tensor(apo_backbone, dtype=torch.float32),
        holo_backbone=torch.tensor(holo_backbone, dtype=torch.float32),
        apo_backbone_mask=torch.tensor(apo_backbone_mask, dtype=torch.bool),
        holo_backbone_mask=torch.tensor(holo_backbone_mask, dtype=torch.bool),
        edge_index=edge_index,
        edge_label=edge_label,
        sequence=torch.tensor(sequence_ids, dtype=torch.long),
        num_nodes=len(sequence_ids),
    )
    torch.save(data, out_path)
    return "ok", ""


_ARGS: Optional[argparse.Namespace] = None


def init_worker(args: argparse.Namespace) -> None:
    global _ARGS
    _ARGS = args


def process_pair(pair: tuple[str, str]) -> tuple[str, str, str, str]:
    assert _ARGS is not None
    uniprot, model_id = pair
    try:
        status, message = build_one_graph(uniprot, model_id, _ARGS)
        return status, uniprot, model_id, message
    except Exception as exc:
        return "exception", uniprot, model_id, str(exc).replace("\n", " ")


def write_skip_log(log_path: Path, rows: list[tuple[str, str, str, str]]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["status", "uniprotAccession", "modelEntityId", "message"])
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = read_pairs(csv_path)
    if args.limit > 0:
        pairs = pairs[: args.limit]

    print(f"Total pairs: {len(pairs):,}")
    print(f"Workers: {args.nproc}")
    print(f"Output dir: {out_dir.resolve()}")

    counts: dict[str, int] = {}
    skip_rows: list[tuple[str, str, str, str]] = []

    iterator: Iterable[tuple[str, str, str, str]]
    with mp.Pool(processes=args.nproc, initializer=init_worker, initargs=(args,)) as pool:
        iterator = pool.imap_unordered(process_pair, pairs, chunksize=args.chunksize)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(pairs), desc="Building PyG graphs")

        for status, uniprot, model_id, message in iterator:
            counts[status] = counts.get(status, 0) + 1
            if status not in {"ok", "skipped_existing"}:
                skip_rows.append((status, uniprot, model_id, message))

    write_skip_log(Path(args.log_csv), skip_rows)

    print("\n========== Build Summary ==========")
    for status in sorted(counts):
        print(f"{status}: {counts[status]:,}")
    print(f"Skip/error log: {Path(args.log_csv).resolve()}")


if __name__ == "__main__":
    main()
