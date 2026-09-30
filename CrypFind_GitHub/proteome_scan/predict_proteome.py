"""
Predict per-residue P(bind) across the human proteome using a trained
PockMon model with AF2 binder pair features.

Reads AlphaFold PDBs and pre-extracted binder pair features, runs PockMon
inference, and writes per-protein CSV predictions.

Output per protein:  {output_dir}/{UniProt_2char_suffix}/{UniProt_ID}-F1-model_v6.csv
  Columns: rank, chain, resi, resn, p(bind), arr_i, plddt

Usage:
    # Full proteome (single GPU)
    python predict_proteome.py --skip-existing

    # 16-way parallel sharding
    python predict_proteome.py --skip-existing --num-shards 16 --shard-id 0

    # Quick smoke test
    python predict_proteome.py --max-proteins 5
"""

import argparse
import gc
import json
import os
import re
import sys
import tempfile
import time

import numpy as np
import torch

SCAN_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCAN_DIR)
sys.path.insert(0, PROJECT_ROOT)

from training.train_pocketminer import make_model, get_num_atoms
from pockmon.utils import load_checkpoint

NONSTANDARD_TO_STANDARD = {
    "MSE": "MET", "PCA": "GLU", "CME": "CYS", "CSO": "CYS",
    "SEP": "SER", "TPO": "THR", "PTR": "TYR", "HYP": "PRO",
    "MLY": "LYS", "SAC": "SER", "DAL": "ALA", "AIB": "ALA",
    "CYM": "CYS", "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
    "HSE": "HIS", "HSD": "HIS", "HSP": "HIS",
}

AF_PDB_PATTERN = re.compile(r"^AF-(.+)-F(\d+)-model_v(\d+)\.pdb$")

DEFAULT_CONFIG = os.path.join(
    PROJECT_ROOT, "configs", "af2bind.json",
)
DEFAULT_CHECKPOINT = os.path.join(
    PROJECT_ROOT, "outputs", "af2bind", "checkpoints", "best_model.pt",
)
DEFAULT_PDB_DIR = os.path.join(SCAN_DIR, "proteome_pdbs")
DEFAULT_BINDER_DIR = os.path.join(SCAN_DIR, "proteome_binder_pair")
DEFAULT_OUTPUT_DIR = os.path.join(SCAN_DIR, "preds")


# ============================================================================
#  Lazy binder pair provider (avoids loading 23k npz into RAM)
# ============================================================================

class LazyBinderPairProvider:
    """Drop-in replacement for AF2BinderPairProvider that loads features
    on demand from disk instead of pre-loading everything into memory."""

    def __init__(self, binder_dir):
        self.binder_dir = binder_dir
        self.by_id = {}
        self.by_seq = {}

    def _load_one(self, sid):
        """Load a single npz file by structure ID (lazy, cached in self.by_id)."""
        if sid in self.by_id:
            return self.by_id[sid]

        fpath = os.path.join(self.binder_dir, f"{sid}.npz")
        if not os.path.isfile(fpath):
            self.by_id[sid] = None
            return None

        data = np.load(fpath)
        feat = data["features"]
        aatype = data["aatype"] if "aatype" in data else None
        entry = (feat, aatype)
        self.by_id[sid] = entry
        return entry

    def evict(self, sid):
        """Remove a cached entry to free memory."""
        self.by_id.pop(sid, None)

    @staticmethod
    def _align_features(binder_feat, binder_aatype, cache_seq):
        """Align binder features to cache sequence via sliding-window match."""
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

        if binder_len > cache_len:
            best_start, best_matches = 0, -1
            for start in range(binder_len - cache_len + 1):
                matches = int(np.sum(
                    binder_aatype[start:start + cache_len] == cache_seq
                ))
                if matches > best_matches:
                    best_matches = matches
                    best_start = start
            aligned[:] = binder_feat[
                best_start:best_start + cache_len
            ].astype(np.float32)
        else:
            best_start, best_matches = 0, -1
            for start in range(cache_len - binder_len + 1):
                matches = int(np.sum(
                    cache_seq[start:start + binder_len] == binder_aatype
                ))
                if matches > best_matches:
                    best_matches = matches
                    best_start = start
            aligned[
                best_start:best_start + binder_len
            ] = binder_feat.astype(np.float32)

        return aligned

    def get_binder_features(self, S, mask, device, meta=None):
        B = S.shape[0]
        results = []
        for b in range(B):
            valid = mask[b].bool()
            cache_seq = S[b, valid].cpu().numpy()
            n_valid = int(valid.sum().item())

            feat_np = None

            if meta is not None and b < len(meta):
                sid = meta[b]
                entry = self._load_one(sid)
                if entry is not None:
                    raw_feat, raw_aatype = entry
                    if raw_feat.shape[0] == n_valid:
                        feat_np = raw_feat.astype(np.float32)
                    else:
                        feat_np = self._align_features(
                            raw_feat, raw_aatype, cache_seq,
                        )

            if feat_np is not None:
                results.append(torch.from_numpy(feat_np).to(device))
            else:
                results.append(None)
        return results


# ============================================================================
#  PDB processing (reused from train_af2bind.py)
# ============================================================================

def fix_nonstandard_residues(struc):
    import mdtraj as md
    for res in struc.top.residues:
        if res.name in NONSTANDARD_TO_STANDARD:
            res.name = NONSTANDARD_TO_STANDARD[res.name]
    return struc


def load_and_process_pdb(pdb_path, chain_id="A", num_atoms=4):
    """Load PDB, extract chain, compute (X, S, mask, resnames, resis, plddt)."""
    import mdtraj as md
    from pockmon.datasets import abbrev, lookup

    try:
        struc = md.load(pdb_path)
    except Exception as e:
        return None
    struc = fix_nonstandard_residues(struc)

    chain_idx = None
    for c in struc.top.chains:
        if c.chain_id == chain_id or (chain_id == "" and c.index == 0):
            chain_idx = c.index
            break
    if chain_idx is None:
        chain_idx = 0

    atom_indices = struc.top.select(f"chainid {chain_idx}")
    if len(atom_indices) == 0:
        return None
    chain_struc = struc.atom_slice(atom_indices)

    prot_iis = chain_struc.top.select(
        "protein and (name N or name CA or name C or name O)"
    )
    if len(prot_iis) == 0:
        return None
    prot_bb = chain_struc.atom_slice(prot_iis)

    bb_names = {"N", "CA", "C", "O"}
    complete_res_idx = []
    for res in prot_bb.top.residues:
        atoms_in_res = {a.name for a in res.atoms}
        if bb_names.issubset(atoms_in_res) and len(atoms_in_res) == 4:
            complete_res_idx.append(res.index)

    if len(complete_res_idx) == 0:
        return None

    if len(complete_res_idx) < prot_bb.top.n_residues:
        keep_atoms = []
        for res in prot_bb.top.residues:
            if res.index in set(complete_res_idx):
                keep_atoms.extend([a.index for a in res.atoms])
        prot_bb = prot_bb.atom_slice(keep_atoms)

    L = prot_bb.top.n_residues
    xyz = prot_bb.xyz.reshape(L, 4, 3)

    seq = [r.name for r in prot_bb.top.residues]
    try:
        S = np.array([lookup[abbrev[a]] for a in seq], dtype=np.int32)
    except KeyError:
        return None

    resnames = [abbrev.get(r.name, "X") for r in prot_bb.top.residues]
    resis = [r.resSeq for r in prot_bb.top.residues]

    # pLDDT: AlphaFold stores it in B-factor column of CA atoms
    plddt = np.zeros(L, dtype=np.float32)
    ca_indices = prot_bb.top.select("name CA")
    if len(ca_indices) == L:
        plddt = prot_bb.xyz[0][ca_indices]  # dummy, need actual b-factors
    # mdtraj doesn't store B-factors; parse from raw PDB lines
    plddt = _parse_ca_bfactors(pdb_path, chain_id, resis)

    X = xyz.astype(np.float32)
    mask = np.ones(L, dtype=np.float32)
    return {
        "X": X, "S": S, "mask": mask, "L": L,
        "resnames": resnames, "resis": resis, "plddt": plddt,
    }


def _parse_ca_bfactors(pdb_path, chain_id, target_resis):
    """Parse B-factor (pLDDT) for CA atoms from PDB file."""
    bfactors = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            ch = line[21]
            if chain_id and ch != chain_id:
                continue
            try:
                resi = int(line[22:26].strip())
                bfac = float(line[60:66].strip())
            except (ValueError, IndexError):
                continue
            bfactors[resi] = bfac

    result = np.zeros(len(target_resis), dtype=np.float32)
    for i, resi in enumerate(target_resis):
        result[i] = bfactors.get(resi, 0.0)
    return result


# ============================================================================
#  Proteome discovery
# ============================================================================

def discover_proteome_pdbs(pdb_dir):
    """Find all AF-{UniProt}-F1-model_v*.pdb files, return (uniprot_id, fname, pdb_path)."""
    entries = []
    for fname in sorted(os.listdir(pdb_dir)):
        m = AF_PDB_PATTERN.match(fname)
        if m:
            uniprot_id = m.group(1)
            frag = int(m.group(2))
            version = m.group(3)
            entries.append((uniprot_id, frag, version, fname,
                            os.path.join(pdb_dir, fname)))
    return entries


def output_csv_path(output_dir, uniprot_id, fname):
    """Create bucketed output path: preds/{last2chars}/{fname}.csv"""
    suffix = uniprot_id[-2:] if len(uniprot_id) >= 2 else "XX"
    subdir = os.path.join(output_dir, suffix)
    os.makedirs(subdir, exist_ok=True)
    csv_name = fname.replace(".pdb", ".csv")
    return os.path.join(subdir, csv_name)


# ============================================================================
#  Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict per-residue P(bind) across human proteome with PockMon."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--pdb-dir", default=DEFAULT_PDB_DIR)
    parser.add_argument("--binder-pair-dir", default=DEFAULT_BINDER_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-proteins", type=int, default=None)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--f1-only", action="store_true", default=True,
                        help="Only process F1 fragments (default True). "
                             "Use --no-f1-only to include all fragments.")
    parser.add_argument("--no-f1-only", dest="f1_only", action="store_false")
    return parser.parse_args()


def main():
    args = parse_args()
    assert 0 <= args.shard_id < args.num_shards

    # --- Load config ---
    with open(args.config, "r") as f:
        config = json.load(f)

    # Override binder_pair_dir to point to proteome features
    # (but don't actually preload — we'll replace the provider below)
    config["model"]["binder_pair_dir"] = args.binder_pair_dir

    device = torch.device(args.device)
    num_atoms = get_num_atoms(config)

    # --- Discover PDBs ---
    all_entries = discover_proteome_pdbs(args.pdb_dir)
    if args.f1_only:
        all_entries = [(uid, frag, ver, fn, path)
                       for uid, frag, ver, fn, path in all_entries if frag == 1]

    if args.max_proteins:
        all_entries = all_entries[:args.max_proteins]

    total_before_shard = len(all_entries)
    if args.num_shards > 1:
        all_entries = [e for i, e in enumerate(all_entries)
                       if i % args.num_shards == args.shard_id]

    # Filter already-done
    to_process = []
    for uid, frag, ver, fname, pdb_path in all_entries:
        csv_path = output_csv_path(args.output_dir, uid, fname)
        if args.skip_existing and os.path.isfile(csv_path):
            continue
        binder_path = os.path.join(args.binder_pair_dir, f"{uid}.npz")
        if not os.path.isfile(binder_path):
            continue
        to_process.append((uid, frag, ver, fname, pdb_path, csv_path))

    print("=" * 70)
    print("  PockMon Proteome-wide Binding Site Prediction")
    print("=" * 70)
    print(f"  Config         : {args.config}")
    print(f"  Checkpoint     : {args.checkpoint}")
    print(f"  PDB dir        : {args.pdb_dir}")
    print(f"  Binder pair dir: {args.binder_pair_dir}")
    print(f"  Output dir     : {args.output_dir}")
    print(f"  Device         : {device}")
    print(f"  Total PDBs     : {total_before_shard}")
    if args.num_shards > 1:
        print(f"  Shard          : {args.shard_id} / {args.num_shards}  "
              f"(this shard: {len(all_entries)} proteins)")
    print(f"  To predict     : {len(to_process)}  "
          f"(skipped {len(all_entries) - len(to_process)} existing/no-binder)")
    print()

    if not to_process:
        print("  Nothing to process.")
        return

    # --- Build model ---
    # Use an empty temp directory so AF2BinderPairProvider loads 0 entries
    # (avoids loading all 23k npz into RAM). We replace it with a lazy
    # provider after loading the checkpoint.
    saved_binder_dir = config["model"]["binder_pair_dir"]
    _placeholder = tempfile.mkdtemp(prefix="lazy_binder_")
    config["model"]["binder_pair_dir"] = _placeholder

    model = make_model(config)
    model.to(device)

    # Materialize lazy parameters if needed
    if any(isinstance(p, torch.nn.parameter.UninitializedParameter)
           for p in model.parameters()):
        x_dummy = torch.zeros(1, 10, num_atoms, 3, device=device)
        s_dummy = torch.zeros(1, 10, dtype=torch.long, device=device)
        m_dummy = torch.ones(1, 10, device=device)
        model.eval()
        with torch.no_grad():
            model(x_dummy, s_dummy, m_dummy, train=False, res_level=True)

    # Load checkpoint
    load_checkpoint(model, None, args.checkpoint)
    model.to(device)
    model.eval()

    # Replace binder_provider with lazy version
    lazy_provider = LazyBinderPairProvider(saved_binder_dir)
    model.binder_provider = lazy_provider

    import shutil
    shutil.rmtree(_placeholder, ignore_errors=True)

    print(f"Model loaded. Using LazyBinderPairProvider from {saved_binder_dir}")
    print()

    # --- Inference loop ---
    successes, failures, skipped = 0, 0, 0
    t_start = time.time()

    for idx, (uid, frag, ver, fname, pdb_path, csv_path) in enumerate(to_process):
        progress = f"[{idx + 1}/{len(to_process)}]"
        t0 = time.time()

        try:
            result = load_and_process_pdb(pdb_path, chain_id="A", num_atoms=num_atoms)
            if result is None:
                print(f"  {progress} {uid}: SKIP (PDB parse failed)")
                skipped += 1
                continue

            X = result["X"]
            S = result["S"]
            mask = result["mask"]
            L = result["L"]
            resnames = result["resnames"]
            resis = result["resis"]
            plddt = result["plddt"]

            X_t = torch.from_numpy(X).unsqueeze(0).to(device)
            S_t = torch.from_numpy(S).unsqueeze(0).to(device)
            mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)

            with torch.no_grad():
                probs = model(X_t, S_t, mask_t, train=False, res_level=True,
                              meta=[uid])
            p_bind = probs[0, :L].cpu().numpy()

            # Write CSV (sorted by p(bind) descending, matching af2bind/preds format)
            indices = np.argsort(p_bind)[::-1]
            with open(csv_path, "w") as f:
                f.write("rank,chain,resi,resn,p(bind),arr_i,plddt\n")
                for rank, i in enumerate(indices):
                    f.write(f"{rank},A,{resis[i]},{resnames[i]},"
                            f"{p_bind[i]:.7f},{i},{plddt[i]:.1f}\n")

            elapsed = time.time() - t0
            print(f"  {progress} {uid}: L={L}, max_p={p_bind.max():.3f}, "
                  f"{elapsed:.1f}s")
            successes += 1

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  {progress} {uid}: FAILED {elapsed:.1f}s — {e}")
            failures += 1

        finally:
            lazy_provider.evict(uid)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if (idx + 1) % 200 == 0:
            elapsed_total = time.time() - t_start
            rate = (idx + 1) / elapsed_total * 3600
            remaining = (len(to_process) - idx - 1) / rate * 60
            print(f"  --- Progress: {idx + 1}/{len(to_process)}, "
                  f"{rate:.0f}/h, ~{remaining:.0f}min remaining ---")

    total_time = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  DONE: {successes} succeeded, {failures} failed, "
          f"{skipped} skipped, {total_time / 60:.1f} min total")
    print(f"  Output: {args.output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
