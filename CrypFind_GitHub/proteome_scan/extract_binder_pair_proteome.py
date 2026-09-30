"""
Extract AF2 binder pair representations for the human proteome.

Scans a directory of AlphaFold-predicted PDB files (AF-{UniProt}-F1-model_v4.pdb)
and extracts per-residue binder pair features [L, 5120] using the ColabDesign
AF2 binder protocol — identical to af2bind/extract_binder_pair_all.py.

Output per protein:  {output_dir}/{uniprot_id}.npz
  - features : float16, [L, 5120]  (pair_A ++ pair_B)
  - aatype   : int32,   [L]

Usage:
    # Extract all (single GPU)
    python extract_binder_pair_proteome.py --skip-existing

    # 16-way parallel sharding
    python extract_binder_pair_proteome.py --skip-existing --num-shards 16 --shard-id 0

    # Quick smoke test with 5 proteins
    python extract_binder_pair_proteome.py --max-proteins 5
"""

import argparse
import gc
import glob
import os
import re
import tempfile
import time

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PDB_DIR = os.path.join(BASE_DIR, "proteome_pdbs")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "proteome_binder_pair")
DEFAULT_PARAMS_DIR = os.path.join(os.path.dirname(BASE_DIR), "datasets", "af2bind", "params")

NONSTANDARD_TO_STANDARD = {
    "CYM": "CYS", "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
    "HSE": "HIS", "HSD": "HIS", "HSP": "HIS", "MSE": "MET",
    "MLY": "LYS", "PCA": "GLU", "CME": "CYS", "CSO": "CYS",
    "SEP": "SER", "TPO": "THR", "PTR": "TYR", "HYP": "PRO",
    "SAC": "SER", "DAL": "ALA", "AIB": "ALA",
}

AF_PDB_PATTERN = re.compile(r"^AF-(.+)-F1-model_v\d+\.pdb$")


def discover_proteome_pdbs(pdb_dir):
    """Find all AF-{UniProt}-F1-model_v*.pdb files and return (uniprot_id, pdb_path) list."""
    entries = []
    for fname in sorted(os.listdir(pdb_dir)):
        m = AF_PDB_PATTERN.match(fname)
        if m:
            uniprot_id = m.group(1)
            entries.append((uniprot_id, os.path.join(pdb_dir, fname)))
    return entries


def detect_chain_id(pdb_path):
    chains = set()
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM"):
                chains.add(line[21])
    if len(chains) == 1:
        chain = chains.pop()
        return chain if chain.strip() else "A"
    if len(chains) == 0:
        return "A"
    return "A"


def preprocess_pdb(pdb_path):
    needs_fix = False
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM"):
                resname = line[17:20].strip()
                if resname in NONSTANDARD_TO_STANDARD:
                    needs_fix = True
                    break

    if not needs_fix:
        return pdb_path, None

    tmp = tempfile.NamedTemporaryFile(
        suffix=".pdb", prefix="proteome_ext_", delete=False, mode="w"
    )
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM"):
                resname = line[17:20].strip()
                if resname in NONSTANDARD_TO_STANDARD:
                    std = NONSTANDARD_TO_STANDARD[resname]
                    line = line[:17] + f"{std:>3s}" + line[20:]
            tmp.write(line)
    tmp.close()
    return tmp.name, tmp.name


def extract_binder_features(af_model, pdb_path, chain, output_path):
    """Run AF2 binder protocol and save [L, 5120] features."""
    af_model.prep_inputs(
        pdb_filename=pdb_path,
        chain=chain,
        binder_len=20,
        rm_target_sc=True,
        rm_target_seq=False,
    )
    target_len = af_model._target_len
    wt_aatype = np.asarray(
        af_model._pdb["batch"]["aatype"][:target_len], dtype=np.int32,
    )
    assert wt_aatype.shape[0] == target_len

    r_idx = af_model._inputs["residue_index"][-20] + (1 + np.arange(20)) * 50
    af_model._inputs["residue_index"][-20:] = r_idx.flatten()
    af_model.set_seq("ACDEFGHIKLMNPQRSTVWY")
    af_model.predict(verbose=False)

    pair_full = af_model.aux["debug"]["outputs"]["representations"]["pair"]
    pair_A = pair_full[:-20, -20:]
    pair_B = pair_full[-20:, :-20].swapaxes(0, 1)

    pair_A = pair_A.reshape(pair_A.shape[0], -1)
    pair_B = pair_B.reshape(pair_B.shape[0], -1)
    features = np.concatenate([pair_A, pair_B], axis=-1)

    np.savez_compressed(
        output_path,
        features=features.astype(np.float16),
        aatype=wt_aatype,
    )
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    return target_len, features.shape[-1], size_mb


def main():
    parser = argparse.ArgumentParser(
        description="Extract AF2 binder pair features for the human proteome."
    )
    parser.add_argument("--pdb-dir", default=DEFAULT_PDB_DIR,
                        help="Directory containing AF-*-F1-model_v*.pdb files")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--params-dir", default=DEFAULT_PARAMS_DIR,
                        help="AlphaFold2 model parameters directory")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-proteins", type=int, default=None,
                        help="Limit to first N proteins (for testing)")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    assert 0 <= args.shard_id < args.num_shards, (
        f"--shard-id must be in [0, --num-shards); got "
        f"shard_id={args.shard_id}, num_shards={args.num_shards}"
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Discover all proteome PDBs ---
    all_entries = discover_proteome_pdbs(args.pdb_dir)
    if not all_entries:
        print(f"ERROR: No AF-*-F1-model_v*.pdb files found in {args.pdb_dir}")
        return

    if args.max_proteins:
        all_entries = all_entries[:args.max_proteins]

    total_before_shard = len(all_entries)
    if args.num_shards > 1:
        all_entries = [e for i, e in enumerate(all_entries)
                       if i % args.num_shards == args.shard_id]

    # --- Filter out already-extracted ---
    to_process = []
    for uniprot_id, pdb_path in all_entries:
        out_path = os.path.join(args.output_dir, f"{uniprot_id}.npz")
        if args.skip_existing and os.path.isfile(out_path):
            continue
        to_process.append((uniprot_id, pdb_path, out_path))

    print("=" * 70)
    print("  AF2 Binder Pair Extraction — Human Proteome")
    print("=" * 70)
    print(f"  PDB dir        : {args.pdb_dir}")
    print(f"  Output dir     : {args.output_dir}")
    print(f"  Params dir     : {args.params_dir}")
    print(f"  Total PDBs     : {total_before_shard}")
    if args.num_shards > 1:
        print(f"  Shard          : {args.shard_id} / {args.num_shards}  "
              f"(this shard: {len(all_entries)} proteins)")
    print(f"  To extract     : {len(to_process)}  "
          f"(skipped {len(all_entries) - len(to_process)} existing)")
    est_hours = len(to_process) * 40 / 3600
    print(f"  Est. time      : ~{est_hours:.1f} hours (~40s per protein)")
    print()

    if not to_process:
        print("  Nothing to process.")
        return

    # --- Initialize AF2 model ---
    import jax
    print(f"JAX {jax.__version__}  |  Devices: {jax.devices()}")

    from colabdesign import mk_afdesign_model, clear_mem

    print("Creating AF2 model (protocol=binder, debug=True) ...")
    clear_mem()
    af_model = mk_afdesign_model(
        protocol="binder", debug=True, data_dir=args.params_dir,
    )
    print("Model ready.\n")

    successes, failures = 0, 0
    t_start = time.time()

    for idx, (uniprot_id, pdb_path, out_path) in enumerate(to_process):
        progress = f"[{idx + 1}/{len(to_process)}]"
        t0 = time.time()

        chain = detect_chain_id(pdb_path)
        actual_pdb, tmp_file = preprocess_pdb(pdb_path)
        try:
            L, F, size_mb = extract_binder_features(
                af_model, actual_pdb, chain, out_path
            )
            elapsed = time.time() - t0
            print(f"  {progress} {uniprot_id}: L={L}, F={F}, "
                  f"{size_mb:.1f}MB, {elapsed:.1f}s")
            successes += 1
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  {progress} {uniprot_id}: FAILED {elapsed:.1f}s — {e}")
            failures += 1
        finally:
            if tmp_file and os.path.exists(tmp_file):
                os.unlink(tmp_file)
            gc.collect()
            try:
                jax.clear_caches()
            except AttributeError:
                pass

        if (idx + 1) % 100 == 0:
            elapsed_total = time.time() - t_start
            rate = (idx + 1) / elapsed_total * 3600
            remaining = (len(to_process) - idx - 1) / rate * 60
            print(f"  --- Progress: {idx + 1}/{len(to_process)}, "
                  f"{rate:.0f}/h, ~{remaining:.0f}min remaining ---")

    total_time = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  DONE: {successes} succeeded, {failures} failed, "
          f"{total_time / 60:.1f} min total")
    print(f"  Output: {args.output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
