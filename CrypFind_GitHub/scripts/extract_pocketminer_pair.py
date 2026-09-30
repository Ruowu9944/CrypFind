"""
Extract AF2 combined pair representations for PocketMiner proteins.

From a single binder-protocol forward pass, extract BOTH:
  1. Target-Binder fingerprint  [L, 5120]  (identical to binder pair)
  2. Target-Target global pair  [L, L, 128] (analogous to fixbb pair,
     but computed in the presence of a binder context)

Output per protein:  {output_dir}/{protein_name}.npz
  - features  : float16, [L, 5120]    (binder fingerprint)
  - pair_repr : float16, [L, L, 128]  (global residue-residue pair)
  - aatype    : int32,   [L]

Usage:
    python extract_pocketminer_combined_pair.py
    python extract_pocketminer_combined_pair.py --skip-existing
    python extract_pocketminer_combined_pair.py --shard-id 0 --num-shards 4
"""

import argparse
import gc
import os
import sys
import tempfile
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

PDB_DIR = os.path.join(PROJECT_ROOT, "datasets", "pocketminer", "training-data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "datasets", "pocketminer", "combined_pair")
PARAMS_DIR = os.path.join(PROJECT_ROOT, "datasets", "af2bind", "params")

NONSTANDARD_TO_STANDARD = {
    "CYM": "CYS", "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
    "HSE": "HIS", "HSD": "HIS", "HSP": "HIS", "MSE": "MET",
    "MLY": "LYS", "PCA": "GLU", "CME": "CYS", "CSO": "CYS",
    "SEP": "SER", "TPO": "THR", "PTR": "TYR", "HYP": "PRO",
    "SAC": "SER", "DAL": "ALA", "AIB": "ALA",
}


def get_train_protein_names():
    """Return the 38 PocketMiner training protein names."""
    data_dir = os.path.join(PROJECT_ROOT, "datasets", "pocketminer")
    npy_path = os.path.join(
        data_dir, "task2",
        "X-train-gp-to-nearest-resi-procedure-min-rank-7-window-40-stride-1.npy",
    )
    X = np.load(npy_path, allow_pickle=True)
    names = set()
    for row in X:
        pdb_name = row[1].split("/")[-1].replace(".pdb", "")
        names.add(pdb_name)
    return sorted(names)


def get_val_test_protein_names():
    """Return the 61 PocketMiner val+test protein names from apo ID files."""
    pm_dir = os.path.join(PROJECT_ROOT, "datasets", "pocketminer", "pm-dataset")
    names = set()
    for fname in ("val_apo_ids_with_chainids.npy", "test_apo_ids_with_chainids.npy"):
        fpath = os.path.join(pm_dir, fname)
        if os.path.isfile(fpath):
            ids = np.load(fpath, allow_pickle=True)
            for sid in ids:
                names.add(str(sid))
    return sorted(names)


def get_chain_from_pdb(pdb_path):
    """Get the first chain ID from ATOM records."""
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM"):
                chain = line[21]
                return chain if chain.strip() else "A"
    return "A"


def preprocess_pdb(pdb_path):
    """Clean PDB for ColabDesign: strip MODEL/ENDMDL, fix missing chain ID,
    replace non-standard residues.  PocketMiner PDBs come from MD simulations
    (MDTraj / GROMACS trjconv) and contain MODEL/ENDMDL records that
    ColabDesign cannot parse (it sees zero atoms -> 'Empty file.')."""
    lines = []
    in_first_model = False
    past_first_model = False
    needs_rewrite = False

    with open(pdb_path) as f:
        for line in f:
            if line.startswith("MODEL"):
                if not past_first_model:
                    in_first_model = True
                    needs_rewrite = True
                    continue
                else:
                    break
            if line.startswith("ENDMDL"):
                past_first_model = True
                in_first_model = False
                needs_rewrite = True
                continue

            if line.startswith(("ATOM", "HETATM")):
                if line[21] == " ":
                    line = line[:21] + "A" + line[22:]
                    needs_rewrite = True
                resname = line[17:20].strip()
                if resname in NONSTANDARD_TO_STANDARD:
                    std = NONSTANDARD_TO_STANDARD[resname]
                    line = line[:17] + f"{std:>3s}" + line[20:]
                    needs_rewrite = True

            lines.append(line)

    if not needs_rewrite:
        return pdb_path, None

    tmp = tempfile.NamedTemporaryFile(
        suffix=".pdb", prefix="pm_combined_", delete=False, mode="w"
    )
    for line in lines:
        tmp.write(line)
    tmp.close()
    return tmp.name, tmp.name


def extract_combined_features(af_model, pdb_path, chain, output_path):
    """Run AF2 binder protocol and save both binder fingerprint [L, 5120]
    and target-target global pair [L, L, 128]."""
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

    # --- Binder fingerprint [L, 5120] (same as binder-only extraction) ---
    pair_A = pair_full[:-20, -20:]
    pair_B = pair_full[-20:, :-20].swapaxes(0, 1)
    pair_A = pair_A.reshape(pair_A.shape[0], -1)
    pair_B = pair_B.reshape(pair_B.shape[0], -1)
    features = np.concatenate([pair_A, pair_B], axis=-1)

    # --- Target-Target global pair [L, L, 128] ---
    pair_repr = pair_full[:target_len, :target_len, :]

    del pair_full, pair_A, pair_B

    np.savez_compressed(
        output_path,
        features=features.astype(np.float16),
        pair_repr=pair_repr.astype(np.float16),
        aatype=wt_aatype,
    )
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    return target_len, features.shape[-1], size_mb


def main():
    parser = argparse.ArgumentParser(
        description="Extract AF2 combined pair features for PocketMiner."
    )
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--pdb-dir", default=PDB_DIR)
    parser.add_argument("--params-dir", default=PARAMS_DIR)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-proteins", type=int, default=None)
    parser.add_argument("--shard-id", type=int, default=0,
                        help="This shard's index in [0, num_shards).")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of parallel shards.")
    parser.add_argument("--include-val-test", action="store_true",
                        help="Also extract for val/test apo proteins.")
    args = parser.parse_args()

    assert 0 <= args.shard_id < args.num_shards, (
        f"--shard-id must be in [0, --num-shards); got "
        f"shard_id={args.shard_id}, num_shards={args.num_shards}"
    )

    os.makedirs(args.output_dir, exist_ok=True)

    protein_names = get_train_protein_names()
    print(f"PocketMiner training proteins: {len(protein_names)}")

    if args.include_val_test:
        vt_names = get_val_test_protein_names()
        print(f"PocketMiner val/test proteins: {len(vt_names)}")
        protein_names = sorted(set(protein_names) | set(vt_names))
        print(f"Total unique proteins: {len(protein_names)}")

    # Determine PDB source for each protein
    apo_dir = os.path.join(
        PROJECT_ROOT, "datasets", "pocketminer", "pm-dataset", "apo-structures"
    )

    # Build case-insensitive lookup for apo-structures/ (files are {stem}_clean_h.pdb)
    apo_lookup = {}
    if os.path.isdir(apo_dir):
        for fn in os.listdir(apo_dir):
            if fn.endswith("_clean_h.pdb"):
                stem = fn[: -len("_clean_h.pdb")]
                apo_lookup[stem.lower()] = os.path.join(apo_dir, fn)

    to_process = []
    for name in protein_names:
        # Training proteins are in training-data/{name}.pdb
        pdb_path = os.path.join(args.pdb_dir, f"{name}.pdb")
        if not os.path.isfile(pdb_path):
            # Val/test proteins are in apo-structures/{STEM}_clean_h.pdb
            pdb_path = apo_lookup.get(name.lower())
        out_path = os.path.join(args.output_dir, f"{name}.npz")
        if args.skip_existing and os.path.isfile(out_path):
            print(f"  SKIP (exists): {name}")
            continue
        if pdb_path is None or not os.path.isfile(pdb_path):
            print(f"  WARNING: PDB not found: {name}")
            continue
        chain = get_chain_from_pdb(pdb_path)
        to_process.append((name, chain, pdb_path, out_path))

    if args.max_proteins:
        to_process = to_process[:args.max_proteins]

    total_before_shard = len(to_process)
    if args.num_shards > 1:
        to_process = [item for i, item in enumerate(to_process)
                      if i % args.num_shards == args.shard_id]

    print("=" * 70)
    print("  AF2 Combined Pair Extraction for PocketMiner")
    print("=" * 70)
    if args.num_shards > 1:
        print(f"  Shard          : {args.shard_id} / {args.num_shards} "
              f"(this shard: {len(to_process)} / {total_before_shard})")
    print(f"  To extract     : {len(to_process)}")
    print(f"  Output dir     : {args.output_dir}")
    print(f"  PDB dir        : {args.pdb_dir}")
    print(f"  Params dir     : {args.params_dir}")

    if not to_process:
        print("  Nothing to process.")
        return

    est_min = len(to_process) * 30 / 60
    print(f"  Estimated time : ~{est_min:.0f} minutes (~30s per protein)")
    print()

    to_process = [
        (name, chain, os.path.abspath(pdb), os.path.abspath(out))
        for name, chain, pdb, out in to_process
    ]
    params_abs = os.path.abspath(args.params_dir)
    os.chdir(os.path.dirname(params_abs))

    import jax
    print(f"JAX {jax.__version__}  |  Devices: {jax.devices()}")

    from colabdesign import mk_afdesign_model, clear_mem

    print("Creating AF2 model (protocol=binder, debug=True) ...")
    clear_mem()
    af_model = mk_afdesign_model(
        protocol="binder", debug=True, data_dir=params_abs,
    )
    print("Model ready.\n")

    successes, failures = 0, 0
    t_start = time.time()

    for idx, (name, chain, pdb_path, out_path) in enumerate(to_process):
        progress = f"[{idx+1}/{len(to_process)}]"
        t0 = time.time()

        actual_pdb, tmp_file = preprocess_pdb(pdb_path)
        try:
            L, F, size_mb = extract_combined_features(
                af_model, actual_pdb, chain, out_path,
            )
            elapsed = time.time() - t0
            print(f"  {progress} {name}: L={L}, F={F}, "
                  f"{size_mb:.1f}MB, {elapsed:.1f}s")
            successes += 1
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  {progress} {name}: FAILED {elapsed:.1f}s — {e}")
            failures += 1
        finally:
            if tmp_file and os.path.exists(tmp_file):
                os.unlink(tmp_file)
            if hasattr(af_model, 'aux') and af_model.aux is not None:
                af_model.aux.get("debug", {}).pop("outputs", None)
            gc.collect()

    total_time = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  DONE: {successes} succeeded, {failures} failed, "
          f"{total_time/60:.1f} min total")
    print(f"  Output: {args.output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
