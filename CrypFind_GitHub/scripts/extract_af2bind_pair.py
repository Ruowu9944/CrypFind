"""
Extract AF2 binder pair representations for ALL AF2BIND proteins (~1897).

These per-residue features [L, 5120] are needed to train PockMon/GVP models
on the AF2BIND binding-site prediction task.

Output per protein:  binder_pair_all/{prot_id}.npz
  - features : float16, [L, 5120]  (pair_A + pair_B concatenated)
  - aatype   : int32,   [L]

Usage:
    # Extract all proteins (train+val+test)
    python extract_binder_pair_all.py

    # Resume interrupted run
    python extract_binder_pair_all.py --skip-existing

    # Only extract for a specific split (0=train, 1=val, 2=test)
    python extract_binder_pair_all.py --split 0

    # Quick test with N proteins
    python extract_binder_pair_all.py --max-proteins 5
"""

import os
import sys
import re
import gc
import time
import tempfile
import argparse
import pickle
import numpy as np

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets", "af2bind")
PARAMS_DIR = os.path.join(BASE_DIR, "params")
OUTPUT_DIR = os.path.join(BASE_DIR, "binder_pair_all")
PDB_DIR = os.path.join(BASE_DIR, "all_pdbs")
ASSIGNMENTS_PATH = os.path.join(BASE_DIR, "training", "assignments_2k.pkl")
LABELS_PATH = os.path.join(BASE_DIR, "training", "new_labels_2k.pkl")

NONSTANDARD_TO_STANDARD = {
    "CYM": "CYS", "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
    "HSE": "HIS", "HSD": "HIS", "HSP": "HIS", "MSE": "MET",
    "MLY": "LYS", "PCA": "GLU", "CME": "CYS", "CSO": "CYS",
    "SEP": "SER", "TPO": "THR", "PTR": "TYR", "HYP": "PRO",
    "SAC": "SER", "DAL": "ALA", "AIB": "ALA",
}


def download_pdb(pdb_code, out_dir):
    out_path = os.path.join(out_dir, f"{pdb_code}.pdb")
    if os.path.isfile(out_path) and os.path.getsize(out_path) > 100:
        return out_path
    url = f"https://files.rcsb.org/view/{pdb_code}.pdb"
    ret = os.system(f'wget -q -O "{out_path}" "{url}"')
    if ret != 0 or not os.path.isfile(out_path) or os.path.getsize(out_path) < 100:
        if os.path.exists(out_path):
            os.remove(out_path)
        return None
    return out_path


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
        suffix=".pdb", prefix="binder_ext_", delete=False, mode="w"
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
    # NOTE: Under protocol="binder" ColabDesign does NOT expose `_wt_aatype`
    # (that attribute exists only under fixbb).  The wildtype target sequence
    # is, however, parsed from the PDB and stored verbatim at
    # `_pdb["batch"]["aatype"]` (length = target_len + binder_len; the first
    # `target_len` entries are the wildtype target, the trailing `binder_len`
    # entries are bait placeholders).  Reading from there guarantees:
    #   1. zero external PDB-parser dependency (no mdtraj / biopython needed),
    #   2. identical chain selection / non-standard residue handling as the
    #      AF2 forward pass actually consumes (no risk of drift),
    #   3. byte-for-byte equality with fixbb's `_wt_aatype` on the same PDB,
    #   4. encoding identical to AF2's `restype_order` (A,R,N,D,C,Q,E,G,H,I,
    #      L,K,M,F,P,S,T,W,Y,V) which is the same scheme used by the trainer
    #      side `crypfind/datasets.py:lookup` table, so the
    #      saved aatype can be used directly by `AF2BinderPairProvider` to
    #      hash-match incoming batches.
    wt_aatype = np.asarray(
        af_model._pdb["batch"]["aatype"][:target_len], dtype=np.int32,
    )
    assert wt_aatype.shape[0] == target_len, (
        f"target aatype length {wt_aatype.shape[0]} != _target_len "
        f"{target_len}; ColabDesign chain selection / non-standard residue "
        f"handling may have diverged for {pdb_path} chain={chain}."
    )

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
        description="Extract AF2 binder pair features for AF2BIND dataset."
    )
    parser.add_argument("--seed", type=str, default="all",
                        help="Fold index (0-9) or 'all' to union across all folds")
    parser.add_argument("--split", type=int, default=None,
                        help="Only process split: 0=train, 1=val, 2=test")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-proteins", type=int, default=None)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--pdb-dir", default=PDB_DIR)
    # Sharding: split work across N parallel jobs (e.g. 2 GPUs).
    # Each job processes prot_ids[i] for which i % num_shards == shard_id.
    # Combined with --skip-existing this is robust to crashes / restarts.
    parser.add_argument("--shard-id", type=int, default=0,
                        help="This shard's index in [0, num_shards).")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of parallel shards (e.g. 2 for 2 GPUs).")
    args = parser.parse_args()
    assert 0 <= args.shard_id < args.num_shards, (
        f"--shard-id must be in [0, --num-shards); got "
        f"shard_id={args.shard_id}, num_shards={args.num_shards}"
    )

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.pdb_dir, exist_ok=True)

    with open(ASSIGNMENTS_PATH, "rb") as f:
        assignments = pickle.load(f)

    with open(LABELS_PATH, "rb") as f:
        labels = pickle.load(f)

    if args.seed == "all":
        all_ids = set()
        for fold_dict in assignments:
            all_ids.update(k for k in fold_dict if k in labels)
        prot_ids = sorted(all_ids)
        split_name = "all (union of all folds)"
    else:
        seed_idx = int(args.seed)
        assigned = assignments[seed_idx]
        if args.split is not None:
            prot_ids = sorted(k for k, v in assigned.items()
                              if v == args.split and k in labels)
            split_name = {0: "train", 1: "val", 2: "test"}[args.split]
        else:
            prot_ids = sorted(k for k in assigned if k in labels)
            split_name = f"all (seed={seed_idx})"

    if args.max_proteins:
        prot_ids = prot_ids[:args.max_proteins]

    total_before_shard = len(prot_ids)
    if args.num_shards > 1:
        prot_ids = [pid for i, pid in enumerate(prot_ids)
                    if i % args.num_shards == args.shard_id]

    print("=" * 70)
    print("  AF2 Binder Pair Extraction for AF2BIND Dataset")
    print("=" * 70)
    print(f"  Split          : {split_name}")
    if args.num_shards > 1:
        print(f"  Shard          : {args.shard_id} / {args.num_shards}  "
              f"(this shard owns {len(prot_ids)} of {total_before_shard} "
              f"proteins, by interleaved index)")
    print(f"  Total proteins : {len(prot_ids)}")
    print(f"  Output dir     : {args.output_dir}")
    print(f"  PDB dir        : {args.pdb_dir}")

    # --- Phase 1: Download PDBs ---
    print(f"\n--- Phase 1: Downloading PDBs ---")
    download_ok, download_fail = 0, 0
    for pid in prot_ids:
        pdb_code = pid.split("_")[0]
        pdb_path = os.path.join(args.pdb_dir, f"{pdb_code}.pdb")
        if os.path.isfile(pdb_path) and os.path.getsize(pdb_path) > 100:
            download_ok += 1
            continue
        result = download_pdb(pdb_code, args.pdb_dir)
        if result:
            download_ok += 1
        else:
            download_fail += 1
            print(f"  WARN: Failed to download {pdb_code}")
    print(f"  Downloaded: {download_ok} OK, {download_fail} failed")

    # --- Phase 2: Filter processable proteins ---
    to_process = []
    for pid in prot_ids:
        pdb_code = pid.split("_")[0]
        chain = pid.split("_")[1] if "_" in pid else "A"
        out_path = os.path.join(args.output_dir, f"{pid}.npz")
        if args.skip_existing and os.path.isfile(out_path):
            continue
        pdb_path = os.path.join(args.pdb_dir, f"{pdb_code}.pdb")
        if not os.path.isfile(pdb_path):
            continue
        to_process.append((pid, pdb_code, chain, pdb_path, out_path))

    if not to_process:
        print("  Nothing to process (all exist or no PDBs found).")
        return

    print(f"\n--- Phase 2: Extracting binder pair features ---")
    print(f"  To extract: {len(to_process)}")
    est_hours = len(to_process) * 30 / 3600
    print(f"  Estimated time: ~{est_hours:.1f} hours (~30s per protein)")
    print()

    # --- Initialize AF2 model ---
    import jax
    print(f"JAX {jax.__version__}  |  Devices: {jax.devices()}")

    from colabdesign import mk_afdesign_model, clear_mem

    print("Creating AF2 model (protocol=binder, debug=True) ...")
    clear_mem()
    af_model = mk_afdesign_model(
        protocol="binder", debug=True, data_dir=PARAMS_DIR,
    )
    print("Model ready.\n")

    successes, failures = 0, 0
    t_start = time.time()

    for idx, (pid, pdb_code, chain, pdb_path, out_path) in enumerate(to_process):
        progress = f"[{idx+1}/{len(to_process)}]"
        t0 = time.time()

        actual_pdb, tmp_file = preprocess_pdb(pdb_path)
        try:
            L, F, size_mb = extract_binder_features(
                af_model, actual_pdb, chain, out_path
            )
            elapsed = time.time() - t0
            print(f"  {progress} {pid}: L={L}, F={F}, "
                  f"{size_mb:.1f}MB, {elapsed:.1f}s")
            successes += 1
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  {progress} {pid}: FAILED {elapsed:.1f}s — {e}")
            failures += 1
        finally:
            if tmp_file and os.path.exists(tmp_file):
                os.unlink(tmp_file)
            gc.collect()

        if (idx + 1) % 50 == 0:
            elapsed_total = time.time() - t_start
            rate = (idx + 1) / elapsed_total * 3600
            remaining = (len(to_process) - idx - 1) / rate * 60
            print(f"  --- Progress: {idx+1}/{len(to_process)}, "
                  f"{rate:.0f}/h, ~{remaining:.0f}min remaining ---")

    total_time = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  DONE: {successes} succeeded, {failures} failed, "
          f"{total_time/60:.1f} min total")
    print(f"  Output: {args.output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
