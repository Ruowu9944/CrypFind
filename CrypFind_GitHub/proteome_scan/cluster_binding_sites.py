"""
Post-process PockMon per-residue P(bind) predictions into binding sites.

Implements the AF2BIND paper's site-detection pipeline:
  1. pLDDT filter: zero out residues with pLDDT < 50
  2. Threshold: select residues with P(bind) > threshold (default 0.28)
  3. DBSCAN clustering on heavy-atom distance (eps=6.0 Å, min_samples=3)
  4. Discard clusters with < 5 residues
  5. Score clusters via CDF z-score and cluster_rank

Input:  preds/{XX}/{UniProt}-F1-model_v6.csv  (from predict_proteome.py)
        proteome_pdbs/AF-{UniProt}-F1-model_v6.pdb  (for 3D coordinates)

Output: sites/{XX}/{UniProt}-F1-model_v6_sites.csv  (binding sites with scores)
        sites_summary.csv  (all proteins × sites summary)

Usage:
    python cluster_binding_sites.py
    python cluster_binding_sites.py --plddt-threshold 50 --pbind-threshold 0.28
    python cluster_binding_sites.py --num-shards 8 --shard-id 0
"""

import argparse
import csv
import gc
import os
import re
import sys
import time

import numpy as np

SCAN_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PREDS_DIR = os.path.join(SCAN_DIR, "preds")
DEFAULT_PDB_DIR = os.path.join(SCAN_DIR, "proteome_pdbs")
DEFAULT_SITES_DIR = os.path.join(SCAN_DIR, "sites")

AF_PDB_PATTERN = re.compile(r"^AF-(.+)-F(\d+)-model_v(\d+)\.pdb$")


# ============================================================================
#  Parse prediction CSV
# ============================================================================

def load_predictions(csv_path):
    """Load per-residue predictions from PockMon output CSV.

    Returns dict with arrays: resi, resn, p_bind, arr_i, plddt, chain
    """
    resis, resns, p_binds, arr_is, plddts, chains = [], [], [], [], [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            resis.append(int(row["resi"]))
            resns.append(row["resn"])
            p_binds.append(float(row["p(bind)"]))
            arr_is.append(int(row["arr_i"]))
            plddts.append(float(row["plddt"]))
            chains.append(row["chain"])

    # Sort by arr_i to restore sequential residue order
    order = np.argsort(arr_is)
    return {
        "resi": np.array(resis)[order],
        "resn": np.array(resns)[order],
        "p_bind": np.array(p_binds)[order],
        "arr_i": np.array(arr_is)[order],
        "plddt": np.array(plddts)[order],
        "chain": np.array(chains)[order],
    }


# ============================================================================
#  Extract CA coordinates from PDB
# ============================================================================

def parse_ca_coords(pdb_path, chain_id="A"):
    """Parse CA atom coordinates and residue numbers from PDB file.

    Returns: coords [N, 3] in Angstroms, resi_list [N]
    """
    coords = []
    resi_list = []
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
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                resi = int(line[22:26].strip())
            except (ValueError, IndexError):
                continue
            coords.append([x, y, z])
            resi_list.append(resi)

    return np.array(coords, dtype=np.float64), np.array(resi_list, dtype=np.int32)


# ============================================================================
#  DBSCAN clustering (pure numpy, no sklearn dependency at runtime)
# ============================================================================

def dbscan_distance_matrix(coords, eps, min_samples):
    """Simple DBSCAN on precomputed distance matrix.

    Args:
        coords: [N, 3] coordinates
        eps: distance threshold
        min_samples: minimum points for core point

    Returns:
        labels: [N] cluster labels (-1 = noise)
    """
    N = len(coords)
    if N == 0:
        return np.array([], dtype=np.int32)

    # Pairwise distance matrix
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))

    neighbors = [np.where(dist[i] <= eps)[0] for i in range(N)]
    is_core = np.array([len(nb) >= min_samples for nb in neighbors])

    labels = np.full(N, -1, dtype=np.int32)
    cluster_id = 0

    for i in range(N):
        if not is_core[i] or labels[i] != -1:
            continue

        queue = [i]
        labels[i] = cluster_id
        head = 0
        while head < len(queue):
            curr = queue[head]
            head += 1
            for nb in neighbors[curr]:
                if labels[nb] != -1:
                    continue
                labels[nb] = cluster_id
                if is_core[nb]:
                    queue.append(nb)

        cluster_id += 1

    return labels


# ============================================================================
#  Site scoring (following AF2BIND paper)
# ============================================================================

# Training set statistics for z-score (from AF2BIND paper's approach)
# These are approximate values; ideally computed from your training set.
TRAINING_MEAN_PBIND = 0.35
TRAINING_STD_PBIND = 0.15


def compute_site_scores(cluster_p_binds):
    """Compute site scores following AF2BIND paper.

    Returns:
        mean_p: mean P(bind) of cluster residues
        zscore: z-score of mean P(bind) relative to training set stats
        cdf_zscore: CDF of z-score (0-1, higher = more confident)
        cluster_rank: mean of top-N P(bind) values (N=min(23, cluster_size))
    """
    from scipy.stats import norm

    mean_p = np.mean(cluster_p_binds)
    zscore = (mean_p - TRAINING_MEAN_PBIND) / max(TRAINING_STD_PBIND, 1e-8)
    cdf_zscore = float(norm.cdf(zscore))

    n_top = min(23, len(cluster_p_binds))
    sorted_p = np.sort(cluster_p_binds)[::-1]
    cluster_rank = float(np.mean(sorted_p[:n_top]))

    return {
        "mean_p_bind": float(mean_p),
        "max_p_bind": float(np.max(cluster_p_binds)),
        "zscore": float(zscore),
        "cdf_zscore": cdf_zscore,
        "cluster_rank": cluster_rank,
    }


# ============================================================================
#  Process one protein
# ============================================================================

def process_one_protein(pred_csv, pdb_path, plddt_thresh, pbind_thresh,
                        dbscan_eps, dbscan_min_samples, min_site_residues):
    """Full pipeline for one protein: filter → cluster → score.

    Returns list of site dicts, or empty list if no sites found.
    """
    pred = load_predictions(pred_csv)
    ca_coords, ca_resis = parse_ca_coords(pdb_path, chain_id="A")

    L = len(pred["resi"])

    # Map pred residues to CA coords
    resi_to_coord = {}
    for i, resi in enumerate(ca_resis):
        resi_to_coord[resi] = ca_coords[i]

    # Step 1: pLDDT filter — zero out P(bind) for low-confidence residues
    p_bind = pred["p_bind"].copy()
    p_bind[pred["plddt"] < plddt_thresh] = 0.0

    # Step 2: Select residues above threshold
    above_thresh = np.where(p_bind > pbind_thresh)[0]
    if len(above_thresh) < dbscan_min_samples:
        return []

    # Get coordinates for selected residues
    selected_resis = pred["resi"][above_thresh]
    selected_coords = []
    valid_indices = []
    for j, idx in enumerate(above_thresh):
        resi = pred["resi"][idx]
        if resi in resi_to_coord:
            selected_coords.append(resi_to_coord[resi])
            valid_indices.append(j)

    if len(selected_coords) < dbscan_min_samples:
        return []

    selected_coords = np.array(selected_coords)
    valid_above_thresh = above_thresh[valid_indices]

    # Step 3: DBSCAN clustering
    labels = dbscan_distance_matrix(selected_coords, dbscan_eps, dbscan_min_samples)

    # Step 4: Build sites, discard small ones
    sites = []
    unique_labels = set(labels)
    unique_labels.discard(-1)

    for cid in sorted(unique_labels):
        members = np.where(labels == cid)[0]
        if len(members) < min_site_residues:
            continue

        orig_indices = valid_above_thresh[members]
        site_resis = pred["resi"][orig_indices]
        site_resns = pred["resn"][orig_indices]
        site_chains = pred["chain"][orig_indices]
        site_p_binds = p_bind[orig_indices]
        site_plddts = pred["plddt"][orig_indices]
        site_coords = selected_coords[members]

        scores = compute_site_scores(site_p_binds)

        site_center = site_coords.mean(axis=0)

        sites.append({
            "site_id": len(sites) + 1,
            "n_residues": len(members),
            "residues": list(zip(site_chains.tolist(), site_resis.tolist(),
                                 site_resns.tolist())),
            "center_x": float(site_center[0]),
            "center_y": float(site_center[1]),
            "center_z": float(site_center[2]),
            "mean_plddt": float(np.mean(site_plddts)),
            **scores,
        })

    # Sort by cdf_zscore descending
    sites.sort(key=lambda s: s["cdf_zscore"], reverse=True)
    for i, s in enumerate(sites):
        s["site_id"] = i + 1

    return sites


# ============================================================================
#  Write site outputs
# ============================================================================

def write_site_csv(sites, output_path):
    """Write per-protein site CSV."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "site_id", "n_residues", "mean_p_bind", "max_p_bind",
            "cdf_zscore", "cluster_rank", "zscore",
            "center_x", "center_y", "center_z", "mean_plddt",
            "residue_list",
        ])
        for s in sites:
            res_str = ";".join(f"{ch}:{ri}:{rn}"
                               for ch, ri, rn in s["residues"])
            writer.writerow([
                s["site_id"], s["n_residues"],
                f"{s['mean_p_bind']:.4f}", f"{s['max_p_bind']:.4f}",
                f"{s['cdf_zscore']:.4f}", f"{s['cluster_rank']:.4f}",
                f"{s['zscore']:.4f}",
                f"{s['center_x']:.2f}", f"{s['center_y']:.2f}",
                f"{s['center_z']:.2f}",
                f"{s['mean_plddt']:.1f}",
                res_str,
            ])


# ============================================================================
#  Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Cluster per-residue P(bind) predictions into binding sites."
    )
    parser.add_argument("--preds-dir", default=DEFAULT_PREDS_DIR)
    parser.add_argument("--pdb-dir", default=DEFAULT_PDB_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_SITES_DIR)
    parser.add_argument("--plddt-threshold", type=float, default=50.0)
    parser.add_argument("--pbind-threshold", type=float, default=0.28)
    parser.add_argument("--dbscan-eps", type=float, default=6.0,
                        help="DBSCAN distance threshold in Angstroms")
    parser.add_argument("--dbscan-min-samples", type=int, default=3)
    parser.add_argument("--min-site-residues", type=int, default=5)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-proteins", type=int, default=None)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    return parser.parse_args()


def discover_prediction_csvs(preds_dir):
    """Find all prediction CSVs in bucketed subdirectories."""
    entries = []
    for subdir in sorted(os.listdir(preds_dir)):
        subdir_path = os.path.join(preds_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for fname in sorted(os.listdir(subdir_path)):
            if not fname.endswith(".csv"):
                continue
            csv_path = os.path.join(subdir_path, fname)
            # Extract UniProt ID from filename: AF-{UID}-F1-model_v6.csv
            m = re.match(r"^AF-(.+)-F(\d+)-model_v(\d+)\.csv$", fname)
            if m:
                uid = m.group(1)
                entries.append((uid, fname, csv_path))
    return entries


def main():
    args = parse_args()
    assert 0 <= args.shard_id < args.num_shards

    all_entries = discover_prediction_csvs(args.preds_dir)
    if args.max_proteins:
        all_entries = all_entries[:args.max_proteins]

    total_before_shard = len(all_entries)
    if args.num_shards > 1:
        all_entries = [e for i, e in enumerate(all_entries)
                       if i % args.num_shards == args.shard_id]

    to_process = []
    for uid, fname, csv_path in all_entries:
        pdb_fname = fname.replace(".csv", ".pdb")
        pdb_path = os.path.join(args.pdb_dir, pdb_fname)
        if not os.path.isfile(pdb_path):
            continue
        site_fname = fname.replace(".csv", "_sites.csv")
        suffix = uid[-2:] if len(uid) >= 2 else "XX"
        site_path = os.path.join(args.output_dir, suffix, site_fname)
        if args.skip_existing and os.path.isfile(site_path):
            continue
        to_process.append((uid, csv_path, pdb_path, site_path))

    print("=" * 70)
    print("  Binding Site Clustering (DBSCAN)")
    print("=" * 70)
    print(f"  Preds dir      : {args.preds_dir}")
    print(f"  PDB dir        : {args.pdb_dir}")
    print(f"  Output dir     : {args.output_dir}")
    print(f"  pLDDT thresh   : {args.plddt_threshold}")
    print(f"  P(bind) thresh : {args.pbind_threshold}")
    print(f"  DBSCAN eps     : {args.dbscan_eps} Å, min_samples={args.dbscan_min_samples}")
    print(f"  Min site size  : {args.min_site_residues}")
    print(f"  Total preds    : {total_before_shard}")
    if args.num_shards > 1:
        print(f"  Shard          : {args.shard_id} / {args.num_shards}  "
              f"(this shard: {len(all_entries)})")
    print(f"  To process     : {len(to_process)}")
    print()

    if not to_process:
        print("  Nothing to process.")
        return

    total_sites = 0
    proteins_with_sites = 0
    summary_rows = []
    successes, failures = 0, 0
    t_start = time.time()

    for idx, (uid, csv_path, pdb_path, site_path) in enumerate(to_process):
        try:
            sites = process_one_protein(
                csv_path, pdb_path,
                plddt_thresh=args.plddt_threshold,
                pbind_thresh=args.pbind_threshold,
                dbscan_eps=args.dbscan_eps,
                dbscan_min_samples=args.dbscan_min_samples,
                min_site_residues=args.min_site_residues,
            )

            if sites:
                write_site_csv(sites, site_path)
                proteins_with_sites += 1
                total_sites += len(sites)
                for s in sites:
                    summary_rows.append({
                        "uniprot_id": uid,
                        "site_id": s["site_id"],
                        "n_residues": s["n_residues"],
                        "mean_p_bind": s["mean_p_bind"],
                        "max_p_bind": s["max_p_bind"],
                        "cdf_zscore": s["cdf_zscore"],
                        "cluster_rank": s["cluster_rank"],
                        "mean_plddt": s["mean_plddt"],
                    })

            successes += 1

            if (idx + 1) % 500 == 0:
                elapsed = time.time() - t_start
                rate = (idx + 1) / elapsed
                print(f"  [{idx + 1}/{len(to_process)}] "
                      f"{proteins_with_sites} proteins with sites, "
                      f"{total_sites} total sites, "
                      f"{rate:.0f} proteins/s")

        except Exception as e:
            print(f"  [{idx + 1}/{len(to_process)}] {uid}: FAILED — {e}")
            failures += 1

    # Write summary CSV
    if summary_rows:
        summary_path = os.path.join(
            args.output_dir,
            f"sites_summary_shard{args.shard_id}.csv"
            if args.num_shards > 1 else "sites_summary.csv"
        )
        os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "uniprot_id", "site_id", "n_residues", "mean_p_bind",
                "max_p_bind", "cdf_zscore", "cluster_rank", "mean_plddt",
            ])
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\n  Summary written to {summary_path}")

    total_time = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  DONE: {successes} succeeded, {failures} failed, "
          f"{total_time / 60:.1f} min total")
    print(f"  Proteins with sites: {proteins_with_sites}")
    print(f"  Total sites found:   {total_sites}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
