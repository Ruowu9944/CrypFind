"""
Train GVP or PockMon models on AF2BIND binding-site labels.

Reuses the model architecture from PocketMiner (MQAModel / PockMon) but
trains on AF2BIND's binding-site labels (dist_all < 5A) instead of
PocketMiner's cryptic-pocket labels.

Data source:
  - Labels:    af2bind/training/new_labels_2k.pkl
  - Splits:    af2bind/training/assignments_2k.pkl
  - TM-scores: af2bind/training/tmscores_2k.txt.gz  (for sample weights)
  - PDBs:      downloaded from RCSB, single chain per protein
  - Binder pair features: af2bind/binder_pair_all/{prot_id}.npz  (optional)

Metrics:
  - Training loss: BCE
  - Validation:    AF2BIND Recovery (top-k) and ROC AUC, both simple and
                   TM-score-weighted averages.

Usage:
    python train_af2bind.py config_af2bind.json

    # For PockMon with binder pair:
    python train_af2bind.py config_PocketMon_af2bind.json
"""

import argparse
import gc
import gzip
import json
import os
import pickle
import sys
import time
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

try:
    import wandb
except ModuleNotFoundError:
    wandb = None

import mdtraj as md
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from training.train_pocketminer import make_model
from crypfind.utils import save_checkpoint, load_checkpoint

# ============================================================================
#  Constants / Paths
# ============================================================================

AF2BIND_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets", "af2bind"
)
LABELS_PATH = os.path.join(AF2BIND_DIR, "training", "new_labels_2k.pkl")
ASSIGNMENTS_PATH = os.path.join(AF2BIND_DIR, "training", "assignments_2k.pkl")
TMSCORES_PATH = os.path.join(AF2BIND_DIR, "training", "tmscores_2k.txt.gz")

NONSTANDARD_TO_STANDARD = {
    "MSE": "MET", "PCA": "GLU", "CME": "CYS", "CSO": "CYS",
    "SEP": "SER", "TPO": "THR", "PTR": "TYR", "HYP": "PRO",
    "MLY": "LYS", "SAC": "SER", "DAL": "ALA", "AIB": "ALA",
    "CYM": "CYS", "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
    "HSE": "HIS", "HSD": "HIS", "HSP": "HIS",
}


# ============================================================================
#  Data loading
# ============================================================================

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_af2bind_data(seed=0):
    """Load AF2BIND labels, splits, and compute sample weights."""
    with open(LABELS_PATH, "rb") as f:
        labels = pickle.load(f)
    with open(ASSIGNMENTS_PATH, "rb") as f:
        assignments = pickle.load(f)

    assigned = assignments[seed]

    valid_ids = {k for k in assigned if k in labels}

    train_ids = sorted(k for k in valid_ids if assigned[k] == 0)
    val_ids = sorted(k for k in valid_ids if assigned[k] == 1)
    test_ids = sorted(k for k in valid_ids if assigned[k] == 2)

    # Compute TM-score based sample weights
    weights = _compute_sample_weights(assigned, valid_ids)

    return labels, train_ids, val_ids, test_ids, weights


def _compute_sample_weights(assigned, valid_ids):
    tms = {}
    with gzip.open(TMSCORES_PATH, "rt") as f:
        for line in f:
            a, b, tm_a, tm_b = line.rstrip().split()
            tmscore = max(float(tm_a), float(tm_b))
            tms.setdefault(a, {})[b] = tmscore
            tms.setdefault(b, {})[a] = tmscore

    all_prots = list(assigned.keys())
    weights = {}
    for a in all_prots:
        counts = 1
        for b in all_prots:
            if a != b and a in tms and b in tms.get(a, {}):
                if tms[a][b] > 0.5:
                    counts += 1
        weights[a] = 1.0 / counts
    return weights


def prepare_labels(labels, prot_id):
    """Prepare AF2BIND labels: binding residues and validity mask.

    Mirrors author's get_data mask logic:
      bind_all = dist_all < 5.0
      bind_sub = dist_sub < 5.0
      mask[bind_all & ~bind_sub] = False  (low-confidence binders masked)
    """
    y = labels[prot_id]
    y_bind_all = y["dist_all"] < 5.0
    y_bind_sub = y["dist_sub"] < 5.0
    y_mask = y["mask"].copy()
    y_mask[np.logical_and(y_bind_all, ~y_bind_sub)] = False
    return y_bind_all.astype(np.float32), y_mask.astype(np.float32)


# ============================================================================
#  PDB processing
# ============================================================================

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


def fix_nonstandard_residues(struc):
    for res in struc.top.residues:
        if res.name in NONSTANDARD_TO_STANDARD:
            res.name = NONSTANDARD_TO_STANDARD[res.name]
    return struc


def load_and_process_pdb(pdb_path, chain_id, num_atoms=4):
    """Load a PDB, extract single chain, compute (X, S, mask) features."""
    from crypfind.datasets import abbrev, lookup, _extract_5atom_xyz

    try:
        struc = md.load(pdb_path)
    except Exception as e:
        print(f"    WARNING: mdtraj failed to load {pdb_path}: {e}")
        return None, None, None, 0
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
        return None, None, None, 0

    chain_struc = struc.atom_slice(atom_indices)

    prot_iis = chain_struc.top.select(
        "protein and (name N or name CA or name C or name O)"
    )
    if len(prot_iis) == 0:
        return None, None, None, 0

    prot_bb = chain_struc.atom_slice(prot_iis)

    # Some PDB residues are missing backbone atoms (e.g. no O).  Keep only
    # residues that have exactly 4 backbone atoms so the reshape is safe.
    bb_names = {"N", "CA", "C", "O"}
    complete_res_idx = []
    for res in prot_bb.top.residues:
        atoms_in_res = {a.name for a in res.atoms}
        if bb_names.issubset(atoms_in_res) and len(atoms_in_res) == 4:
            complete_res_idx.append(res.index)

    if len(complete_res_idx) == 0:
        return None, None, None, 0

    if len(complete_res_idx) < prot_bb.top.n_residues:
        keep_atoms = []
        for res in prot_bb.top.residues:
            if res.index in set(complete_res_idx):
                keep_atoms.extend([a.index for a in res.atoms])
        prot_bb = prot_bb.atom_slice(keep_atoms)

    L = prot_bb.top.n_residues

    if num_atoms == 5:
        all_prot_iis = chain_struc.top.select("protein")
        full_prot = chain_struc.atom_slice(all_prot_iis)
        xyz = _extract_5atom_xyz(full_prot)
    else:
        xyz = prot_bb.xyz.reshape(L, 4, 3)

    seq = [r.name for r in prot_bb.top.residues]
    try:
        S = np.array([lookup[abbrev[a]] for a in seq], dtype=np.int32)
    except KeyError:
        return None, None, None, 0

    X = xyz.astype(np.float32)
    mask = np.ones(L, dtype=np.float32)
    return X, S, mask, L


# ============================================================================
#  Dataset: precompute all proteins
# ============================================================================

def precompute_af2bind_dataset(prot_ids, labels, pdb_dir, num_atoms=4):
    """Download PDBs and precompute (X, S, mask, y_bind, y_mask) for each protein.

    Returns dict: prot_id -> (X, S, mask, y_bind, y_mask)
    """
    data = {}
    failed = []

    for i, pid in enumerate(prot_ids):
        pdb_code = pid.split("_")[0]
        chain = pid.split("_")[1] if "_" in pid else "A"

        pdb_path = download_pdb(pdb_code, pdb_dir)
        if pdb_path is None:
            failed.append(pid)
            continue

        X, S, mask, L = load_and_process_pdb(pdb_path, chain, num_atoms)
        if X is None:
            failed.append(pid)
            continue

        y_bind, y_mask = prepare_labels(labels, pid)

        if L != len(y_bind):
            min_L = min(L, len(y_bind))
            X = X[:min_L]
            S = S[:min_L]
            mask = mask[:min_L]
            y_bind = y_bind[:min_L]
            y_mask = y_mask[:min_L]

        data[pid] = (X, S, mask, y_bind, y_mask)

        if (i + 1) % 100 == 0:
            print(f"  Preprocessed {i+1}/{len(prot_ids)} proteins "
                  f"({len(failed)} failed)")

    if failed:
        print(f"  WARNING: {len(failed)} proteins failed: {failed[:10]}...")

    return data


# ============================================================================
#  Metrics
# ============================================================================

def compute_recovery(pred, y_bind, y_mask):
    valid = y_mask.astype(bool)
    pred_v = pred[valid]
    true_v = y_bind[valid]
    top_k = int(true_v.sum())
    if top_k == 0:
        return float('nan')
    sorted_idx = pred_v.argsort()[::-1][:top_k]
    return true_v[sorted_idx].mean()


def compute_roc_auc(pred, y_bind, y_mask):
    valid = y_mask.astype(bool)
    pred_v = pred[valid]
    true_v = y_bind[valid]
    if true_v.sum() == 0 or true_v.sum() == len(true_v):
        return float('nan')
    return roc_auc_score(true_v, pred_v)


def evaluate_model(model, dataset, weights, device, num_atoms=4):
    """Evaluate model on AF2BIND dataset with Recovery and ROC AUC.

    Both ``simple`` (unweighted mean across proteins) and ``weighted``
    (TM-score sample-weighted mean across proteins) variants are returned.
    The weighted ROC AUC is the metric that strictly matches the AF2BIND
    paper-reported value (~0.936 on test seed=0).
    """
    model.eval()
    recoveries = []
    rec_weights = []
    aucs = []
    auc_weights = []

    with torch.no_grad():
        for pid, (X, S, mask_np, y_bind, y_mask) in dataset.items():
            X_t = torch.from_numpy(X).unsqueeze(0).to(device)
            S_t = torch.from_numpy(S).unsqueeze(0).to(device)
            mask_t = torch.from_numpy(mask_np).unsqueeze(0).to(device)

            probs = model(X_t, S_t, mask_t, train=False, res_level=True, meta=[pid])
            pred = probs[0, :len(y_bind)].cpu().numpy()

            rec = compute_recovery(pred, y_bind, y_mask)
            auc = compute_roc_auc(pred, y_bind, y_mask)
            w = weights.get(pid, 1.0)

            if not np.isnan(rec):
                recoveries.append(rec)
                rec_weights.append(w)
            if not np.isnan(auc):
                aucs.append(auc)
                auc_weights.append(w)

    rec_arr = np.array(recoveries)
    auc_arr = np.array(aucs)
    rw_arr = np.array(rec_weights)
    aw_arr = np.array(auc_weights)

    simple_rec = rec_arr.mean() if len(rec_arr) > 0 else 0.0
    simple_auc = auc_arr.mean() if len(auc_arr) > 0 else 0.0
    weighted_rec = (
        (rec_arr * rw_arr).sum() / rw_arr.sum() if len(rw_arr) > 0 else 0.0
    )
    weighted_auc = (
        (auc_arr * aw_arr).sum() / aw_arr.sum() if len(aw_arr) > 0 else 0.0
    )

    return {
        "recovery_simple": float(simple_rec),
        "recovery_weighted": float(weighted_rec),
        "roc_auc_simple": float(simple_auc),
        "roc_auc_weighted": float(weighted_auc),
        "n_evaluated": len(recoveries),
    }


# ============================================================================
#  Training loop
# ============================================================================

def train_one_epoch(model, train_data, optimizer, device, num_atoms=4,
                    sample_weights=None):
    """One epoch of training over the AF2BIND train split.

    Parameters
    ----------
    sample_weights : Optional[Dict[str, float]]
        If ``None`` (default), behaviour is **byte-for-byte identical** to the
        original plain-BCE training loop:  per-protein loss is the mean BCE
        over valid residues, gradients are not rescaled.

        If provided, applies AF2BIND-style TM-score sample reweighting at the
        protein level.  Per-protein loss becomes ``mean_residue_BCE * w_pid``
        where ``w_pid`` is taken from the dict (default 1.0 if missing).

        IMPORTANT: callers should pre-normalize the dict so that the *mean
        weight over the training set is ~1.0*; this preserves the average
        gradient magnitude and avoids implicit learning-rate retuning.  See
        ``main()`` for the normalization step.
    """
    model.train()
    total_loss = 0.0
    n_samples = 0
    pids = list(train_data.keys())
    np.random.shuffle(pids)

    for pid in pids:
        X, S, mask_np, y_bind, y_mask = train_data[pid]

        X_t = torch.from_numpy(X).unsqueeze(0).to(device)
        S_t = torch.from_numpy(S).unsqueeze(0).to(device)
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).to(device)
        y_t = torch.from_numpy(y_bind).unsqueeze(0).to(device)
        ymask_t = torch.from_numpy(y_mask).unsqueeze(0).to(device).bool()

        probs = model(X_t, S_t, mask_t, train=True, res_level=True, meta=[pid])
        probs_valid = probs[ymask_t]
        labels_valid = y_t[ymask_t]

        if probs_valid.numel() == 0:
            continue

        loss = F.binary_cross_entropy(probs_valid, labels_valid)
        if sample_weights is not None:
            w_pid = float(sample_weights.get(pid, 1.0))
            loss = loss * w_pid
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * probs_valid.numel()
        n_samples += probs_valid.numel()

    return total_loss / max(n_samples, 1)


# ============================================================================
#  Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train GVP/PockMon on AF2BIND binding-site labels."
    )
    parser.add_argument("config", help="Path to JSON config.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override seed in config. Output dir and wandb "
                             "run name are auto-adjusted to include seed id.")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.seed is not None:
        config["training"]["seed"] = args.seed
        base_out = config["output"]["output_dir"].rstrip("/")
        config["output"]["output_dir"] = f"{base_out}_seed{args.seed}"
        config["output"]["checkpoint_dir"] = os.path.join(
            config["output"]["output_dir"], "checkpoints",
        )
        wb = config.get("wandb", {})
        if wb.get("run_name"):
            wb["run_name"] = f"{wb['run_name']}_seed{args.seed}"

    seed = config["training"].get("seed", 0)
    num_epochs = config["training"]["num_epochs"]
    lr = config["training"]["learning_rate"]
    num_atoms = 4 if config["model"].get("ablate_sidechain_vectors", True) else 5

    pdb_dir = config.get("af2bind", {}).get(
        "pdb_dir", os.path.join(AF2BIND_DIR, "all_pdbs")
    )
    os.makedirs(pdb_dir, exist_ok=True)

    output_dir = config["output"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    ckpt_dir = config["output"].get("checkpoint_dir", os.path.join(output_dir, "checkpoints"))
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save config
    with open(os.path.join(output_dir, "used_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load data ---
    print("Loading AF2BIND data...")
    labels, train_ids, val_ids, test_ids, weights = load_af2bind_data(seed)
    print(f"  Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    cache_path = os.path.join(output_dir, f"af2bind_cache_{num_atoms}atom.pkl")
    if os.path.isfile(cache_path):
        print(f"  Loading cached data from {cache_path}...")
        with open(cache_path, "rb") as f:
            all_data = pickle.load(f)
        train_data = {k: v for k, v in all_data.items() if k in set(train_ids)}
        val_data = {k: v for k, v in all_data.items() if k in set(val_ids)}
        test_data = {k: v for k, v in all_data.items() if k in set(test_ids)}
    else:
        print("  Preprocessing training proteins...")
        train_data = precompute_af2bind_dataset(train_ids, labels, pdb_dir, num_atoms)
        print("  Preprocessing validation proteins...")
        val_data = precompute_af2bind_dataset(val_ids, labels, pdb_dir, num_atoms)
        print("  Preprocessing test proteins...")
        test_data = precompute_af2bind_dataset(test_ids, labels, pdb_dir, num_atoms)

        all_data = {**train_data, **val_data, **test_data}
        print(f"  Caching preprocessed data to {cache_path}...")
        with open(cache_path, "wb") as f:
            pickle.dump(all_data, f, protocol=4)

    print(f"  Loaded: train={len(train_data)}, val={len(val_data)}, "
          f"test={len(test_data)}")

    # --- Build model ---
    print("Building model...")
    model = make_model(config)
    model.to(device)

    # Warmup: run a dummy forward pass to initialize LazyLinear layers (GVP)
    has_lazy = any(
        isinstance(m, torch.nn.LazyLinear) for m in model.modules()
    )
    if has_lazy and train_data:
        print("  Initializing LazyLinear layers with dummy forward pass...")
        first_pid = next(iter(train_data))
        X0, S0, m0, _, _ = train_data[first_pid]
        with torch.no_grad():
            model(
                torch.from_numpy(X0).unsqueeze(0).to(device),
                torch.from_numpy(S0).unsqueeze(0).to(device),
                torch.from_numpy(m0).unsqueeze(0).to(device),
                train=False, res_level=True, meta=[first_pid],
            )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model type: {config['model'].get('model_type', 'gvp')}")
    print(f"  Trainable params: {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr,
        weight_decay=config["training"].get("weight_decay", 0),
    )

    scheduler = None
    sched_cfg = config["training"].get("scheduler")
    if sched_cfg and sched_cfg.get("type") == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=sched_cfg.get("mode", "max"),
            factor=sched_cfg.get("factor", 0.5),
            patience=sched_cfg.get("patience", 5),
            threshold=sched_cfg.get("threshold", 1e-4),
            min_lr=sched_cfg.get("min_lr", 1e-7),
        )

    # --- Optional pretrained PockMon backbone ---
    pretrained_path = config.get("training", {}).get("pretrained_checkpoint")
    if pretrained_path:
        load_checkpoint(model, optimizer, pretrained_path, scheduler=scheduler)
        print(f"PRETRAINED CHECKPOINT LOADED: {pretrained_path}")
    # --- WandB ---
    wb_cfg = config.get("wandb", {})
    use_wandb = wb_cfg.get("enabled", False) and wandb is not None
    if use_wandb:
        wandb.init(
            project=wb_cfg.get("project", "af2bind_training"),
            name=wb_cfg.get("run_name", "af2bind_run"),
            config=config,
        )

    # ------------------------------------------------------------------
    # Optional AF2BIND-style TM-score sample reweighting (training loss)
    # ------------------------------------------------------------------
    # Activated by ``training.use_tm_sample_weight: true`` in the JSON config.
    # When the key is absent/false we pass ``None`` -> behaviour is byte-for-
    # byte identical to the original plain-BCE loop (no scaling, no LR drift).
    #
    # When enabled we *normalize* the per-protein weights so the mean over
    # the training split equals 1.0; this matches AF2BIND's batch-normalized
    # weighted average ``Σ(bce*w)/Σ(w)`` in expectation while keeping the
    # average gradient magnitude the same as plain BCE (so no implicit LR
    # change).  PocketMiner's ``train.py`` is *not* touched by this code path
    # at all; this branch only affects ``train_af2bind.py``.
    use_tm_weight = bool(
        config["training"].get("use_tm_sample_weight", False)
    )
    train_sample_weights = None
    if use_tm_weight:
        raw_w = {pid: float(weights.get(pid, 1.0)) for pid in train_data.keys()}
        if len(raw_w) == 0:
            print("  [TM weight] WARNING: empty train set; weighting disabled.")
        else:
            mean_w = float(np.mean(list(raw_w.values()))) or 1.0
            train_sample_weights = {pid: w / mean_w for pid, w in raw_w.items()}
            w_vals = np.array(list(train_sample_weights.values()))
            print(f"  [TM weight] enabled: N={len(train_sample_weights)} "
                  f"raw mean={mean_w:.4f}  "
                  f"normalized min/median/max={w_vals.min():.3f}/"
                  f"{np.median(w_vals):.3f}/{w_vals.max():.3f}")
    else:
        print("  [TM weight] disabled (plain BCE).")

    # --- Training ---
    best_val_auc_w = -1.0
    best_epoch = 0

    print(f"\n{'='*70}")
    print(f"  Starting training: {num_epochs} epochs")
    print(f"{'='*70}\n")

    for epoch in range(num_epochs):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_data, optimizer, device, num_atoms,
            sample_weights=train_sample_weights,
        )

        val_metrics = evaluate_model(
            model, val_data, weights, device, num_atoms,
        )
        epoch_time = time.time() - t0

        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:3d} | loss={train_loss:.4f} | "
              f"val_rec={val_metrics['recovery_simple']:.4f} "
              f"(w={val_metrics['recovery_weighted']:.4f}) | "
              f"val_auc={val_metrics['roc_auc_simple']:.4f} "
              f"(w={val_metrics['roc_auc_weighted']:.4f}) | "
              f"lr={current_lr:.2e} | {epoch_time:.1f}s")

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "val/recovery_simple": val_metrics["recovery_simple"],
                "val/recovery_weighted": val_metrics["recovery_weighted"],
                "val/roc_auc_simple": val_metrics["roc_auc_simple"],
                "val/roc_auc_weighted": val_metrics["roc_auc_weighted"],
                "lr": current_lr,
            })

        if scheduler:
            scheduler.step(val_metrics["roc_auc_weighted"])

        if val_metrics["roc_auc_weighted"] > best_val_auc_w:
            best_val_auc_w = val_metrics["roc_auc_weighted"]
            best_epoch = epoch
            save_checkpoint(
                os.path.join(ckpt_dir, "best_model.pt"),
                model, optimizer,
                epoch=epoch, best_pr_auc=best_val_auc_w, best_epoch=best_epoch,
            )

        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt"),
                model, optimizer,
                epoch=epoch,
            )

    # --- Final evaluation on test set ---
    print(f"\n{'='*70}")
    print(f"  Training complete. Best val AUC(w): {best_val_auc_w:.4f} "
          f"(epoch {best_epoch})")
    print(f"  Loading best model for test evaluation...")

    load_checkpoint(model, optimizer, os.path.join(ckpt_dir, "best_model.pt"))
    model.to(device)

    test_metrics = evaluate_model(
        model, test_data, weights, device, num_atoms,
    )

    print(f"\n  TEST RESULTS:")
    print(f"    Recovery (simple)  : {test_metrics['recovery_simple']:.4f}")
    print(f"    Recovery (weighted): {test_metrics['recovery_weighted']:.4f}")
    print(f"    ROC AUC  (simple)  : {test_metrics['roc_auc_simple']:.4f}")
    print(f"    ROC AUC  (weighted): {test_metrics['roc_auc_weighted']:.4f}  "
          f"<-- AF2BIND paper-comparable")
    print(f"    N evaluated        : {test_metrics['n_evaluated']}")

    # Save results
    results = {
        "best_epoch": best_epoch,
        "best_val_auc_weighted": best_val_auc_w,
        "test": test_metrics,
    }
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    if use_wandb:
        wandb.log({"test/recovery_simple": test_metrics["recovery_simple"],
                    "test/recovery_weighted": test_metrics["recovery_weighted"],
                    "test/roc_auc_simple": test_metrics["roc_auc_simple"],
                    "test/roc_auc_weighted": test_metrics["roc_auc_weighted"]})
        wandb.finish()

    print(f"{'='*70}")


if __name__ == "__main__":
    main()
