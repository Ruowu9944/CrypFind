import argparse
import json
import math
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import wandb
except ModuleNotFoundError:
    wandb = None

from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.metrics import auc as sklearn_auc

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from crypfind.datasets import (
    PrecomputedLoader,
    determine_global_weights,
    load_label_dictionary,
    precompute_all_samples,
    process_apo_ids,
    simulation_dataset,
)
from crypfind.utils import save_checkpoint, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal PocketMiner training entrypoint.")
    parser.add_argument("config", help="Path to JSON config.")
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def get_num_atoms(config):
    """Derive atom count from model config: 5 when sidechain vectors are used, else 4."""
    return 4 if config["model"].get("ablate_sidechain_vectors", True) else 5


def make_model(config):
    model_cfg = config["model"]
    model_type = model_cfg.get("model_type", "gvp")

    if model_type == "pockmon":
        from crypfind.backbone import PockMonModel
        return PockMonModel(
            cutoff=model_cfg.get("cutoff", 1.5),
            n_atom_basis=model_cfg.get("n_atom_basis", 128),
            n_rbf=model_cfg.get("n_rbf", 32),
            n_pos_enc=model_cfg.get("n_pos_enc", 16),
            n_interactions=model_cfg.get("n_interactions", 4),
            num_heads=model_cfg.get("num_heads", 4),
            dropout=model_cfg.get("dropout_rate", 0.1),
            lmax=model_cfg.get("lmax", 2),
            c_pair=model_cfg.get("c_pair", 128),
            c_pair_proj=model_cfg.get("c_pair_proj", 64),
            pair_repr_dir=model_cfg.get("pair_repr_dir"),
            binder_pair_dir=model_cfg.get("binder_pair_dir"),
            combined_pair_dir=model_cfg.get("combined_pair_dir"),
            c_binder_in=model_cfg.get("c_binder_in", 5120),
            init_steerable_from_backbone=model_cfg.get(
                "init_steerable_from_backbone", True
            ),
            use_pair_fusion=model_cfg.get("use_pair_fusion", True),
            use_edge_conditioning=model_cfg.get("use_edge_conditioning", False),
            max_num_neighbors=model_cfg.get("max_num_neighbors", 128),
            layernorm=model_cfg.get("layernorm", "layer"),
            steerable_norm=model_cfg.get("steerable_norm", "layer"),
            edge_updates=model_cfg.get("edge_updates", False),
            sep_htr=model_cfg.get("sep_htr", True),
            sep_dir=model_cfg.get("sep_dir", True),
            sep_tensor=model_cfg.get("sep_tensor", True),
            cls_hidden_dim=model_cfg.get("cls_hidden_dim", 256),
            fusion_strategy=model_cfg.get("fusion_strategy", "late"),
            pair_logit_dropout=model_cfg.get("pair_logit_dropout", 0.0),
            pair_logit_layernorm=model_cfg.get("pair_logit_layernorm", True),
            residual_init_zero=model_cfg.get("residual_init_zero", True),
            use_dihedral=model_cfg.get("use_dihedral", True),
            use_fusion_multiply=model_cfg.get("use_fusion_multiply", True),
            dilated_sequence_edges=model_cfg.get("dilated_sequence_edges"),
            edge_type_dim=model_cfg.get("edge_type_dim", 0),
        )

    raise ValueError("Only the CrypFind geometric encoder is supported")


# ---------------------------------------------------------------------------
#  wandb helpers
# ---------------------------------------------------------------------------

def init_wandb(config):
    """Initialise a wandb run if enabled in config and the library is available."""
    wb_cfg = config.get("wandb", {})
    enabled = wb_cfg.get("enabled", False)
    if not enabled:
        return False
    if wandb is None:
        print("WARNING: wandb enabled in config but the package is not installed. "
              "Install with `pip install wandb`. Falling back to local-only logging.")
        return False

    wb_config = {
        "model": config["model"],
        "training": config["training"],
        "data_filestem": config["data"]["filestem"],
    }
    if "finetune" in config:
        wb_config["finetune"] = config["finetune"]
    wandb.init(
        project=wb_cfg.get("project", "pocketminer"),
        entity=wb_cfg.get("entity"),
        name=wb_cfg.get("run_name") or config["output"].get("run_name"),
        config=wb_config,
        reinit="finish_previous",
    )
    wandb.define_metric("batch_step")
    wandb.define_metric("train/batch_loss", step_metric="batch_step")
    return True


def wb_log(metrics: dict, **kwargs):
    """Thin wrapper: only calls wandb.log when wandb is active."""
    if wandb is not None and wandb.run is not None:
        wandb.log(metrics, **kwargs)


def wb_summary(key, value):
    if wandb is not None and wandb.run is not None:
        wandb.run.summary[key] = value


# ---------------------------------------------------------------------------
#  Prediction / evaluation helpers
# ---------------------------------------------------------------------------

def predict_on_xtals(model, apo_ids, apo_structure_dir, device, num_atoms=4):
    X, S, mask = process_apo_ids(apo_ids, apo_structure_dir, num_atoms=num_atoms)

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    S_t = torch.tensor(S, dtype=torch.long, device=device)
    M_t = torch.tensor(mask, dtype=torch.float32, device=device)

    meta = [str(aid) for aid in apo_ids]

    model.eval()
    with torch.no_grad():
        predictions = model(X_t, S_t, M_t, train=False, res_level=True,
                            meta=meta).cpu().numpy()

    mask_bool = mask.astype(bool)
    return predictions, mask_bool


def _resolve_label_key(apo_id, label_dict):
    candidates = [
        apo_id[:-1],
        apo_id,
        apo_id[:-1].upper(),
        apo_id[:-1].lower(),
        apo_id.upper(),
        apo_id.lower(),
    ]
    for c in candidates:
        if c in label_dict:
            return c
    raise KeyError(
        f"Cannot find label for apo ID '{apo_id}' in label_dict "
        f"(tried: {candidates})"
    )


def compute_recovery(predictions, apo_ids, label_dict):
    """Compute mean per-protein Recovery without using a score threshold.

    For each protein, K is the number of positive residues (label == 1).
    The top-K residues are selected by model score over the whole protein, and
    Recovery is the fraction of those top-K residues that are truly positive.
    Proteins with K=0 return NaN and are excluded from the mean.

    Note: PocketMiner validation/test label dictionaries often mark most
    non-pocket residues as label 2. Excluding label 2 before top-K ranking makes
    positive proteins contain only positive labels and trivially yields
    Recovery=1.0. For Recovery@K, label 2 should be treated as non-positive
    candidate positions rather than removed from the ranking pool.
    """
    protein_recoveries = []

    for apo_id, preds in zip(apo_ids, predictions):
        pdb_key = _resolve_label_key(apo_id, label_dict)
        y_raw = np.asarray(label_dict[pdb_key])

        n = min(len(y_raw), len(preds))
        y_bin = (y_raw[:n] == 1).astype(np.float32)
        p = preds[:n]

        k = int(np.sum(y_bin))
        if k == 0:
            protein_recoveries.append(float("nan"))
            continue

        top_k_idx = np.argsort(p)[::-1][:k]
        recovery = float(np.sum(y_bin[top_k_idx])) / k
        protein_recoveries.append(recovery)

    valid_recoveries = [r for r in protein_recoveries if not np.isnan(r)]
    mean_recovery = float(np.mean(valid_recoveries)) if valid_recoveries else 0.0
    return mean_recovery, protein_recoveries


def assess_performance(predictions, mask, apo_ids, label_dict, test=False):
    """Compute validation / test metrics.

    Label convention (per author's README & evaluation scripts):
        0 = negative (confirmed non-pocket residues),
        1 = positive (confirmed cryptic-pocket residues),
        2 = uncertain / excluded (not classified with high certainty).
    Only residues with label 0 or 1 participate in metric computation.
    """
    true_labels_raw = []
    for apo_id in apo_ids:
        pdb_key = _resolve_label_key(apo_id, label_dict)
        true_labels_raw.append(label_dict[pdb_key])

    protein_aucs, protein_pr_aucs = [], []
    all_y_true, all_y_pred = [], []

    for y_raw, preds in zip(true_labels_raw, predictions):
        n = len(y_raw)
        p = preds[:n]
        valid = (y_raw == 0) | (y_raw == 1)
        y_bin = (y_raw[valid] == 1).astype(np.float32)
        p_valid = p[valid]

        if len(np.unique(y_bin)) < 2:
            protein_aucs.append(float("nan"))
            protein_pr_aucs.append(float("nan"))
        else:
            protein_aucs.append(roc_auc_score(y_bin, p_valid))
            prec_arr, rec_arr, _ = precision_recall_curve(y_bin, p_valid)
            protein_pr_aucs.append(sklearn_auc(rec_arr, prec_arr))

        all_y_true.extend(y_bin.tolist())
        all_y_pred.extend(p_valid.tolist())

    y_true_all = np.array(all_y_true, dtype=np.float32)
    y_pred_all = np.array(all_y_pred, dtype=np.float32)

    loss = F.binary_cross_entropy(
        torch.tensor(y_pred_all), torch.tensor(y_true_all)
    ).item()

    overall_auc = roc_auc_score(y_true_all, y_pred_all)
    prec_arr, rec_arr, _ = precision_recall_curve(y_true_all, y_pred_all)
    overall_pr_auc = sklearn_auc(rec_arr, prec_arr)

    if test:
        y_pred_bin = (y_pred_all >= 0.5).astype(np.float32)
        tp = float(np.sum((y_pred_bin == 1) & (y_true_all == 1)))
        fp = float(np.sum((y_pred_bin == 1) & (y_true_all == 0)))
        tn = float(np.sum((y_pred_bin == 0) & (y_true_all == 0)))
        fn = float(np.sum((y_pred_bin == 0) & (y_true_all == 1)))
        acc = (tp + tn) / max(tp + fp + tn + fn, 1)
        prec_val = tp / max(tp + fp, 1)
        recall_val = tp / max(tp + fn, 1)
        return (loss, tp, fp, tn, fn, acc, prec_val, recall_val,
                overall_auc, overall_pr_auc, y_pred_all, y_true_all,
                protein_aucs, protein_pr_aucs)

    return (loss, overall_auc, overall_pr_auc, y_pred_all, y_true_all,
            protein_aucs, protein_pr_aucs)


# ---------------------------------------------------------------------------
#  Index / weight selection helpers (unchanged)
# ---------------------------------------------------------------------------

def valid_mask(y):
    return np.array(y) >= 0


def get_indices(y, config):
    train_cfg = config["training"]
    pos_thresh = train_cfg["pos_thresh"]
    train_on_intermediates = train_cfg["train_on_intermediates"]
    neg_thresh = train_cfg.get("neg_thresh", pos_thresh)
    y_np = np.array(y)
    valid = valid_mask(y_np)
    if train_on_intermediates:
        selected = valid
    else:
        selected = valid & ((y_np >= pos_thresh) | (y_np < neg_thresh))
    struct_indices, residue_indices = np.where(selected)
    return [[int(si), int(ri)] for si, ri in zip(struct_indices, residue_indices)]


def use_global_weights(y, positive_weight, negative_weight, config):
    train_cfg = config["training"]
    pos_thresh = train_cfg["pos_thresh"]
    train_on_intermediates = train_cfg["train_on_intermediates"]
    neg_thresh = train_cfg.get("neg_thresh", pos_thresh)
    y_np = np.array(y)
    valid = valid_mask(y_np)
    if train_on_intermediates:
        selected = valid
    else:
        selected = valid & ((y_np >= pos_thresh) | (y_np < neg_thresh))

    struct_indices, residue_indices = np.where(selected)
    weights = []
    for si, ri in zip(struct_indices, residue_indices):
        weights.append(positive_weight if y_np[si, ri] >= pos_thresh else negative_weight)
    return [[int(si), int(ri)] for si, ri in zip(struct_indices, residue_indices)], weights


def get_weights(y, config):
    train_cfg = config["training"]
    pos_thresh = train_cfg["pos_thresh"]
    train_on_intermediates = train_cfg["train_on_intermediates"]
    neg_thresh = train_cfg.get("neg_thresh", pos_thresh)
    y_np = np.array(y)
    iis = []
    weights = []

    for struct_index, example in enumerate(y_np):
        valid = example >= 0
        positive_mask = valid & (example >= pos_thresh)
        if train_on_intermediates:
            negative_mask = valid & (example < pos_thresh)
        else:
            negative_mask = valid & (example < neg_thresh)

        pos_count = int(positive_mask.sum())
        neg_count = int(negative_mask.sum())

        if pos_count == 0 or neg_count == 0:
            selected = np.where(valid)[0] if train_on_intermediates else np.where(positive_mask | negative_mask)[0]
            for res_index in selected:
                iis.append([struct_index, int(res_index)])
                weights.append(1.0)
            continue

        if train_on_intermediates:
            selected = np.where(valid)[0]
        else:
            selected = np.where(positive_mask | negative_mask)[0]

        total = pos_count + neg_count
        for res_index in selected:
            iis.append([struct_index, int(res_index)])
            weights.append(
                1 / pos_count * (total / 2.0)
                if example[res_index] >= pos_thresh
                else 1 / neg_count * (total / 2.0)
            )

    return iis, weights


def choose_balanced_inds_constant_size(y, n_residues, config):
    train_cfg = config["training"]
    pos_thresh = train_cfg["pos_thresh"]
    train_on_intermediates = train_cfg["train_on_intermediates"]
    neg_thresh = train_cfg.get("neg_thresh", pos_thresh)
    y_np = np.array(y)
    valid = valid_mask(y_np)
    pos_mask = valid & (y_np >= pos_thresh)
    neg_mask = valid & ((y_np < pos_thresh) if train_on_intermediates else (y_np < neg_thresh))

    positive_count = int(pos_mask.sum())
    negative_count = int(neg_mask.sum())
    if positive_count > 0 and negative_count > 0:
        pos_indices = np.array(list(zip(*np.where(pos_mask))))
        neg_indices = np.array(list(zip(*np.where(neg_mask))))
        target_size = int(n_residues / 2)
        pos_selection = (
            pos_indices[np.random.choice(range(positive_count), target_size, replace=positive_count < target_size)]
            if positive_count != target_size
            else pos_indices
        )
        neg_selection = (
            neg_indices[np.random.choice(range(negative_count), target_size, replace=negative_count < target_size)]
            if negative_count != target_size
            else neg_indices
        )
        selection = np.concatenate((pos_selection, neg_selection))
        np.random.shuffle(selection)
        return selection.tolist()

    valid_indices = np.array(list(zip(*np.where(valid))))
    if len(valid_indices) > n_residues:
        valid_indices = valid_indices[np.random.choice(range(len(valid_indices)), n_residues, replace=False)]
    return valid_indices.tolist()


def choose_balanced_inds_oversampling(y, config):
    train_cfg = config["training"]
    pos_thresh = train_cfg["pos_thresh"]
    train_on_intermediates = train_cfg["train_on_intermediates"]
    neg_thresh = train_cfg.get("neg_thresh", pos_thresh)
    y_np = np.array(y)
    valid = valid_mask(y_np)
    pos_mask = valid & (y_np >= pos_thresh)
    neg_mask = valid & ((y_np < pos_thresh) if train_on_intermediates else (y_np < neg_thresh))
    positive_count = int(pos_mask.sum())
    negative_count = int(neg_mask.sum())
    if positive_count > 0 and negative_count > 0:
        pos_indices = np.array(list(zip(*np.where(pos_mask))))
        neg_indices = np.array(list(zip(*np.where(neg_mask))))
        pos_selection = (
            pos_indices
            if positive_count >= negative_count
            else pos_indices[np.random.choice(range(positive_count), negative_count)]
        )
        neg_selection = (
            neg_indices
            if positive_count <= negative_count
            else neg_indices[np.random.choice(range(negative_count), positive_count)]
        )
        selection = np.concatenate((pos_selection, neg_selection))
        np.random.shuffle(selection)
        return selection.tolist()
    return np.array(list(zip(*np.where(valid)))).tolist()


def choose_balanced_inds_undersampling(y, config):
    train_cfg = config["training"]
    pos_thresh = train_cfg["pos_thresh"]
    train_on_intermediates = train_cfg["train_on_intermediates"]
    neg_thresh = train_cfg.get("neg_thresh", pos_thresh)
    y_np = np.array(y)
    valid = valid_mask(y_np)
    pos_mask = valid & (y_np >= pos_thresh)
    neg_mask = valid & ((y_np < pos_thresh) if train_on_intermediates else (y_np < neg_thresh))
    positive_count = int(pos_mask.sum())
    negative_count = int(neg_mask.sum())
    if positive_count > 0 and negative_count > 0:
        pos_indices = np.array(list(zip(*np.where(pos_mask))))
        neg_indices = np.array(list(zip(*np.where(neg_mask))))
        pos_selection = (
            pos_indices
            if positive_count <= negative_count
            else pos_indices[np.random.choice(range(positive_count), negative_count, replace=False)]
        )
        neg_selection = (
            neg_indices
            if positive_count >= negative_count
            else neg_indices[np.random.choice(range(negative_count), positive_count, replace=False)]
        )
        selection = np.concatenate((pos_selection, neg_selection))
        np.random.shuffle(selection)
        return selection.tolist()
    return np.array(list(zip(*np.where(valid)))).tolist()


def select_indices(y, config, positive_weight=None, negative_weight=None):
    train_cfg = config["training"]
    if train_cfg["balance_classes"]:
        if train_cfg["oversample"]:
            return choose_balanced_inds_oversampling(y, config), None
        if train_cfg["undersample"]:
            return choose_balanced_inds_undersampling(y, config), None
        if train_cfg["constant_size_balanced_sets"]:
            return choose_balanced_inds_constant_size(
                y, train_cfg["number_residues_per_draw"], config
            ), None
    if train_cfg["weight_loss"]:
        if train_cfg["weight_globally"]:
            return use_global_weights(y, positive_weight, negative_weight, config)
        return get_weights(y, config)
    return get_indices(y, config), None


def threshold_labels(y, config):
    if config["training"].get("use_continuous_labels", False):
        return y.clamp(0.0, 1.0)
    pos_thresh = config["training"]["pos_thresh"]
    return (y >= pos_thresh).float()


# ---------------------------------------------------------------------------
#  Training loops
# ---------------------------------------------------------------------------

def train_residue_batches(dataset, model, optimizer, loss_fn, config, device,
                          positive_weight=1.0, negative_weight=1.0,
                          global_step=0, log_interval=10):
    losses = []
    y_pred, y_true = [], []
    residues_per_batch = config["training"]["number_residues_per_batch"]
    batch_idx = 0
    t0 = time.time()

    for batch in dataset:
        batch_t0 = time.time()
        X, S, y, _meta, M = batch
        indices, weights = select_indices(y, config, positive_weight, negative_weight)
        if not indices:
            batch_idx += 1
            continue
        num_batches = int(math.ceil(len(indices) / residues_per_batch))
        index_splits = list(split(indices, num_batches))
        weight_splits = list(split(weights, num_batches)) if weights is not None else [None] * len(index_splits)
        batch_losses = []

        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        S_t = torch.tensor(S, dtype=torch.long, device=device)
        M_t = torch.tensor(M, dtype=torch.float32, device=device)
        y_t = torch.tensor(y, dtype=torch.float32, device=device)

        for i, batch_indices in enumerate(index_splits):
            optimizer.zero_grad()
            prediction = model(X_t, S_t, M_t, train=True, res_level=True)

            idx = torch.tensor(batch_indices, dtype=torch.long, device=device)
            y_sel = y_t[idx[:, 0], idx[:, 1]]
            y_sel = threshold_labels(y_sel, config)
            prediction_sel = prediction[idx[:, 0], idx[:, 1]]

            if weight_splits[i] is not None:
                per_sample = loss_fn(prediction_sel, y_sel)
                w = torch.tensor(weight_splits[i], dtype=torch.float32, device=device)
                loss_value = (per_sample * w).mean()
            else:
                loss_value = loss_fn(prediction_sel, y_sel).mean()

            loss_value.backward()
            optimizer.step()
            cur_loss = float(loss_value.item())
            losses.append(cur_loss)
            batch_losses.append(cur_loss)
            y_pred.extend(prediction_sel.detach().cpu().numpy().flatten().tolist())
            y_true.extend(y_sel.detach().cpu().numpy().flatten().tolist())

        batch_idx += 1
        global_step += 1
        elapsed = time.time() - t0
        batch_elapsed = time.time() - batch_t0

        if batch_idx % log_interval == 0:
            avg_loss = np.mean(batch_losses)
            print(
                f"  batch {batch_idx} | loss {avg_loss:.6f} "
                f"| batch {batch_elapsed:.1f}s | total {elapsed:.1f}s",
                flush=True,
            )
            wb_log({"train/batch_loss": avg_loss, "batch_step": global_step})

    return np.mean(losses) if losses else np.nan, y_pred, y_true, global_step


def train_protein_batches(dataset, model, optimizer, loss_fn, config, device,
                          positive_weight=1.0, negative_weight=1.0,
                          global_step=0, log_interval=10):
    losses = []
    y_pred, y_true = [], []
    batch_idx = 0
    t0 = time.time()

    for batch in dataset:
        batch_t0 = time.time()
        X, S, y, _meta, M = batch

        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        S_t = torch.tensor(S, dtype=torch.long, device=device)
        M_t = torch.tensor(M, dtype=torch.float32, device=device)
        y_t = torch.tensor(y, dtype=torch.float32, device=device)

        optimizer.zero_grad()
        prediction = model(X_t, S_t, M_t, train=True, res_level=True)
        indices, weights = select_indices(y, config, positive_weight, negative_weight)
        if not indices:
            batch_idx += 1
            continue

        idx = torch.tensor(indices, dtype=torch.long, device=device)
        y_sel = y_t[idx[:, 0], idx[:, 1]]
        y_sel = threshold_labels(y_sel, config)
        prediction_sel = prediction[idx[:, 0], idx[:, 1]]

        if weights is not None:
            per_sample = loss_fn(prediction_sel, y_sel)
            w = torch.tensor(weights, dtype=torch.float32, device=device)
            loss_value = (per_sample * w).mean()
        else:
            loss_value = loss_fn(prediction_sel, y_sel).mean()

        loss_value.backward()
        optimizer.step()
        cur_loss = float(loss_value.item())
        losses.append(cur_loss)
        y_pred.extend(prediction_sel.detach().cpu().numpy().flatten().tolist())
        y_true.extend(y_sel.detach().cpu().numpy().flatten().tolist())

        batch_idx += 1
        global_step += 1
        elapsed = time.time() - t0
        batch_elapsed = time.time() - batch_t0

        if batch_idx % log_interval == 0:
            print(
                f"  batch {batch_idx} | loss {cur_loss:.6f} "
                f"| batch {batch_elapsed:.1f}s | total {elapsed:.1f}s",
                flush=True,
            )
            wb_log({"train/batch_loss": cur_loss, "batch_step": global_step})

    return np.mean(losses) if losses else np.nan, y_pred, y_true, global_step


def maybe_global_weights(config):
    train_cfg = config["training"]
    if not train_cfg["weight_globally"]:
        return None, None
    neg_cutoff = train_cfg["pos_thresh"] if train_cfg["train_on_intermediates"] else train_cfg["neg_thresh"]
    return determine_global_weights(
        data_dir=config["data"]["data_dir"],
        filestem=config["data"]["filestem"],
        positive_cutoff=train_cfg["pos_thresh"],
        negative_cutoff=neg_cutoff,
        dataset_subdir=config["data"].get("dataset_subdir", "task2"),
    )


def _build_scheduler(optimizer, train_cfg):
    """Create an LR scheduler from train_cfg['scheduler'] (or return None).

    Supported types:
        * plateau  -> torch.optim.lr_scheduler.ReduceLROnPlateau
        * cosine   -> torch.optim.lr_scheduler.CosineAnnealingLR
        * none / missing -> no scheduler
    """
    sched_cfg = train_cfg.get("scheduler", {}) or {}
    sched_type = str(sched_cfg.get("type", "none")).lower()
    if sched_type in ("", "none", "null"):
        return None
    if sched_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=sched_cfg.get("mode", "max"),
            factor=sched_cfg.get("factor", 0.5),
            patience=sched_cfg.get("patience", 5),
            threshold=sched_cfg.get("threshold", 1e-4),
            min_lr=sched_cfg.get("min_lr", 1e-7),
        )
    if sched_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sched_cfg.get("T_max", train_cfg.get("num_epochs", 100)),
            eta_min=sched_cfg.get("eta_min", 0.0),
        )
    raise ValueError(f"Unsupported scheduler type: {sched_type}")


def _scheduler_step(scheduler, monitor_metric, train_cfg):
    """Step the scheduler with the appropriate signature."""
    if scheduler is None:
        return
    sched_cfg = train_cfg.get("scheduler", {}) or {}
    sched_type = str(sched_cfg.get("type", "none")).lower()
    if sched_type == "plateau":
        if monitor_metric is None:
            return
        scheduler.step(monitor_metric)
    else:
        scheduler.step()


def save_training_config(config, output_dir):
    json_path = os.path.join(output_dir, "used_config.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    config = load_config(args.config)

    seed = config["training"].get("seed", 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = config["output"]["output_dir"]
    ensure_dir(output_dir)
    ensure_dir(config["output"]["checkpoint_dir"])
    ensure_dir(os.path.dirname(config["output"]["checkpoint_prefix"]))
    save_training_config(config, output_dir)

    use_wandb = init_wandb(config)
    wb_cfg = config.get("wandb", {})
    log_interval = wb_cfg.get("log_interval", 10)

    num_atoms = get_num_atoms(config)
    print(f"Using {num_atoms}-atom representation "
          f"(ablate_sidechain_vectors={config['model'].get('ablate_sidechain_vectors', True)})")

    filestem = config["data"]["filestem"]
    cache_dir = os.path.join(config["data"]["data_dir"], "cache")
    cache_path = os.path.join(cache_dir, f"{filestem}_{num_atoms}atom.pkl")

    samples = precompute_all_samples(
        data_dir=config["data"]["data_dir"],
        structure_dir=config["data"]["structure_dir"],
        filestem=filestem,
        dataset_subdir=config["data"].get("dataset_subdir", "task2"),
        num_atoms=num_atoms,
        cache_path=cache_path,
    )

    trainset = PrecomputedLoader(
        samples,
        batch_size=config["training"]["batch_size"],
        shuffle=config["training"].get("shuffle", True),
    )
    print(f"Training dataset: {len(samples)} samples, "
          f"batch_size={config['training']['batch_size']}, filestem={filestem}")

    # --- Load validation data ---
    val_cfg = config.get("validation", {})
    val_apo_ids_path = val_cfg.get("val_apo_ids")
    val_label_dict_path = val_cfg.get("val_label_dict")
    apo_structure_dir = val_cfg.get("apo_structure_dir", "")

    has_val = val_apo_ids_path and val_label_dict_path and apo_structure_dir
    if has_val:
        val_apo_ids = np.load(val_apo_ids_path, allow_pickle=True)
        val_label_dict = load_label_dictionary(val_label_dict_path)
        print(f"Loaded validation set: {len(val_apo_ids)} proteins")
    else:
        print("WARNING: validation config not provided – skipping per-epoch validation")

    # --- Load test data (optional, evaluated only at the end) ---
    test_cfg = config.get("test", {})
    test_apo_ids_path = test_cfg.get("test_apo_ids")
    test_label_dict_path = test_cfg.get("test_label_dict")
    test_apo_structure_dir = test_cfg.get("apo_structure_dir", apo_structure_dir)

    has_test = test_apo_ids_path and test_label_dict_path and test_apo_structure_dir
    if has_test:
        test_apo_ids = np.load(test_apo_ids_path, allow_pickle=True)
        test_label_dict = load_label_dictionary(test_label_dict_path)
        print(f"Loaded test set: {len(test_apo_ids)} proteins")

    model = make_model(config)
    model.to(device)

    train_cfg = config["training"]

    # ------------------------------------------------------------------
    # Optional multi-GPU wrapping (DataParallel)
    # ------------------------------------------------------------------
    # Activated by setting ``training.num_gpus > 1`` in the JSON config.
    # The underlying training/validation/test logic is *not* changed:
    # DataParallel only scatters the batch dimension across GPUs and
    # gathers outputs back, leaving the loss / optimizer / scheduler /
    # evaluation code paths identical to single-GPU runs.
    # Note: DataParallel speeds up forward/backward only when the batch
    # being passed in has at least ``num_gpus`` samples (so the batch
    # dimension can be split). For batch_size=1 + residue_batches=true
    # the training loop will benefit only on validation / test forward
    # passes (which batch all val/test proteins at once).
    num_gpus = int(train_cfg.get("num_gpus", 1))
    if num_gpus > 1:
        if not torch.cuda.is_available():
            print(f"WARNING: num_gpus={num_gpus} requested but CUDA is "
                  f"unavailable; falling back to single device.")
        else:
            available = torch.cuda.device_count()
            if num_gpus > available:
                print(f"WARNING: num_gpus={num_gpus} requested but only "
                      f"{available} CUDA devices visible; using {available}.")
                num_gpus = available
            if num_gpus > 1:
                model = nn.DataParallel(
                    model, device_ids=list(range(num_gpus)),
                )
                print(f"Multi-GPU enabled: nn.DataParallel across "
                      f"{num_gpus} GPUs (device_ids={list(range(num_gpus))})")

    lr = train_cfg["learning_rate"]
    weight_decay = train_cfg.get("weight_decay", 0.0)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )
    print(f"Optimizer: AdamW(lr={lr}, weight_decay={weight_decay})")

    scheduler = _build_scheduler(optimizer, train_cfg)
    if scheduler is not None:
        sched_cfg = train_cfg.get("scheduler", {})
        print(f"LR scheduler: {sched_cfg.get('type')} "
              f"(monitor={sched_cfg.get('monitor', 'val_pr_auc')})")

    loss_fn = nn.BCELoss(reduction='none')
    train_func = train_residue_batches if train_cfg["residue_batches"] else train_protein_batches
    positive_weight, negative_weight = maybe_global_weights(config)

    finetune_cfg = config.get("finetune", {})
    pretrained_path = finetune_cfg.get("pretrained_checkpoint")
    if pretrained_path:
        load_checkpoint(model, optimizer, pretrained_path, scheduler=scheduler)
        print(f"Loaded pretrained checkpoint for fine-tuning: {pretrained_path}")
        if finetune_cfg.get("reset_optimizer", True):
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay,
            )
            scheduler = _build_scheduler(optimizer, train_cfg)
            print(f"Optimizer reset with lr={lr}, weight_decay={weight_decay}")

    best_epoch, best_val_loss, best_pr_auc = 0, float("inf"), 0.0
    global_step = 0
    start_epoch = 0

    resume_cfg = config.get("resume", {})
    resume_path = resume_cfg.get("checkpoint")
    if resume_path:
        ckpt_meta = load_checkpoint(model, optimizer, resume_path, scheduler=scheduler)
        start_epoch = ckpt_meta.get('epoch', -1) + 1
        global_step = ckpt_meta.get('global_step', 0)
        best_pr_auc = ckpt_meta.get('best_pr_auc', 0.0)
        best_epoch = ckpt_meta.get('best_epoch', 0)
        print(f"Resuming training from epoch {start_epoch}, "
              f"global_step={global_step}, best_pr_auc={best_pr_auc:.4f}")
    wandb_watch_registered = False

    run_prefix = config["output"]["run_name"] or datetime.now().strftime("%Y%m%d_%H%M%S")

    for epoch in range(start_epoch, config["training"]["num_epochs"]):
        epoch_t0 = time.time()

        # ---- Training ----
        model.train()
        if positive_weight is not None:
            loss, y_pred, y_true, global_step = train_func(
                trainset, model, optimizer, loss_fn, config, device,
                positive_weight=positive_weight,
                negative_weight=negative_weight,
                global_step=global_step,
                log_interval=log_interval,
            )
        else:
            loss, y_pred, y_true, global_step = train_func(
                trainset, model, optimizer, loss_fn, config, device,
                global_step=global_step,
                log_interval=log_interval,
            )

        if use_wandb and not wandb_watch_registered:
            wandb.watch(model, log="gradients", log_freq=100)
            wandb_watch_registered = True

        epoch_elapsed = time.time() - epoch_t0
        print(f"EPOCH {epoch} training loss: {loss:.6f}  ({epoch_elapsed:.1f}s)")

        wb_log({
            "epoch": epoch,
            "train/epoch_loss": float(loss),
            "train/epoch_time_s": epoch_elapsed,
        })

        epoch_prefix = os.path.join(
            config["output"]["checkpoint_dir"], f"{run_prefix}_epoch_{epoch:03d}"
        )
        save_checkpoint(epoch_prefix, model, optimizer,
                        epoch=epoch, global_step=global_step,
                        best_pr_auc=best_pr_auc, best_epoch=best_epoch,
                        scheduler=scheduler)

        # ---- Validation ----
        if has_val:
            predictions, mask = predict_on_xtals(
                model, val_apo_ids, apo_structure_dir, device,
                num_atoms=num_atoms,
            )

            val_loss, val_auc, val_pr_auc, val_y_pred, val_y_true, \
                protein_aucs, protein_pr_aucs = assess_performance(
                    predictions, mask, val_apo_ids, val_label_dict, test=False
                )

            val_recovery, _ = compute_recovery(
                predictions, val_apo_ids, val_label_dict
            )

            print(f"  EPOCH {epoch} val loss: {val_loss:.6f}  "
                  f"AUC: {val_auc:.4f}  PR-AUC: {val_pr_auc:.4f}  "
                  f"Recovery: {val_recovery:.4f}")

            valid_p_aucs = [a for a in protein_aucs if not np.isnan(a)]
            valid_p_pr_aucs = [a for a in protein_pr_aucs if not np.isnan(a)]

            val_metrics = {
                "epoch": epoch,
                "val/loss": float(val_loss),
                "val/auc": float(val_auc),
                "val/pr_auc": float(val_pr_auc),
                "val/recovery": float(val_recovery),
            }
            if valid_p_aucs:
                val_metrics["val/mean_protein_auc"] = float(np.mean(valid_p_aucs))
            if valid_p_pr_aucs:
                val_metrics["val/mean_protein_pr_auc"] = float(np.mean(valid_p_pr_aucs))

            wb_log(val_metrics)

            improved = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss

            if val_pr_auc > best_pr_auc:
                best_epoch = epoch
                best_pr_auc = val_pr_auc
                improved = " ** new best PR-AUC **"
                wb_summary("best_val_pr_auc", float(val_pr_auc))
                wb_summary("best_epoch", epoch)

            if improved:
                print(f"  {improved}")

            _scheduler_step(scheduler, val_pr_auc, train_cfg)
            if scheduler is not None:
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"  current lr: {current_lr:.3e}")
                wb_log({"train/lr": float(current_lr), "epoch": epoch})
        else:
            _scheduler_step(scheduler, None, train_cfg)

    # ---- Load best model based on validation PR-AUC ----
    if has_val:
        print(f"\nBest PR-AUC ({best_pr_auc:.4f}) at epoch {best_epoch}")
        best_ckpt = os.path.join(
            config["output"]["checkpoint_dir"], f"{run_prefix}_epoch_{best_epoch:03d}"
        )
        load_checkpoint(model, optimizer, best_ckpt)
        save_checkpoint(config["output"]["checkpoint_prefix"], model, optimizer)
        print(f"Best model (epoch {best_epoch}) saved to {config['output']['checkpoint_prefix']}")
    else:
        save_checkpoint(config["output"]["checkpoint_prefix"], model, optimizer)
        print(f"Final checkpoint saved to {config['output']['checkpoint_prefix']}")

    # ---- Test evaluation with best model ----
    if has_test:
        print("\n--- Test Evaluation ---")
        test_predictions, test_mask = predict_on_xtals(
            model, test_apo_ids, test_apo_structure_dir, device,
            num_atoms=num_atoms,
        )

        (test_loss, tp, fp, tn, fn, acc, prec, recall,
         test_auc, test_pr_auc, test_y_pred, test_y_true,
         test_protein_aucs, test_protein_pr_aucs) = assess_performance(
            test_predictions, test_mask, test_apo_ids, test_label_dict, test=True
        )

        test_recovery, _ = compute_recovery(
            test_predictions, test_apo_ids, test_label_dict
        )

        test_mean_protein_auc = float(np.nanmean(test_protein_aucs))
        test_mean_protein_pr_auc = float(np.nanmean(test_protein_pr_aucs))

        print(f"Test loss: {test_loss:.6f}  Accuracy: {acc:.4f}")
        print(f"Test AUC: {test_auc:.4f}  Test PR-AUC: {test_pr_auc:.4f}")
        print(f"Test mean_protein_auc: {test_mean_protein_auc:.4f}  "
              f"Test mean_protein_pr_auc: {test_mean_protein_pr_auc:.4f}")
        print(f"Test Recovery: {test_recovery:.4f}")
        print(f"TP: {tp:.0f}  FP: {fp:.0f}  TN: {tn:.0f}  FN: {fn:.0f}")
        print(f"Precision: {prec:.4f}  Recall: {recall:.4f}")

        wb_log({
            "test/loss": float(test_loss),
            "test/auc": float(test_auc),
            "test/pr_auc": float(test_pr_auc),
            "test/mean_protein_auc": test_mean_protein_auc,
            "test/mean_protein_pr_auc": test_mean_protein_pr_auc,
            "test/recovery": float(test_recovery),
            "test/accuracy": float(acc),
            "test/precision": float(prec),
            "test/recall": float(recall),
            "test/tp": tp, "test/fp": fp, "test/tn": tn, "test/fn": fn,
        })
        wb_summary("test_auc", float(test_auc))
        wb_summary("test_pr_auc", float(test_pr_auc))
        wb_summary("test_recovery", float(test_recovery))

    if wandb is not None and wandb.run is not None:
        wandb.finish()
        print("wandb run finished.")


if __name__ == "__main__":
    main()
