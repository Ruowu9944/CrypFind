import os

import torch
import torch.nn as nn


def _unwrap_model(model):
    """Return the underlying module if *model* is wrapped by DataParallel /
    DistributedDataParallel, otherwise return *model* unchanged.

    Used to keep checkpoint format identical to single-GPU runs (no
    ``module.`` prefix), so checkpoints are interchangeable between
    single-GPU and multi-GPU runs.

    NOTE: we deliberately use ``isinstance`` (rather than ``hasattr``) so
    that for any plain (non-wrapped) model this function is a perfect
    no-op and returns the input unchanged -- preserving 1:1 behaviour
    with the original single-GPU code path.
    """
    parallel_types = (nn.DataParallel, nn.parallel.DistributedDataParallel)
    return model.module if isinstance(model, parallel_types) else model


def save_checkpoint(path, model, optimizer, epoch=None, global_step=None,
                    best_pr_auc=None, best_epoch=None, scheduler=None,
                    **extra_metadata):
    save_path = path if path.endswith('.pt') else path + '.pt'
    dir_name = os.path.dirname(save_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    checkpoint = {
        'model_state_dict': _unwrap_model(model).state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if global_step is not None:
        checkpoint['global_step'] = global_step
    if best_pr_auc is not None:
        checkpoint['best_pr_auc'] = best_pr_auc
    if best_epoch is not None:
        checkpoint['best_epoch'] = best_epoch
    checkpoint.update(extra_metadata)
    torch.save(checkpoint, save_path)
    print(f"CHECKPOINT SAVED TO {save_path}")


def _safe_torch_load(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except Exception:
        return torch.load(path, map_location='cpu', weights_only=False)


def _load_matching_state_dict(model, state_dict, source_name='checkpoint'):
    model_obj = _unwrap_model(model)
    target_state = model_obj.state_dict()
    loadable = {}
    skipped = []

    for key, value in state_dict.items():
        if key.startswith('classifier.'):
            skipped.append(key)
            continue
        if key in target_state and target_state[key].shape == value.shape:
            loadable[key] = value
        else:
            skipped.append(key)

    missing, unexpected = model_obj.load_state_dict(loadable, strict=False)
    print(
        f"PARTIAL CHECKPOINT RESTORED FROM {source_name}: "
        f"loaded={len(loadable)}, skipped={len(skipped)}, "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )
    if skipped:
        preview = ', '.join(skipped[:8])
        suffix = ' ...' if len(skipped) > 8 else ''
        print(f"Skipped incompatible keys: {preview}{suffix}")
    return {
        'loaded_keys': sorted(loadable.keys()),
        'skipped_keys': skipped,
        'missing_keys': missing,
        'unexpected_keys': unexpected,
    }


def _extract_pretrained_backbone_state(checkpoint):
    if 'pockmon_backbone' in checkpoint:
        return checkpoint['pockmon_backbone']

    # Pretraining checkpoints saved by training_code_staging/train_pretrain.py
    # also contain a full wrapper state dict. Keep this fallback so older
    # pretraining runs remain usable.
    if 'model' in checkpoint:
        prefix = 'encoder.backbone.'
        extracted = {}
        for key, value in checkpoint['model'].items():
            if key.startswith(prefix):
                extracted[key[len(prefix):]] = value
        if extracted:
            return extracted

    return None


def load_checkpoint(model, optimizer, path, scheduler=None):
    load_path = path if path.endswith('.pt') else path + '.pt'
    checkpoint = _safe_torch_load(load_path)

    pretrained_backbone = _extract_pretrained_backbone_state(checkpoint)
    if pretrained_backbone is not None:
        meta = _load_matching_state_dict(
            model,
            pretrained_backbone,
            source_name=load_path,
        )
        checkpoint['partial_load_metadata'] = meta
        print(f"PRETRAINED POCKMON BACKBONE RESTORED FROM {load_path}")
        return checkpoint

    _unwrap_model(model).load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print(f"CHECKPOINT RESTORED FROM {load_path}")
    return checkpoint
