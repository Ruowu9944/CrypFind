# CrypFind

Geometric contrastive pretraining of residue contact rearrangements for cryptic pocket prediction.

## Installation

Install Python dependencies with `pip install -r requirements.txt`. Install PyTorch Geometric extensions for your PyTorch/CUDA build. AF2 pair feature extraction additionally uses ColabDesign/AlphaFold parameters; GROMACS is needed only for MD validation.

## Data

Download the public AlphaFold DB monomers, AlphaFold-Multimer homodimer metadata/structures, UniRef50 mapping, PocketMiner, AF2BIND, and relevant PDB structures yourself. Place PocketMiner under `datasets/pocketminer/` and AF2BIND under `datasets/af2bind/`. The source repository contains no structures, labels, generated graphs, weights, or results. Pretrained weights are not bundled in the source repository.

## Pretraining data construction

From the repository root, after obtaining the public homodimer metadata CSV and UniRef50 mapping TSVs:

```bash
python scripts/filter_structures.py --input data/model_entity_metadata_mapping.csv --output data/quality.csv
python scripts/select_uniref50.py --main-csv data/quality.csv --mapping-dir data/uniref50 --out-csv data/selected.csv
python scripts/download_monomers.py --csv data/selected.csv --output-dir data/monomers
python scripts/download_homodimers.py --csv data/selected.csv --output-dir data/homodimers --temp-dir data/tmp
python scripts/build_pretraining_data.py --csv data/selected.csv --apo-dir data/monomers --holo-dir data/homodimers --out-dir data/graphs --log-csv data/graph_skips.csv
```

The filter applies ipTM > 0.6, pDockQ > 0.5, and fewer than 50 heavy-atom clashes. UniRef50 selection retains the highest-ipTM member. Graph construction globally aligns residues, identifies interface residues, forms apo contacts below 8 Å, and labels expansions above 3 Å. The 95:5 training/validation split is made by the training script with seed 42.

## Pretraining

```bash
python train_pretrain.py --data-dir data/graphs --save-dir checkpoints/pretrain --epochs 24 --batch-size 8 --lr 2e-5 --weight-decay 1e-5 --grad-clip 1.0 --temperature 0.1
```

The shared encoder is trained with InfoNCE and contact-expansion BCE. The best validation checkpoint is used for downstream fine-tuning.

## PocketMiner

Extract the combined AF2 pair features with `python scripts/extract_pocketminer_pair.py --output-dir datasets/pocketminer/combined_pair --pdb-dir datasets/pocketminer/training-data --params-dir datasets/af2bind/params`. Then run:

```bash
python training/train_pocketminer.py configs/pocketminer.json
```

The training entry evaluates ROC-AUC, PR-AUC, precision, recall, and recovery on the benchmark splits. Set `finetune.pretrained_checkpoint` to `null` in the config for a scratch control.

## AF2BIND

Extract binder features with `python scripts/extract_af2bind_pair.py --output-dir datasets/af2bind/binder_pair_all --pdb-dir datasets/af2bind/all_pdbs`. Then run:

```bash
python training/train_af2bind.py configs/af2bind.json
```

The source computes inverse TM-score neighbourhood weights (TM-score > 0.5), similarity-weighted ROC-AUC, and weighted recovery. Set `training.pretrained_checkpoint` to `null` for a scratch control. Both fine-tuning commands train and evaluate using their configured splits.

## Inference

For a trained AF2BIND fine-tuned checkpoint and downloaded human AlphaFold proteome PDBs:

```bash
python proteome_scan/extract_binder_pair_proteome.py --pdb-dir datasets/human_proteome --output-dir data/proteome_pair --params-dir datasets/af2bind/params
python proteome_scan/predict_proteome.py --config configs/af2bind.json --checkpoint outputs/af2bind/checkpoints/best_model.pt --pdb-dir datasets/human_proteome --binder-pair-dir data/proteome_pair --output-dir data/proteome_predictions
python proteome_scan/cluster_binding_sites.py --preds-dir data/proteome_predictions --pdb-dir datasets/human_proteome --output-dir data/proteome_sites
```

## MD validation

The reported p67phox and UbiC cases used public PDB entries 1WM5 and 1XLR (with 1FW9 as a UbiC reference), GROMACS, AMBER99SB-ILDN, and independent 100 ns replicas at 300 K. A representative production parameter file is `configs/md_production.mdp`. No trajectory or docking outputs are included.

## Citation

Citation information will be updated upon publication.
