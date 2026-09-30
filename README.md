# CrypFind

**Geometric Contrastive Pretraining of Residue Contact Rearrangements for Cryptic Pocket Prediction**

CrypFind is an SE(3)-equivariant framework that learns from paired protein structures to predict cryptic pocket residues. It combines residue-level contrastive learning with contact-expansion prediction, then fine-tunes the learned encoder on PocketMiner and AF2BIND.

## Installation

```bash
pip install -r requirements.txt
```

Install PyTorch and its geometric extensions for your system and CUDA version. AF2 pair-feature extraction also requires ColabDesign and AlphaFold parameters.

## Data

Datasets and model weights are not included. Download the relevant data from AlphaFold DB, the AlphaFold-Multimer homodimer collection, UniRef50, PocketMiner, and AF2BIND. Place PocketMiner and AF2BIND files under `datasets/pocketminer/` and `datasets/af2bind/`, respectively. Pretrained weights are not bundled in this source repository.

## Usage

Run the commands below from the repository root after preparing the required data and pair features.

### Build pretraining graphs

```bash
python scripts/filter_structures.py --input data/model_entity_metadata_mapping.csv --output data/quality.csv
python scripts/select_uniref50.py --main-csv data/quality.csv --mapping-dir data/uniref50 --out-csv data/selected.csv
python scripts/download_monomers.py --csv data/selected.csv --output-dir data/monomers
python scripts/download_homodimers.py --csv data/selected.csv --output-dir data/homodimers --temp-dir data/tmp
python scripts/build_pretraining_data.py --csv data/selected.csv --apo-dir data/monomers --holo-dir data/homodimers --out-dir data/graphs --log-csv data/graph_skips.csv
```

### Pretrain

```bash
python train_pretrain.py --data-dir data/graphs --save-dir checkpoints/pretrain --epochs 24
```

### Fine-tune and evaluate

```bash
python training/train_pocketminer.py configs/pocketminer.json
python training/train_af2bind.py configs/af2bind.json
```

These commands use the supplied benchmark splits and evaluate the trained models. The configs expect a pretrained checkpoint at `checkpoints/pretrain/best.pt`. To train from scratch, set the relevant `pretrained_checkpoint` value to `null`.

### Human proteome inference

The `proteome_scan/` scripts extract pair features, predict per-residue scores, and cluster candidate sites. They require downloaded AlphaFold proteome structures and a fine-tuned checkpoint.

## Repository contents

- `pockmon/` — geometric encoder, dataset loading, and checkpoint utilities.
- `scripts/` — structural filtering, graph construction, and pair-feature extraction.
- `training/` — PocketMiner and AF2BIND training and evaluation.
- `proteome_scan/` — proteome inference and candidate-site clustering.
- `configs/` — benchmark settings and a representative MD production configuration.

## Citation

Citation information will be updated upon publication.
