[README.md](https://github.com/user-attachments/files/32232720/README.md)
# CrypFind

**Geometric contrastive pretraining of residue-contact rearrangements for cryptic pocket prediction**

CrypFind is an SE(3)-equivariant geometric deep-learning framework for prioritizing cryptic pockets and structurally dynamic binding sites from protein structures. It learns transferable signatures of local conformational plasticity from large-scale structural proxy pairs, then transfers these representations to residue-level cryptic-pocket and ligand-binding-site prediction.

The central premise is deliberately modest: an interface-induced conformation in a protein homodimer is **not** treated as a literal small-molecule holo structure. Instead, pairing an AlphaFold monomer with its corresponding AlphaFold-Multimer homodimer provides scalable supervision for selected geometric events relevant to conformational accessibility—especially local residue-contact expansion and rearrangement. CrypFind uses this proxy task to learn structure-aware representations that can be fine-tuned for cryptic-pocket prediction.

This repository accompanies:

> Wu, Y., Chen, D., and Tian, L. *CrypFind: Geometric Contrastive Pretraining of Residue Contact Rearrangements for Cryptic Pocket Prediction.*

## Highlights

- A UniRef50-reduced collection of **365,221** AlphaFold monomer–homodimer structural-proxy pairs.
- An SE(3)-equivariant residue graph encoder trained with residue-level contrastive alignment and contact-expansion supervision.
- Pooled edge-level pretraining performance of **PR-AUC 0.9489** for contact expansion.
- Fine-tuned cryptic-pocket prediction on PocketMiner (**PR-AUC 0.8493**), an improvement of 18.54 percentage points over the reproduced baseline reported in the manuscript.
- Similarity-weighted evaluation on AF2BIND (**ROC-AUC 0.9295**).
- Proteome-scale prioritization of **48,396** candidate sites across AlphaFold-predicted human proteins.
- Multi-replica molecular-dynamics case studies for UbiC and p67phox, interpreted as computational structural evidence rather than experimental validation of ligand binding.

## Overview

Cryptic pockets are transient or conditionally accessible protein cavities that may be poorly represented in one static apo structure. Their discovery is challenging because experimentally resolved apo–holo pairs are limited, whereas molecular dynamics (MD) sampling is expensive at the scale needed for representation learning.

CrypFind addresses this limitation with a structural-proxy pretraining strategy. AlphaFold monomer predictions serve as apo-like structures, while corresponding high-confidence AlphaFold-Multimer homodimer predictions provide interface-induced conformations. After sequence and residue correspondence are established, each state is represented as a residue graph. The model learns (i) aligned residue representations across paired states and (ii) which apo-state residue contacts expand in the interface-induced state. The pretrained geometric encoder is subsequently fine-tuned for cryptic-pocket and binding-site prediction.

The framework is intended for **prioritization**. A high CrypFind score indicates a computationally predicted dynamic or binding-relevant region; it does not establish ligand binding, affinity, selectivity, or therapeutic tractability without experimental follow-up.

## Method at a glance

```text
AlphaFold monomers + AlphaFold-Multimer homodimers
                    |
          confidence and geometry filtering
                    |
           UniRef50 representative selection
                    |
       apo-like / interface-induced structural pairing
                    |
   residue mapping, atom checks, and paired graph construction
                    |
     SE(3)-equivariant geometric contrastive pretraining
       ├── residue-level apo ↔ induced-state alignment
       └── apo-contact expansion prediction
                    |
    fine-tuning for cryptic-pocket / binding-site prediction
                    |
    benchmark evaluation and proteome-scale prioritization
```

### Structural-proxy dataset construction

The dataset-building workflow begins with AlphaFold and AlphaFold-Multimer monomer–homodimer records. Initial homodimer candidates are filtered using model-confidence and geometry criteria: interface predicted TM-score (ipTM) greater than 0.6, pDockQ greater than 0.5, and fewer than 50 heavy-atom steric clashes. UniRef50 representative selection reduces sequence redundancy before paired-structure assembly.

For each retained target, the workflow downloads or resolves the apo-like monomer and the interface-induced homodimer structure, identifies corresponding chains and residues, validates required backbone atoms, and verifies structural availability. A paired sample is retained only when residue correspondence and structure checks support meaningful comparison. The resulting graphs use residues as nodes and spatially defined residue contacts as edges. An apo contact is labelled as expanded when its paired inter-residue distance increases by more than 3 Å in the interface-induced structure.

The construction funnel reported in the manuscript is:

| Stage | Retained records |
| --- | ---: |
| Initial homodimer candidates | 2,086,838 |
| High-confidence candidates after ipTM/pDockQ/clash filtering | 1,531,764 |
| UniRef50 representatives | 454,787 |
| Final paired graphs | 365,221 |

The dataset contains 96,428,054 directed supervised edges (48,214,027 unique undirected contacts), including 2,054,942 directed positive contact-expansion labels (1,027,471 unique undirected positives; 2.131%). Full graph collections and source structure archives are intentionally handled as external data assets rather than ordinary Git-tracked files.

### Quality control and dataset characterization

The repository includes construction audits and characterization utilities for tracing the attrition funnel, graph statistics, failed or excluded samples, residue-mapping coverage, and structural-pair quality. Pair alignment uses Kabsch/SVD-based Cα RMSD calculations, with independent validation using Biopython alignment routines.

For the final paired dataset, the manuscript reports a median residue-mapping coverage of 1.0000, median global Cα RMSD of 2.145 Å (IQR 0.807–7.888 Å), and median interface RMSD of 1.612 Å (IQR 0.703–5.247 Å). These statistics characterize the structural proxy pairs; they should not be interpreted as evidence that all pairs mimic a ligand-bound state.

## Geometric contrastive pretraining

CrypFind operates on residue-level geometric graphs. The encoder is SE(3)-equivariant: it uses scalar and steerable geometric features so that rotations and translations of an input structure do not change the physical interpretation of a prediction. The implementation uses PyTorch, PyTorch Geometric, e3nn, and radius-graph message passing; the model backbone includes geometric tensor attention and equivariant feature-fusion components.

Pretraining combines two complementary objectives:

1. **Residue-level contrastive alignment.** Corresponding mapped residues from the apo-like and interface-induced graphs are brought together in representation space through an InfoNCE-style contrastive objective. This encourages the encoder to preserve residue identity across conformational change while learning geometry-aware context.
2. **Contact-expansion prediction.** A binary edge objective predicts whether an apo-state residue contact undergoes sufficient distance expansion in the paired interface-induced state. This directs the model toward local rearrangements that may be relevant to pocket accessibility.

On the held-out structural-proxy validation set, the manuscript reports pooled edge PR-AUC 0.9489, F1 0.9092, and ROC-AUC 0.9952 for the contact-expansion task. Per-protein PR-AUC is also reported to avoid allowing proteins with many graph edges to dominate the pooled result.

## Downstream evaluation

### PocketMiner cryptic-pocket benchmark

CrypFind is fine-tuned for residue-level cryptic-pocket prediction on PocketMiner. The manuscript reports PR-AUC 0.8493, ROC-AUC 0.8949, precision 0.8162, recall 0.6785, and pocket recovery 0.4077. The reported reproduced baseline obtains PR-AUC 0.6639 under the study’s evaluation protocol.

### AF2BIND binding-site benchmark

On AF2BIND, CrypFind is evaluated for residue-level small-molecule binding-site prediction with sequence-similarity-weighted metrics. The primary reported result is a similarity-weighted ROC-AUC of 0.9295 and a weighted recovery of 0.6172. These measurements assess prediction under the defined benchmark protocol; they do not by themselves demonstrate prospective target engagement.

### Ablation studies

The manuscript compares pretrained and scratch-initialized variants, as well as auxiliary-objective variants. The results support the contribution of geometric pretraining to transfer performance. Scripts and configuration files preserve the experiment settings used for the reported comparisons, while raw benchmark sources remain governed by their respective licenses and access terms.

## Human proteome application

CrypFind was applied to AlphaFold-predicted human proteins to prioritize candidate dynamic binding sites. The analysis identified 48,396 candidate sites. These predictions are computational hypotheses intended to guide inspection and downstream experimental work. Druggability comparisons with pocket-detection tools are used for structural characterization and prioritization, not as a substitute for biochemical or cellular validation.

## MD and docking case studies

The UbiC and p67phox analyses test whether selected CrypFind-prioritized regions are compatible with dynamic accessibility in explicit structural simulations.

For UbiC, multi-replica apo MD is used to evaluate a ligand-relevant pocket against reference structural information. For p67phox, independent apo simulations support a dynamically accessible, potentially ligandable region. Docking of AZA1 and short ligand-bound simulations provide computational pose-stability and contact observations. These results should be read conservatively: they do **not** constitute experimental confirmation of binding, affinity, selectivity, or efficacy.

## Repository layout

```text
CrypFind/
├── CrypFind_dataset_GitHub/     # Dataset construction, QC, examples, metadata, figures
│   ├── dataset_construction/
│   ├── dataset_qc/
│   ├── examples/
│   ├── metadata/
│   ├── figures/
│   ├── DATASET.md
│   └── requirements.txt
├── pockmon/                     # Equivariant model and geometric layers
├── training/                    # Pretraining and downstream training programs
├── configs/                     # Experiment configurations
├── evaluation/                  # Benchmark evaluation and ablation analysis
├── inference/                   # Structure-level prediction utilities
├── data_preparation/            # Feature and pair-representation preparation
├── proteome_scan/               # Proteome-scale inference and site clustering
├── analysis/                    # Figure and characterization scripts
└── README.md
```

`CrypFind_dataset_GitHub/` is self-documented through its `DATASET.md`, `MANIFEST.tsv`, example tables, and dataset-specific README. It records the provenance and release modifications of each lightweight construction/QC asset.

## Installation

### Core environment

CrypFind requires Python 3.9 or later, PyTorch, PyTorch Geometric, e3nn, NumPy, SciPy, scikit-learn, Biopython, and standard scientific plotting/data packages. GPU-enabled PyTorch installation must match the local CUDA driver and platform.

```bash
git clone https://github.com/Ruowu9944/CrypFind.git
cd CrypFind

conda create -n crypfind python=3.9
conda activate crypfind

pip install torch torchvision
pip install torch-geometric torch-scatter torch-sparse torch-cluster
pip install e3nn numpy scipy scikit-learn biopython pandas matplotlib tqdm
```

The dataset-specific dependencies are listed in `CrypFind_dataset_GitHub/requirements.txt`. Optional AlphaFold pair-representation preparation uses a separate JAX/ColabDesign-compatible environment. Install the relevant CUDA/JAX build for the local system rather than copying environment settings from another server.

## Inputs, outputs, and minimal workflow

### Dataset construction and QC

The lightweight dataset release provides scripts for candidate filtering, UniRef50 representative selection, apo-like monomer and homodimer acquisition, residue mapping, paired graph construction, stage auditing, RMSD QC, contact auditing, and dataset-characterization figures.

```bash
cd CrypFind_dataset_GitHub
pip install -r requirements.txt

python dataset_construction/filter_homodimer_candidates.py --help
python dataset_construction/select_uniref50_representatives.py --help
python dataset_construction/build_paired_graphs.py --help
python dataset_qc/dataset_stage_audit.py --help
python dataset_qc/finalize_rmsd_qc.py --help
python dataset_qc/make_dataset_characterization_figure.py --help
```

Example inputs are supplied under `examples/`; small summary outputs and quality-control statistics are under `metadata/`. The scripts accept paths to user-provided external structure and sequence resources. They do not bundle the full AlphaFold structure collection, UniRef databases, or the 365,221 paired graph files.

### Model workflows

Model programs use residue-level structures and derived graph features. The pretraining program accepts a paired-graph directory through `--data-dir`, while configuration files specify model and optimization settings. Fine-tuning programs operate on benchmark-specific structures/features and output checkpoints, per-residue scores, and evaluation tables. Inference utilities take an input structure and produce residue-level probability scores; structure writers can encode scores for visualization.

Run each program’s `--help` interface and its corresponding configuration before launching a full job. Full pretraining, benchmark reconstruction, proteome scans, MD, and docking require substantial external data and compute resources; they are not appropriate as minimal smoke tests.

## Reproducibility notes

- Use the released configurations and fixed random seeds for reported benchmark reconstructions.
- Preserve the official benchmark split definitions and external-data licenses.
- Report both pooled and per-protein metrics where relevant, especially for highly imbalanced graph-edge labels.
- Treat AlphaFold monomer–homodimer pairs as structural proxies, not experimentally determined apo–holo pairs.
- Record GPU, CUDA, PyTorch, PyTorch Geometric, e3nn, and JAX versions in any reproduction report.

## Data availability

This repository provides source code, processing scripts, lightweight dataset metadata, examples, configuration files, and figure-generation utilities. Large primary resources—including AlphaFold/AlphaFold-Multimer structures, UniRef50 resources, full paired graphs, intermediate tensors, benchmark data subject to third-party terms, and simulation trajectories—are not suitable for ordinary Git storage. Obtain them from their original providers and place them at paths supplied to the relevant scripts.

## Citation

If you use CrypFind, please cite the associated manuscript:

```text
Wu Y, Chen D, Tian L. CrypFind: Geometric Contrastive Pretraining of
Residue Contact Rearrangements for Cryptic Pocket Prediction.
```

Please also cite the original sources of AlphaFold/AlphaFold-Multimer, UniRef, PocketMiner, AF2BIND, and any other third-party resources used in a reproduction.

## License

This repository is distributed under the included license. Third-party datasets, structural resources, and software dependencies remain subject to their own licenses and terms of use.

## Contact

For questions about CrypFind, please open a GitHub issue or contact the corresponding authors listed in the manuscript.
