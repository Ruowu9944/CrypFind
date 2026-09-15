# Dataset specification

## Released characterization

| Stage | Count |
| --- | ---: |
| Initial candidates (external metadata input) | 2,086,838 |
| High-confidence homodimer candidates | 1,531,764 |
| UniRef50 representatives | 454,787 |
| Final paired graphs | 365,221 |

The released summaries report 96,428,054 directed edges, 48,214,027 unique undirected apo contacts, and 1,027,471 contacts with `Δd > 3 Å`. Corrected global Cα RMSD median is 2.145 Å; interface RMSD after global fit is 1.612 Å; median mapping coverage is 1.0000. The Kabsch implementation was independently checked against Biopython SVD for 10 preflight and 30 distributed samples.

## Required external inputs

* AlphaFold-Multimer entity metadata with `modelEntityId`, `uniprotAccession`, `taxId`, `chunk`, `ipTM`, `pDockQ`, and `N_clash_heavyAtom`.
* UniProt-to-UniRef50 mapping TSV(s).
* AlphaFold DB apo monomer structures and the selected multimer/homodimer structures.
* Generated `.pt` paired graphs for whole-corpus QC.

The original raw metadata (4.3 GB), full target lists, structures, and graph tensors are deliberately excluded. The summaries in `metadata/` let readers inspect reported QC without downloading the corpus.

## Graph schema

Each PyG `Data` object stores `apo_pos`, `holo_pos`, `apo_backbone`, `holo_backbone`, backbone masks, `edge_index`, `edge_label`, amino-acid `sequence`, and `num_nodes`. QC expects these fields and validates their dimensions. Graph files are not shipped here.

## Citation/provenance note

All files listed in `MANIFEST.tsv` were copied from the CrypFind data workspace and then minimally publicized only inside this staging release. Paths in the original workspace are not part of this release.
