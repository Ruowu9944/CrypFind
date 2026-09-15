# CrypFind pretraining dataset construction and QC

This release contains the small code, examples, summaries, and figure assets needed to document CrypFind's pretraining-data construction and its dataset-level characterization. It does **not** include models, training code, benchmark code, molecular-dynamics or docking files, or the full graph/structure corpus.

## System and installation

Validated source-environment version pins were not recoverable from the scoped project files, so this release does not claim a fully locked environment. Use Python 3.10+ and install the declared dependencies in an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`torch` and `torch-geometric` must be installed in mutually compatible CPU/GPU builds for the target platform; follow their official installation selectors before running full graph construction/QC.

## Construction workflow

1. `dataset_construction/filter_homodimer_candidates.py` streams the AlphaFold-Multimer metadata table and retains homodimer candidates satisfying `ipTM > 0.6`, `pDockQ > 0.5`, and heavy-atom clashes `< 50`.
2. `prepare_uniprot_mapping_batches.py` prepares UniProt accessions for an external UniRef50 mapping request; `select_uniref50_representatives.py` joins that mapping and retains the highest-ipTM member of every cluster.
3. `download_apo_monomers.py` retrieves apo-like AlphaFold DB monomers by UniProt accession; `download_holo_complexes.py` and `holo_download_io.py` retrieve the corresponding multimer structures. Both are resumable and accept file locations as arguments.
4. `build_paired_graphs.py` pairs the structures, selects holo Chain-A residues within 5 Å of Chain B, globally maps apo and holo sequences, validates N/CA/C backbone atoms, and writes paired PyG graphs. Nodes are mapped interface residues; reciprocal apo Cα contacts below 8 Å are edges; an edge is labelled positive when `d_holo - d_apo > 3 Å`.

`examples/` contains 10-row, source-derived format examples only. It is not a training subset.

### Minimal format check

```bash
python - <<'PY'
import csv
with open('examples/paired_targets_example.csv') as fh:
    rows = list(csv.DictReader(fh))
assert len(rows) == 10
assert {'modelEntityId', 'uniprotAccession', 'cluster_id'} <= rows[0].keys()
print('paired-target example: OK')
PY
```

## QC and characterization

`dataset_qc/` provides read-only audits of generated graphs: schema/contact-label reconstruction, directed/undirected edge counts, construction yield and exclusion counts, corrected row-vector Kabsch global/interface RMSD, mapping coverage, and Biopython SVD agreement. The figure script reads the QC tables and regenerates the dataset-characterization layout.

Set `CRYPFIND_DATA_ROOT` to a directory containing `AlphaFold_Data/` and `processed_pairs_backbone/`; set `CRYPFIND_QC_OUT` to a writable output directory. The scripts never modify the supplied source graphs or structures. Example:

```bash
export CRYPFIND_DATA_ROOT=/path/to/crypfind_data
export CRYPFIND_QC_OUT=/path/to/qc_output
python dataset_qc/dataset_stage_audit.py
python dataset_qc/corrected_rmsd_qc.py --mode validate
python dataset_qc/graph_contact_audit.py --start 0 --end 100
```

Use external AlphaFold/AlphaFold-Multimer metadata and structures plus UniRef50/UniProt mapping data. Full inputs and the 365,221 paired graphs are intentionally absent from GitHub; place them in a persistent data repository such as Zenodo (subject to source-data licences).

See [DATASET.md](DATASET.md) for the released statistics and the input/output contract. `MANIFEST.tsv` records provenance and every release-specific modification.

## Release scope and citation

This is the `v1.0.0` dataset-construction/QC source release. A DOI must be added here only after the matching GitHub release has been archived by Zenodo. It is released under the [MIT License](LICENSE).
