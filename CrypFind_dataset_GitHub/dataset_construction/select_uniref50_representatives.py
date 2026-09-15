#!/usr/bin/env python3
"""
Refine protein structure targets by UniRef50 cluster.

Workflow:
1) Read and concat all mapping TSVs under mapping_files/.
2) Deduplicate mappings and normalize columns to:
      uniprotAccession, cluster_id
3) Inner-join with global_high_quality_targets.csv on uniprotAccession.
4) For each cluster_id, keep only the row with the highest ipTM.
   (ties keep the first occurrence)
5) Save refined output to refined_pretrain_final.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refine main target list by UniRef50 non-redundancy."
    )
    parser.add_argument(
        "--main-csv",
        default="global_high_quality_targets.csv",
        help="Path to main CSV file.",
    )
    parser.add_argument(
        "--mapping-dir",
        default="mapping_files",
        help="Directory containing mapping TSV files.",
    )
    parser.add_argument(
        "--mapping-pattern",
        default="*.tsv",
        help="Glob pattern for mapping files.",
    )
    parser.add_argument(
        "--out-csv",
        default="refined_pretrain_final.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def find_mapping_cols(columns: Iterable[str]) -> tuple[str, str]:
    cols = set(columns)

    from_candidates = ("From", "from", "uniprotAccession")
    to_candidates = ("To", "to", "Cluster ID", "cluster_id")

    from_col = next((c for c in from_candidates if c in cols), None)
    to_col = next((c for c in to_candidates if c in cols), None)

    if from_col is None or to_col is None:
        raise ValueError(
            "Cannot locate mapping columns. Expected one of "
            f"{from_candidates} and one of {to_candidates}, got: {list(columns)}"
        )
    return from_col, to_col


def load_and_merge_mapping(mapping_dir: Path, pattern: str) -> pd.DataFrame:
    files = sorted(mapping_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No mapping files found in {mapping_dir} with pattern '{pattern}'."
        )

    parts: list[pd.DataFrame] = []
    for f in files:
        df = pd.read_csv(f, sep="\t", dtype="string")
        from_col, to_col = find_mapping_cols(df.columns)
        small = df[[from_col, to_col]].rename(
            columns={from_col: "uniprotAccession", to_col: "cluster_id"}
        )
        parts.append(small)

    mapping = pd.concat(parts, axis=0, ignore_index=True)
    mapping = mapping.dropna(subset=["uniprotAccession", "cluster_id"])
    mapping["uniprotAccession"] = mapping["uniprotAccession"].str.strip()
    mapping["cluster_id"] = mapping["cluster_id"].str.strip()
    mapping = mapping[
        (mapping["uniprotAccession"] != "") & (mapping["cluster_id"] != "")
    ]

    # Remove exact duplicate pairs first.
    mapping = mapping.drop_duplicates(ignore_index=True)

    return mapping


def main() -> None:
    args = parse_args()

    main_csv = Path(args.main_csv)
    mapping_dir = Path(args.mapping_dir)
    out_csv = Path(args.out_csv)

    if not main_csv.exists():
        raise FileNotFoundError(f"Main CSV not found: {main_csv}")
    if not mapping_dir.exists():
        raise FileNotFoundError(f"Mapping directory not found: {mapping_dir}")

    # 1) Load + merge mapping TSV files
    mapping = load_and_merge_mapping(mapping_dir, args.mapping_pattern)

    pair_count = len(mapping)
    uniq_accession_count = mapping["uniprotAccession"].nunique(dropna=True)
    uniq_cluster_count = mapping["cluster_id"].nunique(dropna=True)

    print(f"[Mapping] deduplicated pair rows: {pair_count:,}")
    print(f"[Mapping] unique uniprotAccession IDs: {uniq_accession_count:,}")
    print(f"[Mapping] unique cluster_id IDs: {uniq_cluster_count:,}")

    # Avoid one-to-many explosion in merge if same accession maps to multiple clusters.
    dup_acc = mapping["uniprotAccession"].duplicated(keep=False).sum()
    if dup_acc > 0:
        n_conflict = mapping["uniprotAccession"].value_counts().gt(1).sum()
        print(
            "[Warning] one accession -> multiple clusters detected for "
            f"{n_conflict:,} accessions. Keeping first cluster per accession."
        )
    mapping_unique = mapping.drop_duplicates(subset=["uniprotAccession"], keep="first")

    # 2) Load main list (only one large CSV, keep full columns as requested)
    main_df = pd.read_csv(main_csv, low_memory=False)
    if "uniprotAccession" not in main_df.columns:
        raise ValueError("Column 'uniprotAccession' not found in main CSV.")
    if "ipTM" not in main_df.columns:
        raise ValueError("Column 'ipTM' not found in main CSV.")

    before_n = len(main_df)

    # 3) Inner join on uniprotAccession
    merged = main_df.merge(
        mapping_unique,
        on="uniprotAccession",
        how="inner",
        copy=False,
    )

    if merged.empty:
        print("[Result] Inner join produced 0 rows. Nothing to save.")
        merged.to_csv(out_csv, index=False)
        return

    # 4) For each UniRef50 cluster, keep highest ipTM row (ties -> first by index)
    merged["ipTM"] = pd.to_numeric(merged["ipTM"], errors="coerce")
    merged = merged.dropna(subset=["ipTM", "cluster_id"])
    idx = merged.groupby("cluster_id", sort=False)["ipTM"].idxmax()
    refined = merged.loc[idx].copy()

    # Keep stable order in output if needed (original row order by index)
    refined = refined.sort_index()

    after_n = len(refined)

    # 5) Output stats and save
    print(f"[Main] before refine rows: {before_n:,}")
    print(f"[Main] after refine rows:  {after_n:,}")
    print(f"[Main] reduced by:         {before_n - after_n:,}")
    print(f"[Output] saving to: {out_csv}")

    refined.to_csv(out_csv, index=False)
    print("[Done] refined_pretrain_final.csv generated successfully.")


if __name__ == "__main__":
    main()
