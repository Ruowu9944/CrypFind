#!/usr/bin/env python3
import argparse
from pathlib import Path
import time
import pandas as pd

# 读取时所需列（包含过滤列）
REQUIRED_COLUMNS = [
    "modelEntityId",
    "uniprotAccession",
    "taxId",
    "chunk",
    "ipTM",
    "pDockQ",
    "N_clash_heavyAtom",
]

# 最终只保留这 6 列
OUTPUT_COLUMNS = [
    "modelEntityId",
    "uniprotAccession",
    "taxId",
    "chunk",
    "ipTM",
    "pDockQ",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter AlphaFold-Multimer homodimer candidates.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    args = parser.parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_csv}")
    if args.output_csv.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_csv}")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    total_rows = 0
    total_kept = 0

    print(f"Processing: {args.input_csv.resolve()}")
    print(f"Chunk size: {args.chunksize:,} rows")

    # 分块读取，避免一次性加载 4.3GB 文件导致内存压力
    reader = pd.read_csv(
        args.input_csv,
        usecols=REQUIRED_COLUMNS,
        chunksize=args.chunksize,
        low_memory=True,
    )

    for chunk_idx, chunk_df in enumerate(reader, start=1):
        chunk_rows = len(chunk_df)
        total_rows += chunk_rows

        # 转换为数值，防止因字符串或脏数据导致比较失败
        iptm_num = pd.to_numeric(chunk_df["ipTM"], errors="coerce")
        pdockq_num = pd.to_numeric(chunk_df["pDockQ"], errors="coerce")
        clash_num = pd.to_numeric(chunk_df["N_clash_heavyAtom"], errors="coerce")

        # 过滤条件：ipTM > 0.6, pDockQ > 0.5, N_clash_heavyAtom < 50
        mask = (
            (iptm_num > 0.6)
            & (pdockq_num > 0.5)
            & (clash_num < 50)
        )

        filtered = chunk_df.loc[mask, OUTPUT_COLUMNS]
        kept_rows = len(filtered)
        total_kept += kept_rows

        # 逐块追加写入结果文件，避免占用大量内存
        filtered.to_csv(
            args.output_csv,
            mode="a",
            header=(chunk_idx == 1),
            index=False,
        )

        elapsed = time.time() - start_time
        print(
            f"[Chunk {chunk_idx}] 本块读取 {chunk_rows:,} 行, "
            f"本块保留 {kept_rows:,} 行, "
            f"累计读取 {total_rows:,} 行, "
            f"累计保留 {total_kept:,} 行, "
            f"耗时 {elapsed:.1f}s"
        )

    total_elapsed = time.time() - start_time
    print("\n处理完成")
    print(f"Output: {args.output_csv.resolve()}")
    print(f"最终获得的数据总数: {total_kept:,} 行")
    print(f"总读取行数: {total_rows:,} 行")
    print(f"总耗时: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
