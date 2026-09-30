#!/usr/bin/env python3
from pathlib import Path
import time
import argparse
import pandas as pd

INPUT_FILE = Path("model_entity_metadata_mapping.csv")
OUTPUT_FILE = Path("global_high_quality_targets.csv")
CHUNK_SIZE = 1_000_000

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    args = parser.parse_args()
    input_file, output_file = args.input, args.output
    if not input_file.exists():
        raise FileNotFoundError(f"未找到输入文件: {input_file.resolve()}")

    # 如果旧结果存在，先删除，避免和新结果混在一起
    if output_file.exists():
        output_file.unlink()

    start_time = time.time()
    total_rows = 0
    total_kept = 0

    print(f"开始处理文件: {input_file.resolve()}")
    print(f"分块大小: {CHUNK_SIZE:,} 行/块")

    # 分块读取，避免一次性加载 4.3GB 文件导致内存压力
    reader = pd.read_csv(
        input_file,
        usecols=REQUIRED_COLUMNS,
        chunksize=CHUNK_SIZE,
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
            output_file,
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
    print(f"输出文件: {output_file.resolve()}")
    print(f"最终获得的数据总数: {total_kept:,} 行")
    print(f"总读取行数: {total_rows:,} 行")
    print(f"总耗时: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
