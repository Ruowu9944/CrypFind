#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import subprocess
import tarfile
import argparse
from pathlib import Path

import pandas as pd

CSV_FILE = "refined_homo_pretrain_final.csv"
TEMP_DOWNLOAD_DIR = "temp_chunks"
FINAL_DIR = "holo_complex_structures"
BASE_URL = "https://ftp.ebi.ac.uk/pub/databases/alphafold/collaborations/nvda/homodimers/"

FAILED_CHUNK_LOG = "failed_holo_chunk_downloads.log"
EXTRACTION_ERROR_LOG = "holo_extraction_errors.log"
MISSING_HOLO_LOG = "missing_holo_targets.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download holo homodimer tar files and stream-extract requested targets."
    )
    parser.add_argument("--csv", default=CSV_FILE, help="Input CSV with modelEntityId/local_tar_name.")
    parser.add_argument("--output-dir", default=FINAL_DIR, help="Directory for extracted holo structures.")
    parser.add_argument("--temp-dir", default=TEMP_DOWNLOAD_DIR, help="Directory for temporary tar files.")
    parser.add_argument("--verbose-extract", action="store_true", help="Print one line per extracted file.")
    return parser.parse_args()


def build_existing_ids(output_dir: str) -> set[str]:
    existing: set[str] = set()
    out = Path(output_dir)
    if not out.exists():
        return existing
    for entry in os.scandir(out):
        if not entry.is_file():
            continue
        name = entry.name
        if name.endswith(".cif.zst"):
            existing.add(name[:-8])
        elif name.endswith(".pdb.zst"):
            existing.add(name[:-8])
    return existing


def normalize_tar_name(local_tar_name: str) -> str:
    name = local_tar_name.strip()
    if not name.endswith(".tar"):
        name += ".tar"
    return name


def download_tar_file(tar_filename: str, tar_path: str) -> bool:
    file_url = BASE_URL + tar_filename
    cmd = [
        "curl",
        "-fL",
        "--connect-timeout",
        "30",
        "--retry",
        "5",
        "--retry-delay",
        "3",
        "-C",
        "-",
        "-o",
        tar_path,
        file_url,
    ]

    print(f"  - 下载: {file_url}")
    result = subprocess.run(cmd)
    ok = result.returncode == 0 and os.path.exists(tar_path) and os.path.getsize(tar_path) > 0
    if not ok:
        print(f"  - 下载失败: {tar_filename}")
    return ok


def extract_targets_from_tar(
    tar_path: str,
    target_ids: set[str],
    output_dir: str,
    verbose_extract: bool = False,
) -> tuple[int, set[str], set[str]]:
    """
    tarfile 流式读取，不解压整包。
    只提取文件名精确匹配 {modelEntityId}.cif.zst 或 {modelEntityId}.pdb.zst 的成员。
    """
    extracted_count = 0
    pending_ids = set(target_ids)
    extracted_ids: set[str] = set()

    # 预构建精确匹配名字，加速查找
    valid_names = set()
    for tid in pending_ids:
        valid_names.add(f"{tid}.cif.zst")
        valid_names.add(f"{tid}.pdb.zst")

    with tarfile.open(tar_path, "r") as tar:
        for member in tar:
            if not pending_ids:
                break
            if not member.isfile():
                continue

            base_name = os.path.basename(member.name)
            if base_name not in valid_names:
                continue

            if base_name.endswith(".cif.zst"):
                tid = base_name[:-8]
                ext = ".cif.zst"
            elif base_name.endswith(".pdb.zst"):
                tid = base_name[:-8]
                ext = ".pdb.zst"
            else:
                continue

            if tid not in pending_ids:
                continue

            out_path = os.path.join(output_dir, f"{tid}{ext}")
            if os.path.exists(out_path):
                pending_ids.discard(tid)
                continue

            src = tar.extractfile(member)
            if src is None:
                continue

            with src:
                with open(out_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)

            extracted_count += 1
            extracted_ids.add(tid)
            pending_ids.discard(tid)
            if verbose_extract:
                print(f"    * 提取成功: {tid}{ext}")

    return extracted_count, pending_ids, extracted_ids


def main() -> None:
    args = parse_args()
    Path(args.temp_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    existing_ids = build_existing_ids(args.output_dir)
    print(f"已存在 holo 结构数量（断点续跑）：{len(existing_ids):,}")

    print(f"读取清单: {args.csv}")
    df = pd.read_csv(args.csv, low_memory=False)
    if "local_tar_name" not in df.columns and "chunk" in df.columns:
        df = df.rename(columns={"chunk": "local_tar_name"})
    df = df[["modelEntityId", "local_tar_name"]]
    df = df.dropna(subset=["modelEntityId", "local_tar_name"]).copy()
    df["modelEntityId"] = df["modelEntityId"].astype(str).str.strip()
    df["local_tar_name"] = df["local_tar_name"].astype(str).str.strip()
    df = df[(df["modelEntityId"] != "") & (df["local_tar_name"] != "")]

    if df.empty:
        raise RuntimeError("输入清单无有效行（modelEntityId/local_tar_name）。")

    df["local_tar_name"] = df["local_tar_name"].map(normalize_tar_name)

    # 每个 tar 包内去重并保序
    tar_target_map = (
        df.groupby("local_tar_name")["modelEntityId"]
        .apply(lambda s: list(dict.fromkeys(s.tolist())))
        .to_dict()
    )

    total_tar = len(tar_target_map)
    total_targets = sum(len(v) for v in tar_target_map.values())

    print(f"需要处理 tar 包数量: {total_tar:,}")
    print(f"目标复合体总数量: {total_targets:,}")
    print(f"下载基地址: {BASE_URL}")

    failed_chunks: list[tuple[str, str]] = []
    extraction_errors: list[tuple[str, str]] = []
    missing_records: list[tuple[str, str]] = []

    processed_tar = 0
    skipped_tar = 0
    extracted_total = 0

    for tar_name, target_ids in tar_target_map.items():
        processed_tar += 1
        print(f"\n[{processed_tar}/{total_tar}] 处理压缩包: {tar_name}")

        # 每个目标处理前都检查是否存在；已存在则跳过
        pending_ids = {tid for tid in target_ids if tid not in existing_ids}
        already_count = len(target_ids) - len(pending_ids)
        if already_count > 0:
            print(f"  - 已存在目标: {already_count:,}，待提取目标: {len(pending_ids):,}")

        # 如果 tar 对应目标都已存在，直接跳过下载
        if not pending_ids:
            skipped_tar += 1
            print("  - 该 tar 所有目标都已存在，跳过下载。")
            continue

        temp_tar_path = os.path.join(args.temp_dir, tar_name)
        ok = download_tar_file(tar_name, temp_tar_path)
        if not ok:
            failed_chunks.append((tar_name, temp_tar_path))
            if os.path.exists(temp_tar_path):
                os.remove(temp_tar_path)
            continue

        try:
            print(f"  - 开始流式提取，目标数: {len(pending_ids):,}")
            extracted_count, missing_ids, extracted_ids = extract_targets_from_tar(
                temp_tar_path,
                pending_ids,
                args.output_dir,
                args.verbose_extract,
            )
            extracted_total += extracted_count
            existing_ids.update(extracted_ids)

            if missing_ids:
                print(f"  - 警告: 本包仍缺失 {len(missing_ids):,} 个目标")
                for tid in sorted(missing_ids):
                    missing_records.append((tar_name, tid))
            else:
                print("  - 本包目标提取完整。")

            print(
                f"  - 本包提取完成: extracted={extracted_count:,}, "
                f"missing={len(missing_ids):,}"
            )
        except Exception as exc:
            extraction_errors.append((tar_name, str(exc)))
            print(f"  - 提取异常: {exc}")
        finally:
            # 边下边删：当前 tar 完成后立即删除
            if os.path.exists(temp_tar_path):
                os.remove(temp_tar_path)
                print("  - 临时 tar 已删除。")

    if failed_chunks:
        with open(FAILED_CHUNK_LOG, "w", encoding="utf-8") as f:
            f.write("tar_name,temp_tar_path\n")
            for tar_name, temp_path in failed_chunks:
                f.write(f"{tar_name},{temp_path}\n")
        print(f"\n下载失败记录已写入: {FAILED_CHUNK_LOG}")

    if extraction_errors:
        with open(EXTRACTION_ERROR_LOG, "w", encoding="utf-8") as f:
            f.write("tar_name,error\n")
            for tar_name, err in extraction_errors:
                safe_err = err.replace("\n", " ").replace("\r", " ").replace(",", ";")
                f.write(f"{tar_name},{safe_err}\n")
        print(f"提取异常记录已写入: {EXTRACTION_ERROR_LOG}")

    if missing_records:
        with open(MISSING_HOLO_LOG, "w", encoding="utf-8") as f:
            f.write("local_tar_name,modelEntityId\n")
            for tar_name, tid in missing_records:
                f.write(f"{tar_name},{tid}\n")
        print(f"缺失目标记录已写入: {MISSING_HOLO_LOG}")

    print("\n========== Holo 批量提取完成 ==========")
    print(f"处理 tar 数: {processed_tar:,}")
    print(f"跳过 tar 数（全已存在）: {skipped_tar:,}")
    print(f"本次新提取结构数: {extracted_total:,}")
    print(f"下载失败 tar 数: {len(failed_chunks):,}")
    print(f"提取异常 tar 数: {len(extraction_errors):,}")
    print(f"仍缺失目标数: {len(missing_records):,}")
    print(f"输出目录: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
