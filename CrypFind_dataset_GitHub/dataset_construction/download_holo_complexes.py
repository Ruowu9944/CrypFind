#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direct per-file downloader for the AFDB homodimer / complex set.

Why this exists
---------------
The earlier `auto_download_holo_range.py` assumed the dataset was *only*
published as ~8 GB `chunk_XXXX.tar` files on the throttled EBI FTP node
(ftp.ebi.ac.uk), and therefore went to great lengths to HTTP-Range into each
tar and carve out individual members. Measured throughput on that route was
~14 KB/s per connection, capped per-IP, with many connections refused
(curl exit 56) -- i.e. parallelism did not help, and the project would have
taken ~24 more days.

It turns out AFDB also serves every structure as an *individual* file via
    https://alphafold.ebi.ac.uk/files/AF-<id>-model_v1.cif
which is backed by Google Cloud Storage (response carries x-goog-* headers)
and is NOT subject to the EBI FTP throttle. Measured ~170 KB/s per
connection and it scales ~linearly with concurrency.

So this script simply downloads each wanted .cif directly and (to stay
byte-compatible with the 29k files already fetched as .cif.zst) pipes the
bytes through `zstd` and writes `<id>-model_v1.cif.zst`. Files that already
exist are skipped, so it is a drop-in resume of the previous run.

Proxy is taken from the http_proxy/https_proxy environment variables
(urllib honours them automatically).
"""

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

CSV_FILE = "refined_homo_pretrain_final.csv"
FINAL_DIR = "holo_complex_structures"
FILES_BASE = "https://alphafold.ebi.ac.uk/files/"   # GCS-backed, not throttled
DEFAULT_ZSTD = "zstd"

FAILED_LOG = "failed_holo_direct.log"
NOTFOUND_LOG = "missing_holo_direct.log"

ID_COL = "modelEntityId"
USER_AGENT = "afdb-holo-direct/1.0 (+research bulk fetch)"


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv", default=CSV_FILE)
    p.add_argument("--output-dir", default=FINAL_DIR)
    p.add_argument("--files-base", default=FILES_BASE)
    p.add_argument("--exts", default="cif",
                   help="Comma list of wanted extensions: cif,pdb.")
    p.add_argument("--workers", type=int, default=128,
                   help="Concurrent downloads. This job is latency-bound (each "
                        "request spends ~0.5s in proxy TLS + ~1s to first byte), "
                        "so throughput scales with concurrency well past the CPU "
                        "count. Measured knee ~128 (~11 files/s, ~4.4 MB/s "
                        "aggregate via the proxy); 192 gives no more.")
    p.add_argument("--zstd", default=DEFAULT_ZSTD,
                   help="Path to zstd CLI used to recompress to .cif.zst.")
    p.add_argument("--zstd-level", type=int, default=15,
                   help="zstd level. Benchmarked on these cif: L15 is <1ms/file "
                        "at 4.6x ratio; L19 is ~100ms/file (100x slower) for only "
                        "~1.2%% smaller output -> L19 makes compression the "
                        "throughput bottleneck. Level only affects size, not the "
                        "decompressed bytes, so downstream .cif.zst reads are "
                        "identical regardless.")
    p.add_argument("--no-compress", action="store_true",
                   help="Store plain .cif instead of .cif.zst.")
    p.add_argument("--timeout", type=float, default=120.0,
                   help="Per-request socket timeout (seconds).")
    p.add_argument("--retries", type=int, default=5,
                   help="Retries per file on transient errors.")
    p.add_argument("--limit", type=int, default=0,
                   help="Process only the first N targets (for testing).")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
EXT_SUFFIX = {"cif": "-model_v1.cif", "pdb": "-model_v1.pdb"}


def remote_name(id_str: str, ext: str) -> str:
    return f"{id_str}{EXT_SUFFIX[ext]}"


def local_name(id_str: str, ext: str, compress: bool) -> str:
    base = remote_name(id_str, ext)
    return base + ".zst" if compress else base


class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.done = 0          # files written this run
        self.skipped = 0       # already present
        self.notfound = 0      # 404 on server
        self.failed = 0        # exhausted retries
        self.bytes_in = 0      # raw cif bytes downloaded
        self.processed = 0     # targets fully handled (for progress)


def main():
    args = parse_args()
    wanted_exts = [e.strip() for e in args.exts.split(",")
                   if e.strip() in ("cif", "pdb")]
    if not wanted_exts:
        raise SystemExit("--exts must include at least one of: cif, pdb")

    compress = not args.no_compress
    if compress and not (os.path.isfile(args.zstd) and os.access(args.zstd, os.X_OK)):
        args.zstd = shutil.which(args.zstd) or args.zstd
    if compress and not (os.path.isfile(args.zstd) and os.access(args.zstd, os.X_OK)):
        raise SystemExit(f"zstd CLI not found/executable: {args.zstd}\n"
                         f"Pass --zstd <path> or use --no-compress.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"读取清单: {args.csv}")
    df = pd.read_csv(args.csv, usecols=[ID_COL], low_memory=False)
    df[ID_COL] = df[ID_COL].astype(str).str.strip()
    df = df[df[ID_COL] != ""]
    ids = list(dict.fromkeys(df[ID_COL].tolist()))   # de-dup, keep order
    if args.limit > 0:
        ids = ids[:args.limit]

    # build the work list: (id_str, ext) for every file not already on disk
    print("扫描已存在文件（断点续跑）...")
    existing = set(os.listdir(out_dir)) if out_dir.exists() else set()
    tasks = []
    pre_skip = 0
    for id_str in ids:
        for ext in wanted_exts:
            if local_name(id_str, ext, compress) in existing:
                pre_skip += 1
            else:
                tasks.append((id_str, ext))

    print(f"目标实体总数: {len(ids):,}")
    print(f"提取文件类型: {wanted_exts}")
    print(f"输出格式: {'.cif.zst (zstd -%d)' % args.zstd_level if compress else 'plain .cif'}")
    print(f"已存在（跳过）: {pre_skip:,}")
    print(f"本次待下载: {len(tasks):,}")
    print(f"并发下载数: {args.workers}")
    print(f"endpoint: {args.files_base}  (GCS-backed, 不限速)")
    print(f"输出目录: {out_dir.resolve()}")
    if not tasks:
        print("没有需要下载的文件，全部已存在。")
        return

    stats = Stats()
    failed_items, notfound_items = [], []
    t0 = time.time()

    def worker(item):
        id_str, ext = item
        return download_one(id_str, ext, args, compress, stats)

    total = len(tasks)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, t): t for t in tasks}
        for fut in as_completed(futures):
            id_str, ext = futures[fut]
            status, detail = fut.result()
            if status == "failed":
                failed_items.append((id_str, ext, detail))
            elif status == "notfound":
                notfound_items.append((id_str, ext))
            with stats.lock:
                stats.processed += 1
                p = stats.processed
                d, sk, nf, fl = stats.done, stats.skipped, stats.notfound, stats.failed
                mb = stats.bytes_in / 1e6
            if args.verbose or p % 200 == 0 or p == total:
                el = time.time() - t0
                rate = p / el if el > 0 else 0
                eta = (total - p) / rate if rate > 0 else 0
                print(f"[{p:,}/{total:,}] ok={d:,} skip={sk:,} 404={nf:,} "
                      f"fail={fl:,} | {mb:,.0f}MB in | {rate:.1f} files/s | "
                      f"ETA {eta/3600:.1f}h", flush=True)

    write_logs(failed_items, notfound_items)
    el = time.time() - t0
    print("\n========== Holo 直连下载完成 ==========")
    print(f"耗时: {el/3600:.2f} h")
    print(f"成功写入: {stats.done:,}")
    print(f"跳过(已存在): {stats.skipped:,}")
    print(f"服务器无此文件(404): {stats.notfound:,}")
    print(f"失败(重试耗尽): {stats.failed:,}")
    print(f"下载原始字节: {stats.bytes_in/1e9:.2f} GB")
    if failed_items:
        print(f"失败清单: {FAILED_LOG}")
    if notfound_items:
        print(f"404 清单: {NOTFOUND_LOG}")


if __name__ == "__main__":
    from _holo_direct_io import download_one, write_logs  # noqa: E402
    main()
