#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import random
import threading
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

CSV_FILE = "refined_pretrain_final.csv"
OUTPUT_DIR = "apo_monomer_structures"
FAILED_LOG_FILE = "failed_apo_ids.log"
FAILED_DETAIL_FILE = "failed_apo_details.tsv"
NOT_FOUND_LOG_FILE = "apo_not_found_ids.log"
MAX_WORKERS = 12
API_TIMEOUT = 20
DOWNLOAD_TIMEOUT = 90
REQUEST_DELAY = 0.1  # 每个请求前的基础间隔(秒)，配合抖动用于降低 429 概率
API_URL_TEMPLATE = "https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
PREFERRED_URL_FIELDS = ("cifUrl", "pdbUrl", "bcifUrl")

log_lock = threading.Lock()

# 每个 worker 线程复用同一个 Session，避免重复 TCP+TLS 握手。
_thread_local = threading.local()


def get_session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = make_session()
        _thread_local.session = session
    return session


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download AlphaFold DB monomer structures via the prediction API."
    )
    parser.add_argument("--csv", default=CSV_FILE, help="Input CSV with uniprotAccession column.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for downloaded structures.")
    parser.add_argument("--ids-file", help="Optional newline-separated UniProt ID list.")
    parser.add_argument("--limit", type=int, help="Optional limit for testing.")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS, help="Download worker count.")
    parser.add_argument(
        "--request-delay",
        type=float,
        default=REQUEST_DELAY,
        help="Base per-request delay in seconds (with jitter) to reduce 429 throttling. 0 disables.",
    )
    return parser.parse_args()


def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def build_existing_ids(output_dir: str) -> set[str]:
    """启动时扫描输出目录一次，构建「已下载的 uniprot_id 集合」。

    用于断点续传：之后每次只需 O(1) 集合查找，替代每个任务的 glob 扫描。
    文件名形如 AF-<uniprot_id>-F1-model_v4.cif，取第 2 段作为 id。
    """
    existing: set[str] = set()
    out_dir = Path(output_dir)
    if not out_dir.exists():
        return existing
    for entry in os.scandir(out_dir):
        if not entry.is_file():
            continue
        name = entry.name
        if name.endswith(".part") or not name.startswith("AF-"):
            continue
        parts = name.split("-")
        if len(parts) >= 2 and parts[1]:
            existing.add(parts[1])
    return existing


def append_line(path: str, line: str) -> None:
    with log_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def fetch_prediction_metadata(session: requests.Session, uniprot_id: str) -> list[dict]:
    url = API_URL_TEMPLATE.format(uniprot_id=uniprot_id)
    resp = session.get(url, timeout=API_TIMEOUT)
    if resp.status_code == 404:
        return []
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    return []


def choose_prediction(predictions: list[dict]) -> dict | None:
    usable = [
        item
        for item in predictions
        if isinstance(item, dict) and any(item.get(field) for field in PREFERRED_URL_FIELDS)
    ]
    if not usable:
        return None

    def version_key(item: dict) -> int:
        try:
            return int(item.get("latestVersion") or 0)
        except (TypeError, ValueError):
            return 0

    return max(usable, key=version_key)


def choose_download_url(prediction: dict) -> str | None:
    for field in PREFERRED_URL_FIELDS:
        url = prediction.get(field)
        if url:
            return str(url)
    return None


def output_path_from_url(url: str, uniprot_id: str, output_dir: str) -> str:
    name = os.path.basename(urlparse(url).path)
    if not name:
        name = f"AF-{uniprot_id}-F1-model.cif"
    return os.path.join(output_dir, name)


def download_one(uniprot_id: str, output_dir: str, existing_ids: set[str], request_delay: float = 0.0):
    """
    下载单个 uniprot 的 Apo 结构。
    返回: (status, uniprot_id, message)
    status: downloaded / skipped / not_found / failed
    """
    # 断点续传：启动时已扫描到的文件直接跳过(O(1) 集合查找)
    if uniprot_id in existing_ids:
        return "skipped", uniprot_id, "file exists"

    # 主动限速 + 抖动，降低对 EBI API 的瞬时压力(429)
    if request_delay > 0:
        time.sleep(request_delay * random.uniform(0.5, 1.5))

    session = get_session()
    try:
        predictions = fetch_prediction_metadata(session, uniprot_id)
        if not predictions:
            append_line(NOT_FOUND_LOG_FILE, uniprot_id)
            return "not_found", uniprot_id, "not found in AlphaFold DB API"

        prediction = choose_prediction(predictions)
        if prediction is None:
            return "failed", uniprot_id, "API returned no downloadable structure URL"

        url = choose_download_url(prediction)
        if url is None:
            return "failed", uniprot_id, "API returned no downloadable structure URL"

        out_path = output_path_from_url(url, uniprot_id, output_dir)
        part_path = out_path + ".part"
        if os.path.exists(out_path):
            return "skipped", uniprot_id, "file exists"

        if os.path.exists(part_path):
            os.remove(part_path)
        with session.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True) as resp:
            resp.raise_for_status()
            with open(part_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 128):
                    if chunk:
                        f.write(chunk)
        os.replace(part_path, out_path)
        return "downloaded", uniprot_id, os.path.basename(out_path)
    except Exception as e:
        if "part_path" in locals() and os.path.exists(part_path):
            os.remove(part_path)
        return "failed", uniprot_id, str(e)
    # 注意：session 由线程局部持有并复用，整个线程池存活期间不关闭。


def main():
    args = parse_args()

    if args.ids_file:
        print(f"Reading UniProt IDs: {args.ids_file}")
        with open(args.ids_file, encoding="utf-8") as f:
            unique_ids = list(dict.fromkeys(line.strip() for line in f if line.strip()))
    else:
        print(f"Reading CSV: {args.csv}")
        df = pd.read_csv(args.csv, usecols=["uniprotAccession"])
        unique_ids = (
            df["uniprotAccession"]
            .dropna()
            .astype(str)
            .str.strip()
            .loc[lambda s: s != ""]
            .drop_duplicates()
            .tolist()
        )

    if args.limit is not None:
        unique_ids = unique_ids[: args.limit]

    print(f"Unique uniprotAccession count: {len(unique_ids)}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 断点续传：启动时扫描一次现有目录，构建已下载集合
    existing_ids = build_existing_ids(args.output_dir)
    print(f"Already downloaded (resume): {len(existing_ids)}")

    downloaded = 0
    skipped = 0
    not_found = 0
    failed = 0
    failed_ids = []

    # 重置本轮详细日志；输出目录中的已下载结构仍会被跳过。
    Path(FAILED_DETAIL_FILE).write_text("uniprotAccession\tstatus\tmessage\n", encoding="utf-8")
    Path(NOT_FOUND_LOG_FILE).write_text("", encoding="utf-8")

    print(f"Start downloading to: {args.output_dir}")
    print(f"Threads: {args.max_workers}")
    print(f"Per-request delay: {args.request_delay}s")
    print("Download URL source: AlphaFold DB API")

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(download_one, uid, args.output_dir, existing_ids, args.request_delay)
            for uid in unique_ids
        ]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading Apo"):
            status, uid, msg = fut.result()

            if status == "downloaded":
                downloaded += 1
            elif status == "skipped":
                skipped += 1
            elif status == "not_found":
                not_found += 1
            else:
                failed += 1
                failed_ids.append(uid)
                print(f"[FAILED] {uid} -> {msg}")
                safe_msg = str(msg).replace("\n", " ").replace("\r", " ")
                append_line(FAILED_DETAIL_FILE, f"{uid}\t{status}\t{safe_msg}")

    print("\nDone.")
    print(f"Total targets: {len(unique_ids)}")
    print(f"Downloaded : {downloaded}")
    print(f"Skipped    : {skipped}")
    print(f"Not found  : {not_found}")
    print(f"Failed     : {failed}")

    if failed_ids:
        with open(FAILED_LOG_FILE, "w", encoding="utf-8") as f:
            for uid in failed_ids:
                f.write(f"{uid}\n")
        print(f"Failed IDs have been written to: {FAILED_LOG_FILE}")


if __name__ == "__main__":
    main()
