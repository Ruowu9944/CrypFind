#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
I/O helpers for auto_download_holo_direct.py:
  * download_one  -- fetch a single AFDB file, optionally zstd-compress, write
  * write_logs    -- dump failed / not-found lists

Kept in a separate module purely to keep each source file small.
"""

import gzip
import os
import subprocess
import threading
import time

import requests

GZIP_MAGIC = b"\x1f\x8b"

# One pooled Session per worker thread. Reusing connections lets keep-alive
# amortise the ~0.5s TLS handshake that the proxy adds to every new
# connection -- measured ~+40% throughput vs a fresh connection per file.
_thread_local = threading.local()


def _session():
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=4, pool_maxsize=4, max_retries=0)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        s.headers["User-Agent"] = USER_AGENT
        _thread_local.session = s
    return s

USER_AGENT = "afdb-holo-direct/1.0 (+research bulk fetch)"

FAILED_LOG = "failed_holo_direct.log"
NOTFOUND_LOG = "missing_holo_direct.log"

EXT_SUFFIX = {"cif": "-model_v1.cif", "pdb": "-model_v1.pdb"}


def _remote_name(id_str, ext):
    return f"{id_str}{EXT_SUFFIX[ext]}"


def _local_name(id_str, ext, compress):
    base = _remote_name(id_str, ext)
    return base + ".zst" if compress else base


class NotFound(Exception):
    """Server has no such file (HTTP 404 / GCS NoSuchKey)."""


def _fetch(url, timeout):
    """GET url -> plaintext bytes via a pooled keep-alive session.

    Raises NotFound for missing files (GCS returns 404 + an <?xml NoSuchKey
    body). Some AFDB members are stored gzip-compressed on the server
    (magic 1f8b); we transparently gunzip them so the caller always sees
    plaintext cif, keeping every output byte-identical to the
    plaintext-then-zstd files already on disk.
    """
    r = _session().get(url, timeout=timeout)
    if r.status_code == 404:
        raise NotFound()
    r.raise_for_status()
    data = r.content
    if data[:2] == GZIP_MAGIC:
        data = gzip.decompress(data)
    return data


def _zstd_compress(raw: bytes, zstd_path: str, level: int) -> bytes:
    """Pipe raw bytes through `zstd -<level> -c` and return compressed bytes.

    Done in-memory (no temp files); stdin/stdout pipes only.
    """
    proc = subprocess.run(
        [zstd_path, f"-{level}", "-c", "-q"],
        input=raw,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"zstd failed: {proc.stderr.decode('utf-8', 'replace')[:200]}")
    return proc.stdout


def download_one(id_str, ext, args, compress, stats):
    """Returns (status, detail) where status in {ok, skipped, notfound, failed}.

    Atomic write: data goes to <final>.part then os.replace -> <final>, so an
    interrupted run never leaves a half-written file that resume would trust.
    """
    fname = _local_name(id_str, ext, compress)
    final_path = os.path.join(args.output_dir, fname)

    # race-safe re-check (another worker / prior run may have written it)
    if os.path.exists(final_path):
        with stats.lock:
            stats.skipped += 1
        return ("skipped", None)

    url = args.files_base + _remote_name(id_str, ext)
    last_err = None

    for attempt in range(1, args.retries + 1):
        try:
            raw = _fetch(url, args.timeout)
            if not raw:
                raise RuntimeError("empty body")

            # sanity: cif should be text starting with 'data_'
            if ext == "cif" and not raw.lstrip()[:5].startswith(b"data_"):
                raise RuntimeError("unexpected cif content (no 'data_' header)")

            payload = _zstd_compress(raw, args.zstd, args.zstd_level) if compress else raw

            tmp = final_path + ".part"
            with open(tmp, "wb") as f:
                f.write(payload)
            os.replace(tmp, final_path)

            with stats.lock:
                stats.done += 1
                stats.bytes_in += len(raw)
            return ("ok", None)

        except NotFound:
            with stats.lock:
                stats.notfound += 1
            return ("notfound", "404")
        except Exception as e:  # noqa: BLE001
            last_err = str(e)

        # transient: backoff then retry (skip sleep on the final attempt)
        if attempt < args.retries:
            time.sleep(min(2.0 * attempt, 15.0))

    # clean any stray .part
    try:
        os.remove(final_path + ".part")
    except OSError:
        pass
    with stats.lock:
        stats.failed += 1
    return ("failed", last_err or "unknown")


def write_logs(failed_items, notfound_items):
    if failed_items:
        with open(FAILED_LOG, "w", encoding="utf-8") as f:
            f.write("modelEntityId,ext,reason\n")
            for id_str, ext, reason in failed_items:
                safe = str(reason).replace("\n", " ").replace(",", ";")
                f.write(f"{id_str},{ext},{safe}\n")
    if notfound_items:
        with open(NOTFOUND_LOG, "w", encoding="utf-8") as f:
            f.write("modelEntityId,ext\n")
            for id_str, ext in notfound_items:
                f.write(f"{id_str},{ext}\n")
