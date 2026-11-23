#!/usr/bin/env python3
"""
Download N-CMAPSS .h5 files from NASA's public PHM data repository.

The official archive (~15.8 GB) is a zip that contains another zip
(``data_set.zip``) holding ``data_set/N-CMAPSS_*.h5``. This module streams the
archive once over HTTP and extracts only the requested .h5 files on the fly:
the 15.8 GB zip is never written to disk, files are CRC-checked, written
atomically (``*.part`` -> final name), already-present files are kept, and the
stream stops as soon as every requested file is done.

CLI:
  python ml_pipeline/datasets/ncmapss_download.py --dest N-CMAPSS/data_set
  python ml_pipeline/datasets/ncmapss_download.py --dest /content/N-CMAPSS --files all
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import struct
import sys
import time
import urllib.request
import zipfile
import zlib
from pathlib import Path

NCMAPSS_URL = "https://phm-datasets.s3.amazonaws.com/NASA/17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip"

# Files used by the Continuous RUL Benchmark (DS08 variants are excluded; DS08d is known to be truncated)
BENCHMARK_FILES = [
    "N-CMAPSS_DS01-005.h5",
    "N-CMAPSS_DS02-006.h5",
    "N-CMAPSS_DS03-012.h5",
    "N-CMAPSS_DS05.h5",
    "N-CMAPSS_DS07.h5",
]

CHUNK = 16 << 20  # 16 MB per HTTP range request / copy step


class RangeReader(io.RawIOBase):
    """Seekable read-only file over HTTP Range requests (or a local file path), with retries."""

    def __init__(self, url: str, retries: int = 8):
        self.url, self.pos, self.retries = url, 0, retries
        self.local = None if url.startswith(("http://", "https://")) else open(url, "rb")
        if self.local is not None:
            self.size = os.fstat(self.local.fileno()).st_size
        else:
            req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "crb-downloader"})
            with urllib.request.urlopen(req, timeout=60) as r:
                self.size = int(r.headers["Content-Length"])
        self.bytes_read = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def _fetch(self, start: int, end: int) -> bytes:
        if self.local is not None:
            self.local.seek(start)
            return self.local.read(end - start + 1)
        last = None
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(
                    self.url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "crb-downloader"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                if len(data) == end - start + 1:
                    return data
                last = IOError(f"short read {len(data)} != {end - start + 1}")
            except Exception as exc:  # network hiccup -> retry with backoff
                last = exc
            time.sleep(min(60, 2 ** attempt))
        raise IOError(f"download failed after {self.retries} retries: {last}")

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        data = self._fetch(self.pos, self.pos + n - 1)
        b[: len(data)] = data
        self.pos += len(data)
        self.bytes_read += len(data)
        return len(data)


def _read_exact(stream, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        part = stream.read(n - len(buf))
        if not part:
            raise EOFError(f"unexpected end of stream ({len(buf)}/{n} bytes)")
        buf += part
    return bytes(buf)


def _zip64_sizes(extra: bytes, usize: int, csize: int):
    i = 0
    while i + 4 <= len(extra):
        hid, hl = struct.unpack("<HH", extra[i : i + 4])
        if hid == 0x0001:
            vals = list(struct.unpack("<" + "Q" * (hl // 8), extra[i + 4 : i + 4 + (hl // 8) * 8]))
            if usize == 0xFFFFFFFF and vals:
                usize = vals.pop(0)
            if csize == 0xFFFFFFFF and vals:
                csize = vals.pop(0)
        i += 4 + hl
    return usize, csize


def _fmt_gb(n: float) -> str:
    return f"{n / 1e9:.2f} GB"


def ensure_ncmapss(dest, files=None, url: str = NCMAPSS_URL, log=print) -> list[Path]:
    """Make sure ``files`` exist in ``dest`` (download/extract missing ones). Returns their paths."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    want_all = files in (None, "all", ["all"])
    wanted = set() if want_all else set(files or BENCHMARK_FILES)
    have = {p.name for p in dest.glob("N-CMAPSS_*.h5")}
    missing = None if want_all else wanted - have
    if missing is not None and not missing:
        log(f"[N-CMAPSS] all {len(wanted)} file(s) already present in {dest}")
        return [dest / f for f in sorted(wanted)]

    log(f"[N-CMAPSS] streaming {url}")
    log(f"[N-CMAPSS] extracting {'all files' if want_all else sorted(missing)} -> {dest}")
    src = RangeReader(url)
    outer = zipfile.ZipFile(io.BufferedReader(src, buffer_size=CHUNK))
    members = [i for i in outer.infolist() if i.filename.endswith("data_set.zip")]
    if not members:
        raise RuntimeError(f"data_set.zip not found in archive: {[i.filename for i in outer.infolist()]}")
    inner = outer.open(members[0])  # sequential, decompressed on the fly

    t0, done = time.time(), []
    while True:
        sig = inner.read(4)
        if len(sig) < 4 or sig != b"PK\x03\x04":
            break  # central directory (or end) reached
        hdr = _read_exact(inner, 26)
        _, flag, method, _, _, crc, csize, usize, nlen, elen = struct.unpack("<HHHHHIIIHH", hdr)
        name = _read_exact(inner, nlen).decode("utf-8", "replace")
        extra = _read_exact(inner, elen)
        usize, csize = _zip64_sizes(extra, usize, csize)
        if flag & 0x08 and csize == 0:
            raise RuntimeError(f"{name}: sizes stored after data (streaming not possible)")
        base = name.rsplit("/", 1)[-1]
        take = base.endswith(".h5") and (want_all or base in missing) and base not in have

        if not take:
            # skip compressed bytes (must still be read from the stream)
            left = csize
            while left:
                left -= len(_read_exact(inner, min(CHUNK, left)))
            if base.endswith(".h5"):
                log(f"[N-CMAPSS] skip {base} ({'present' if base in have else 'not requested'})")
            if missing is not None and missing <= set(done) | have:
                break
            continue

        free = shutil.disk_usage(dest).free
        if free < usize + (1 << 30):
            raise OSError(f"not enough disk space in {dest} for {base}: need {_fmt_gb(usize)}, free {_fmt_gb(free)}")
        tmp = dest / (base + ".part")
        log(f"[N-CMAPSS] extracting {base} ({_fmt_gb(usize)}) ...")
        dec = zlib.decompressobj(-15) if method == 8 else None
        if method not in (0, 8):
            raise RuntimeError(f"{name}: unsupported compression method {method}")
        crc_calc, written, left, last_print = 0, 0, csize, time.time()
        with open(tmp, "wb") as fh:
            while left:
                chunk = _read_exact(inner, min(CHUNK, left))
                left -= len(chunk)
                out = dec.decompress(chunk) if dec else chunk
                fh.write(out)
                crc_calc = zlib.crc32(out, crc_calc)
                written += len(out)
                if time.time() - last_print > 30:
                    rate = src.bytes_read / max(time.time() - t0, 1e-9) / 1e6
                    log(f"    {base}: {100 * written / max(usize, 1):5.1f}%  (download {rate:.0f} MB/s)")
                    last_print = time.time()
            if dec:
                out = dec.flush()
                fh.write(out)
                crc_calc = zlib.crc32(out, crc_calc)
                written += len(out)
        if written != usize or (crc_calc & 0xFFFFFFFF) != crc:
            tmp.unlink(missing_ok=True)
            raise IOError(f"{base}: integrity check failed (size {written}/{usize}, crc {crc_calc:#x}/{crc:#x})")
        os.replace(tmp, dest / base)
        done.append(base)
        log(f"[N-CMAPSS] OK {base} ({_fmt_gb(written)}, CRC verified) [{(time.time() - t0) / 60:.1f} min]")
        if missing is not None and missing <= set(done) | have:
            break  # everything requested is done -> stop streaming early

    if missing is not None:
        still = missing - set(done)
        if still:
            raise FileNotFoundError(f"not found in archive: {sorted(still)}")
        return [dest / f for f in sorted(wanted)]
    return sorted(dest.glob("N-CMAPSS_*.h5"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default=str(Path(__file__).resolve().parents[2] / "N-CMAPSS" / "data_set"))
    ap.add_argument("--files", nargs="+", default=BENCHMARK_FILES, help="file names, or 'all'")
    ap.add_argument("--url", default=NCMAPSS_URL)
    a = ap.parse_args()
    files = "all" if a.files == ["all"] else a.files
    paths = ensure_ncmapss(a.dest, files, a.url)
    print("\n".join(str(p) for p in paths))


if __name__ == "__main__":
    sys.exit(main())
