#!/usr/bin/env python3
"""
Downloads VocalSet (CC BY 4.0, Wilkins et al., ISMIR 2018, https://zenodo.org/records/1442513) one singer
folder at a time, straight out of the 6 GB archive on Zenodo with HTTP range requests (no full download),
leaving out the techniques that make poor voice-model training data (inhaled singing, lip trills, vocal fry,
trills, trillo). Resumable: files already on disk are skipped.

    python3 tools/training/fetch_vocalset.py --out vocalset            # -> vocalset/female1/..., male1/...
    python3 tools/training/fetch_vocalset.py --out vocalset --zip VocalSet1-2.zip   # whole download first
    python3 tools/training/fetch_vocalset.py --out vocalset --manifest vocalset.csv

--manifest also writes the `path,voice` CSV that prepare_dataset.py takes, so all 20 singers are prepared
in one run (this is exactly how the queue's VocalSet voices were made):

    python3 tools/training/prepare_dataset.py --manifest vocalset.csv --out datasets_vocalset

Only the Python standard library is needed.
"""

import argparse
import concurrent.futures
import io
import os
import sys
import time
import urllib.request
import zipfile

URL = "https://zenodo.org/records/1442513/files/VocalSet1-2.zip?download=1"
SKIP = {"inhaled", "lip_trill", "vocal_fry", "trill", "trillo"}
SINGERS = ["female%d" % i for i in range(1, 10)] + ["male%d" % i for i in range(1, 12)]


class HttpRangeFile(io.RawIOBase):
    """A read-only, seekable file over HTTP range requests, with a small block cache."""

    def __init__(self, url, block=1 << 20):
        self.url, self.pos, self.block, self.cache = url, 0, block, {}
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            self.size = int(r.headers["Content-Range"].split("/")[1])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def fetch(self, start, end):
        req = urllib.request.Request(self.url, headers={"Range": "bytes=%d-%d" % (start, end - 1)})
        for attempt in range(6):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                if len(data) == end - start:
                    return data
            except Exception:
                if attempt == 5:
                    raise
            time.sleep(2 ** attempt)
        raise IOError("short read from %s" % self.url)

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        if n == 0:
            return b""
        if n > 4 * self.block:
            data = self.fetch(self.pos, self.pos + n)
        else:
            out = bytearray()
            p, end = self.pos, self.pos + n
            while p < end:
                b = p // self.block
                if b not in self.cache:
                    if len(self.cache) > 64:
                        self.cache.clear()
                    s = b * self.block
                    self.cache[b] = self.fetch(s, min(self.size, s + self.block))
                chunk = self.cache[b][p - b * self.block:]
                take = min(len(chunk), end - p)
                out += chunk[:take]
                p += take
            data = bytes(out)
        self.pos += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def download(url, path):
    """Whole-archive download with resume (fast on a data-centre connection such as Colab's)."""
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    part = path + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    req = urllib.request.Request(url, headers={"Range": "bytes=%d-" % have} if have else {})
    start, last = time.time(), 0.0
    with urllib.request.urlopen(req, timeout=120) as r, open(part, "ab" if have else "wb") as f:
        total = have + int(r.headers.get("Content-Length", 0))
        while True:
            chunk = r.read(1 << 22)
            if not chunk:
                break
            f.write(chunk)
            have += len(chunk)
            if time.time() - last > 30:
                last = time.time()
                print("  %.2f of %.2f GB (%.0f MB/s)" % (have / 1e9, total / 1e9, have / 1e6 / max(1e-3, last - start)),
                      flush=True)
    os.replace(part, path)


def open_archive(src):
    """`src`: the archive's URL (read with range requests) or a downloaded copy."""
    f = io.BufferedReader(HttpRangeFile(src), buffer_size=1 << 20) if "://" in src else open(src, "rb")
    z = zipfile.ZipFile(f)
    # The archive was made on macOS without ZIP64 records: offsets are stored modulo 2^32 and Python adds
    # one 2^32 to all of them. Undo that and count the wraps in central-directory order instead.
    infos = z.infolist()
    if infos and infos[-1].header_offset >= (1 << 32):
        wraps, previous = 0, -1
        for info in infos:
            raw = info.header_offset - (1 << 32)
            if raw < previous:
                wraps += 1
            previous = raw
            info.header_offset = raw + wraps * (1 << 32)
    return z


def wanted(info):
    parts = info.filename.split("/")
    return (info.filename.startswith("data_by_singer/") and info.filename.endswith(".wav")
            and len(parts) > 3 and parts[3] not in SKIP and not os.path.basename(info.filename).startswith("._"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="folder for the singer folders (female1 ... male11)")
    p.add_argument("--manifest", help="also write this path,voice CSV for prepare_dataset.py")
    p.add_argument("--workers", type=int, default=4, help="parallel downloads (each opens its own connection)")
    p.add_argument("--singers", nargs="*", default=SINGERS, help="only these singer folders (default: all 20)")
    p.add_argument("--url", default=URL)
    p.add_argument("--zip", help="download the whole archive to this file first (resumable; faster on a fast "
                                 "connection) instead of reading it remotely")
    args = p.parse_args()
    src = args.url
    if args.zip:
        print("VocalSet: downloading the archive to %s" % args.zip, flush=True)
        download(args.url, args.zip)
        src = args.zip

    todo = [i for i in open_archive(src).infolist() if wanted(i) and i.filename.split("/")[1] in args.singers]
    print("VocalSet: %d recordings (%.1f GB) without %s" % (len(todo), sum(i.file_size for i in todo) / 1e9,
                                                           ", ".join(sorted(SKIP))), flush=True)

    def work(part):
        z = open_archive(src)
        by_name = {i.filename: i for i in z.infolist()}
        done = 0
        for info in part:
            info = by_name[info.filename]
            out = os.path.join(args.out, info.filename[len("data_by_singer/"):])
            if os.path.exists(out) and os.path.getsize(out) == info.file_size:
                continue
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with z.open(info) as src, open(out + ".part", "wb") as dst:
                dst.write(src.read())
            os.replace(out + ".part", out)
            done += 1
        return done

    workers = max(1, args.workers)
    parts = [todo[k::workers] for k in range(workers)]
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(workers) as ex:
        new = sum(ex.map(work, parts))
    print("VocalSet: %d new files in %.0f s -> %s" % (new, time.time() - start, args.out), flush=True)

    if args.manifest:
        with open(args.manifest, "w", encoding="utf-8") as f:
            f.write("path,voice\n")
            for s in args.singers:
                if os.path.isdir(os.path.join(args.out, s)):
                    f.write("%s,vocalset_%s\n" % (os.path.abspath(os.path.join(args.out, s)), s))
        print("manifest:", args.manifest, flush=True)


if __name__ == "__main__":
    sys.exit(main())
