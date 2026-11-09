#!/usr/bin/env python3
"""
Downloads singers of GTSinger (Zhang et al., NeurIPS 2024, https://huggingface.co/datasets/GTSinger/GTSinger)
for training voice models. GTSinger is licensed CC BY-NC-SA 4.0: voices trained on it are for personal,
non-commercial use only, must credit the dataset, and must never be sold, shared under other terms or put
in a public repository.

Each singer sang ~30 pop songs in a normal "Control" take and again with a technique (breathy, vibrato,
falsetto, mixed voice, glissando, pharyngeal), plus spoken versions. The normal takes make the best training
data; --groups adds technique takes. Takes are chosen song by song until --minutes of audio per singer
(studio WAV, 48 kHz / 24 bit), so one training run fits in a free GPU session.

    python3 tools/training/fetch_gtsinger.py --out gtsinger --manifest gtsinger.csv
    python3 tools/training/fetch_gtsinger.py --out gtsinger --singers EN-Tenor-1 --minutes 90
    python3 tools/training/prepare_dataset.py --manifest gtsinger.csv --out datasets_gtsinger

--manifest writes the `path,voice` CSV prepare_dataset.py takes; voices are named gtsinger_en_alto_1 etc.
Resumable: files already on disk are skipped. Only the Python standard library is needed.
"""

import argparse
import concurrent.futures
import json
import os
import re
import sys
import urllib.parse
import urllib.request

REPO = "GTSinger/GTSinger"
API = "https://huggingface.co/api/datasets/%s/tree/main/%s?recursive=true"
FILE = "https://huggingface.co/datasets/%s/resolve/main/%s"
ENGLISH = ["EN-Alto-1", "EN-Alto-2", "EN-Tenor-1"]
BYTES_PER_SECOND = 48000 * 3     # mono 48 kHz, 24 bit


def voice_name(singer):
    return "gtsinger_" + singer.lower().replace("-", "_")


def queue_voices(data):
    """Writes <data>/voices_gtsinger.json, the training queue's entries for the singers prepared (by
    prepare_dataset.py) in <data>/datasets_gtsinger/. Returns their names."""
    report = json.load(open(os.path.join(data, "datasets_gtsinger", "report.json"), encoding="utf-8"))
    voices = []
    for name, r in sorted(report["voices"].items()):
        singer = name[len("gtsinger_"):].replace("_", "-").upper().replace("ALTO", "Alto").replace("TENOR", "Tenor") \
                     .replace("SOPRANO", "Soprano").replace("BASS", "Bass").replace("BARITONE", "Baritone")
        kind = singer.split("-")[1]
        gender = "male" if kind in ("Tenor", "Bass", "Baritone") else "female"
        voices.append({
            "name": name, "dataset": "datasets_gtsinger/" + name,
            "display_name": "GTSinger %s %s" % (kind, singer.split("-")[2]), "gender": gender,
            "genre": "Pop", "styles": "Clean", "corpus": "gtsinger", "style": "contemporary", "listed": True,
            "description": "RVC v2 voice trained on %.0f min of GTSinger singer %s" % (r["minutes"], singer),
            "license": "CC BY-NC-SA 4.0 - GTSinger (Zhang et al., NeurIPS 2024): personal, non-commercial use only",
            "tests": [["tests/light_male_25s.wav", 0 if gender == "male" else 12],
                      ["tests/notmine_female_25s.wav", -12 if gender == "male" else 0]]})
    with open(os.path.join(data, "voices_gtsinger.json"), "w", encoding="utf-8") as f:
        json.dump({"voices": voices}, f, indent=1)
    return [v["name"] for v in voices]


def list_files(folder):
    """Every file under `folder` of the dataset: (path, size), following the API's pages."""
    files, url = [], API % (REPO, urllib.parse.quote(folder))
    while url:
        with urllib.request.urlopen(url, timeout=120) as r:
            files += [(x["path"], x.get("lfs", {}).get("size", x["size"])) for x in json.load(r) if x["type"] == "file"]
            link = r.headers.get("Link", "")
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = m.group(1) if m else None
    return files


def choose(files, groups, minutes):
    """Whole songs, normal takes first, until `minutes` of audio: [(path, size)]."""
    by_song = {}
    for path, size in files:
        parts = path.split("/")   # <language>/<singer>/<technique>/<song>/<group>/<n>.wav
        if not path.endswith(".wav") or len(parts) < 6 or parts[4] not in groups:
            continue
        by_song.setdefault((groups.index(parts[4]), parts[3], parts[2]), []).append((path, size))
    chosen, budget = [], minutes * 60 * BYTES_PER_SECOND
    for key in sorted(by_song):
        takes = sorted(by_song[key])
        size = sum(s for _, s in takes)
        if chosen and size > budget:
            continue
        chosen += takes
        budget -= size
    return chosen


def download(path, size, out):
    dst = os.path.join(out, *path.split("/")[1:])
    if os.path.exists(dst) and os.path.getsize(dst) == size:
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(FILE % (REPO, urllib.parse.quote(path)), timeout=300) as r, open(dst + ".part", "wb") as f:
                while True:
                    block = r.read(1 << 20)
                    if not block:
                        break
                    f.write(block)
            os.replace(dst + ".part", dst)
            return dst
        except OSError as e:
            if attempt == 3:
                raise RuntimeError("%s: %s" % (path, e))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="download folder (<out>/<singer>/...)")
    p.add_argument("--singers", nargs="+", default=ENGLISH, help="GTSinger singer folders, e.g. EN-Alto-1")
    p.add_argument("--groups", nargs="+", default=["Control_Group", "Vibrato_Group", "Mixed_Voice_Group"],
                   help="takes to use, in order of preference (never Paired_Speech_Group: that is speech)")
    p.add_argument("--minutes", type=float, default=70, help="audio per singer (default 70)")
    p.add_argument("--manifest", help="write a path,voice CSV for prepare_dataset.py")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    rows = []   # one folder per singer: song names may hold characters a glob would misread
    for singer in args.singers:
        language = {"EN": "English", "ZH": "Chinese", "JA": "Japanese", "KO": "Korean", "RU": "Russian",
                    "ES": "Spanish", "FR": "French", "DE": "German", "IT": "Italian"}[singer.split("-")[0]]
        takes = choose(list_files(language + "/" + singer), args.groups, args.minutes)
        print("%s: %d takes, %.0f min" % (singer, len(takes), sum(s for _, s in takes) / BYTES_PER_SECOND / 60), flush=True)
        with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
            for n, _ in enumerate(pool.map(lambda t: download(t[0], t[1], args.out), takes), 1):
                if n % 50 == 0 or n == len(takes):
                    print("  %d / %d" % (n, len(takes)), flush=True)
        rows.append((os.path.abspath(os.path.join(args.out, singer)), voice_name(singer)))

    if args.manifest:
        with open(args.manifest, "w", encoding="utf-8") as f:
            f.write("path,voice\n")
            for path, voice in rows:
                f.write('"%s",%s\n' % (path.replace('"', '""'), voice))
    print("done: %d singers -> %s" % (len(rows), args.out))


if __name__ == "__main__":
    sys.exit(main())
