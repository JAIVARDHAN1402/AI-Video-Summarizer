"""Fetch and lay out SumMe + TVSum for highlight-model training.

    python scripts/fetch_datasets.py --root data/highlight

Produces:
    <root>/SumMe/GT/*.mat  +  <root>/SumMe/videos/*
    <root>/TVSum/ydata-tvsum50.mat  +  <root>/TVSum/video/*.mp4

SumMe is a single public zip and downloads automatically. TVSum's archive is
behind Google Drive; this script can't reliably pull it, so it prints the
steps and — once you've dropped the annotation files in place — will fetch the
50 videos from the URLs in ``ydata-tvsum50-info.tsv`` with yt-dlp.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

SUMME_ZIP_URL = "https://data.vision.ee.ethz.ch/cvl/SumMe/SumMe.zip"
TVSUM_HELP = """
TVSum (50 videos, 20 annotators):

  1. Download 'ydata-tvsum50-v1_1.zip' from the link in
     https://github.com/yalesong/tvsum  (Google Drive, ~2.7 GB).
  2. Unzip so you have either:
       {root}/TVSum/ydata-tvsum50.mat
     or the TSV set:
       {root}/TVSum/ydata-tvsum50-anno.tsv
       {root}/TVSum/ydata-tvsum50-info.tsv
  3. If the zip had no 'video/' folder, re-run this script with --tvsum-videos
     to fetch the 50 clips from the URLs in ydata-tvsum50-info.tsv via yt-dlp.
"""


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  have {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
        return
    print(f"  downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")

    def hook(block, bsize, total):
        done = block * bsize
        pct = f"{100 * done / total:.0f}%" if total > 0 else f"{done / 1e6:.0f} MB"
        sys.stdout.write(f"\r    {pct}")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    sys.stdout.write("\n")
    tmp.rename(dest)


def _unzip(zip_path: Path, dest: Path) -> None:
    print(f"  unzipping {zip_path.name} -> {dest}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)


def fetch_summe(root: Path, downloads: Path) -> None:
    summe = root / "SumMe"
    if (summe / "GT").is_dir() and any((summe / "GT").glob("*.mat")):
        print("SumMe: already present")
        return
    print("SumMe:")
    zip_path = downloads / "SumMe.zip"
    try:
        _download(SUMME_ZIP_URL, zip_path)
    except Exception as e:
        print(f"  automatic download failed ({e}).")
        print(f"  Download {SUMME_ZIP_URL} manually into {downloads}/ and re-run.")
        return
    _unzip(zip_path, downloads / "SumMe_extracted")
    # The archive's top folder name has varied ("SumMe", "SumMe/SumMe", ...).
    src = next((p for p in (downloads / "SumMe_extracted").rglob("GT") if p.is_dir()), None)
    if src is None:
        print("  could not find a GT/ folder in the archive; inspect it manually.")
        return
    base = src.parent
    summe.mkdir(parents=True, exist_ok=True)
    for sub in ("GT", "videos"):
        s = base / sub
        if s.is_dir():
            shutil.copytree(s, summe / sub, dirs_exist_ok=True)
    print(f"  ready: {summe}")


def fetch_tvsum_videos(root: Path) -> None:
    tv = root / "TVSum"
    info = next(tv.glob("**/ydata-tvsum50-info.tsv"), None)
    if info is None:
        print("TVSum: no ydata-tvsum50-info.tsv found; place the annotation files first.")
        return
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        print("TVSum: yt-dlp not installed (`pip install yt-dlp`).")
        return
    import subprocess

    vdir = tv / "video"
    vdir.mkdir(parents=True, exist_ok=True)
    rows = info.read_text(encoding="utf-8").splitlines()[1:]
    print(f"TVSum: fetching {len(rows)} videos into {vdir}")
    for line in rows:
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        vid, url = parts[0], parts[3]
        out = vdir / f"{vid}.mp4"
        if out.exists():
            continue
        print(f"  {vid}")
        subprocess.run(
            ["yt-dlp", "-f", "mp4", "-o", str(out), url],
            check=False, capture_output=True,
        )


def verify(root: Path) -> None:
    print("\nLayout check:")
    checks = [
        (root / "SumMe" / "GT", "*.mat", "SumMe annotations"),
        (root / "SumMe" / "videos", "*", "SumMe videos"),
        (root / "TVSum", "**/ydata-tvsum50*.mat", "TVSum .mat"),
        (root / "TVSum" / "video", "*.mp4", "TVSum videos"),
    ]
    for path, pat, label in checks:
        n = len(list(path.glob(pat))) if path.exists() else 0
        print(f"  [{'ok' if n else '  '}] {label:22} {n} files  ({path})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/highlight"))
    ap.add_argument("--tvsum-videos", action="store_true",
                    help="fetch TVSum videos via yt-dlp from the info.tsv URLs")
    ap.add_argument("--skip-summe", action="store_true")
    args = ap.parse_args()

    root = args.root.resolve()
    downloads = root / "downloads"
    root.mkdir(parents=True, exist_ok=True)

    if not args.skip_summe:
        fetch_summe(root, downloads)

    if args.tvsum_videos:
        fetch_tvsum_videos(root)
    else:
        print(TVSUM_HELP.format(root=root))

    verify(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
