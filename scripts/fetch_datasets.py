"""Fetch and lay out SumMe + TVSum for highlight-model training.

    python scripts/fetch_datasets.py --root data/highlight

Produces:
    <root>/SumMe/GT/*.mat  +  <root>/SumMe/videos/*
    <root>/TVSum/ydata-tvsum50.mat  +  <root>/TVSum/video/*.mp4

Both are public single-archive downloads and are fetched automatically —
SumMe ~2.2 GB from ETH Zurich, TVSum ~641 MB from the author's MIT page
(videos included). ``--tvsum-videos`` is only needed if that archive ever
ships without its ``video/`` folder; it re-fetches the 50 clips from the URLs
in ``ydata-tvsum50-info.tsv`` with yt-dlp.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import urllib.request
import tarfile
import zipfile
from pathlib import Path

SUMME_ZIP_URL = "https://data.vision.ee.ethz.ch/cvl/SumMe/SumMe.zip"
# TVSum left Yahoo! Webscope and the old Google Drive mirror; the author now
# serves it directly, which means we can just fetch it (~641 MB, videos included).
TVSUM_TGZ_URL = "http://people.csail.mit.edu/yalesong/tvsum/tvsum50_ver_1_1.tgz"


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

    if not (tmp.exists() and tmp.stat().st_size > 0):
        urllib.request.urlretrieve(url, tmp, reporthook=hook)
        sys.stdout.write("\n")
    else:
        print(f"    resuming from a complete .part ({tmp.stat().st_size / 1e6:.0f} MB)")
    _rename_with_retry(tmp, dest)


def _rename_with_retry(tmp: Path, dest: Path, attempts: int = 12) -> None:
    """Rename, retrying while something else still holds the file.

    On Windows a sync client (OneDrive, Dropbox) or an indexer grabs a
    freshly-written file and the rename fails with WinError 32 — after a
    641 MB download, which is not a thing to redo over a transient lock.
    """
    for i in range(attempts):
        try:
            tmp.replace(dest)
            return
        except OSError as e:
            # Windows reports a failure even when the move went through, and a
            # sync client can whisk the .part away right after. Trust the
            # destination over the error.
            if dest.exists() and dest.stat().st_size > 0:
                return
            if i == attempts - 1:
                raise
            wait = min(2 ** i * 0.25, 10)
            print(f"    {dest.name} is locked ({e.__class__.__name__}); retrying in {wait:.0f}s")
            time.sleep(wait)


def _unpack(archive: Path, dest: Path) -> None:
    print(f"  unpacking {archive.name} -> {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    if archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    else:
        with tarfile.open(archive) as t:
            t.extractall(dest)


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
    _unpack(zip_path, downloads / "SumMe_extracted")
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


def fetch_tvsum(root: Path, downloads: Path) -> None:
    tv = root / "TVSum"
    if any(tv.glob("**/ydata-tvsum50*.mat")):
        print("TVSum: already present")
        return
    print("TVSum:")
    tgz = downloads / "tvsum50_ver_1_1.tgz"
    try:
        _download(TVSUM_TGZ_URL, tgz)
    except Exception as e:
        print(f"  automatic download failed ({e}).")
        print(f"  Download {TVSUM_TGZ_URL} manually into {downloads}/ and re-run.")
        return
    extracted = downloads / "TVSum_extracted"
    _unpack(tgz, extracted)

    # The tarball holds a second layer of zips — ydata-tvsum50-{data,matlab,
    # video,thumbnail}.zip — so unpack the ones we need in place.
    for name in ("data", "matlab", "video"):
        inner = next(extracted.rglob(f"ydata-tvsum50-{name}.zip"), None)
        if inner:
            _unpack(inner, inner.parent)

    tv.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in list(extracted.rglob("ydata-tvsum50*.mat")) + list(extracted.rglob("*.tsv")):
        shutil.copy2(src, tv / src.name)
        copied += 1
    if not copied:
        print("  could not find the annotations in the archive; inspect it manually.")
        return

    vdir = next((p for p in extracted.rglob("video") if p.is_dir() and any(p.iterdir())), None)
    if vdir:
        shutil.copytree(vdir, tv / "video", dirs_exist_ok=True)
    print(f"  ready: {tv}")


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
                    help="also fetch TVSum videos via yt-dlp, if the archive had no video/ folder")
    ap.add_argument("--skip-summe", action="store_true")
    ap.add_argument("--skip-tvsum", action="store_true")
    args = ap.parse_args()

    root = args.root.resolve()
    downloads = root / "downloads"
    root.mkdir(parents=True, exist_ok=True)

    if not args.skip_summe:
        fetch_summe(root, downloads)
    if not args.skip_tvsum:
        fetch_tvsum(root, downloads)
    # Only needed if the archive shipped without its video/ folder.
    if args.tvsum_videos:
        fetch_tvsum_videos(root)

    verify(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
