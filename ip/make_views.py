"""Capture the design-registration views from the running application.

Run the servers first (scripts/serve.ps1), then:

    .venv/Scripts/python.exe ip/make_views.py

It signs in as a throwaway test account so no personal data reaches the
images, captures every screen at 1440 x 900 at a device pixel ratio of two,
cuts the panel views at boundaries measured from the live DOM, and writes the
result to ip/views/. Then rebuild the form with ip/make_idf.py.

Two things are done to the app for the duration of the run and undone
afterwards: a seed page is served from web/public so a session token can be
placed in localStorage for the origin, and the Next.js development badge is
switched off, because it is not part of the design and must not appear in a
filed representation.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
ORIGIN = os.environ.get("VS_VIEWS_ORIGIN", "http://localhost:3000")
VIDEO_ID = os.environ.get("VS_VIEWS_VIDEO", "71138c819b23ed5d")

TMP = os.path.join(tempfile.gettempdir(), "vs-views")
RAW = os.path.join(TMP, "raw")
OUT = os.path.join(ROOT, "ip", "views")
PROF_IN = os.path.join(TMP, "prof-in")
PROF_OUT = os.path.join(TMP, "prof-out")

SEED_PAGE = os.path.join(ROOT, "web", "public", "__seed.html")
NEXT_CONFIG = os.path.join(ROOT, "web", "next.config.ts")

S = 2  # device pixel ratio of every capture

# Boundaries of the three panels and the header, in CSS pixels at 1440x900,
# read from the live DOM with getBoundingClientRect rather than estimated.
HEADER = (0, 0, 1440, 60)
SOURCE = (8, 56, 317, 836)
STUDIO = (1086, 56, 346, 836)

SEED_HTML = """<!doctype html>
<meta charset="utf-8">
<title>seed</title>
<body style="font:14px system-ui;padding:2rem">seeding\u2026</body>
<script>
  // Written by ip/make_views.py for the duration of a capture run and deleted
  // afterwards. It copies a session token and a theme, handed to it in the URL
  // fragment, into localStorage for this origin.
  try {
    var parts = decodeURIComponent(location.hash.slice(1)).split("|");
    if (parts[1]) localStorage.setItem("theme", parts[1]);
    if (parts[0]) localStorage.setItem("vs-session-token", parts[0]);
    else localStorage.removeItem("vs-session-token");
    document.title = "OK";
  } catch (e) {
    document.title = "FAIL";
  }
</script>
"""


# --------------------------------------------------------------- app set-up
def mint_token():
    """A throwaway account, so the captures carry no personal data."""
    from server import db, identity
    session = db.SessionLocal()
    try:
        user = identity._upsert(session, channel="email",
                                destination="design-views@vit.test",
                                method="otp", name="Design Views")
        return identity.issue_token(user)
    finally:
        session.close()


def set_dev_badge(on):
    with open(NEXT_CONFIG, encoding="utf-8") as fh:
        text = fh.read()
    marker = "  devIndicators: false,\n"
    has = marker in text
    if on and has:
        text = text.replace(marker, "")
    elif not on and not has:
        text = text.replace("  allowedDevOrigins: TUNNEL_HOSTS,\n",
                            "  allowedDevOrigins: TUNNEL_HOSTS,\n" + marker)
    else:
        return
    with open(NEXT_CONFIG, "w", encoding="utf-8") as fh:
        fh.write(text)
    time.sleep(6)  # let the dev server pick the change up


# ------------------------------------------------------------------ capture
def chrome(profile, url, out=None, w=1440, h=900, budget=15000,
           transparent=False):
    cmd = [
        CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
        "--no-first-run", "--no-default-browser-check",
        "--disable-features=Translate,MediaRouter",
        f"--user-data-dir={profile}",
        f"--window-size={w},{h}",
        "--force-device-scale-factor=2",
        f"--virtual-time-budget={budget}",
    ]
    if transparent:
        cmd.append("--default-background-color=00000000")
    cmd.append(f"--screenshot={out}" if out else "--dump-dom")
    cmd.append(url)
    # Not capture_output: a lingering child inherits the pipes and holds them
    # open long after Chrome itself has exited, so communicate() never returns.
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   timeout=240)


def shot(profile, name, url, **kw):
    path = os.path.join(RAW, name + ".png")
    if os.path.exists(path):
        os.remove(path)
    chrome(profile, url, path, **kw)
    if not os.path.exists(path):
        raise SystemExit(f"capture failed: {name} <- {url}")
    print(f"  captured {name:14s} {Image.open(path).size}")


def seed(profile, theme, token):
    """Start from an empty profile, then write the token and the theme.

    Reusing a profile that already holds the other theme left the old value in
    place often enough to produce a dark image in the middle of a light run.
    Two passes with a pause, because localStorage is flushed asynchronously.
    """
    shutil.rmtree(profile, ignore_errors=True)
    url = f"{ORIGIN}/__seed.html#{token}|{theme}"
    for _ in range(2):
        chrome(profile, url, None, 600, 300, 4000)
        time.sleep(2)


# ------------------------------------------------------------------- shaping
def load(name):
    return Image.open(os.path.join(RAW, name + ".png")).convert("RGB")


def save(im, name, note):
    im.save(os.path.join(OUT, name + ".png"))
    print(f"  {name:28s} {str(im.size):14s} {note}")


def crop(im, box, pad=0):
    x, y, w, h = box
    out = im.crop((x * S, y * S, (x + w) * S, (y + h) * S))
    if not pad:
        return out
    bg = Image.new("RGB", (out.width + 2 * pad, out.height + 2 * pad), "white")
    bg.paste(out, (pad, pad))
    return bg


def content_bottom(im, tol=6):
    """Last row that is not blank, so a tall capture can be trimmed.

    The entry screen is captured in a window taller than the page so that
    nothing is cut off, which leaves empty white underneath.
    """
    rows = np.where(np.asarray(im.convert("L")).min(axis=1) < 255 - tol)[0]
    return (int(rows[-1]) + 1) if len(rows) else im.height


# ---------------------------------------------------------------------- main
def main():
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    if not os.path.exists(CHROME):
        raise SystemExit(f"Chrome not found at {CHROME}")

    token = mint_token()
    os.makedirs(os.path.dirname(SEED_PAGE), exist_ok=True)
    with open(SEED_PAGE, "w", encoding="utf-8") as fh:
        fh.write(SEED_HTML)
    set_dev_badge(False)

    try:
        print("capturing:")
        seed(PROF_IN, "light", token)
        shot(PROF_IN, "work", f"{ORIGIN}/v/{VIDEO_ID}")
        shot(PROF_IN, "newjob", f"{ORIGIN}/app")
        shot(PROF_IN, "library", f"{ORIGIN}/library")
        shot(PROF_IN, "notebooks", f"{ORIGIN}/notebooks")

        seed(PROF_IN, "dark", token)
        shot(PROF_IN, "work_dark", f"{ORIGIN}/v/{VIDEO_ID}")
        shot(PROF_IN, "newjob_dark", f"{ORIGIN}/app")

        seed(PROF_OUT, "light", "")          # signed out: the entry screen
        shot(PROF_OUT, "entry", f"{ORIGIN}/")
        shot(PROF_OUT, "entry_full", f"{ORIGIN}/", h=3300, budget=20000)
        shot(PROF_OUT, "mark", f"{ORIGIN}/icon.svg", w=1200, h=1200,
             budget=8000, transparent=True)
    finally:
        set_dev_badge(True)
        if os.path.exists(SEED_PAGE):
            os.remove(SEED_PAGE)
        pub = os.path.dirname(SEED_PAGE)
        if os.path.isdir(pub) and not os.listdir(pub):
            os.rmdir(pub)

    print("\nviews ->", OUT)
    work, entry = load("work"), load("entry")
    entry_full = load("entry_full")

    save(work, "view-1-front", "working three-panel screen")
    save(entry, "view-2-rear", "entry screen")
    save(crop(work, SOURCE, pad=24), "view-3-left", "source panel alone")
    save(crop(work, STUDIO, pad=24), "view-4-right", "studio tile column alone")
    save(crop(work, HEADER, pad=20), "view-5-top", "header bar")

    bottom = content_bottom(entry_full) // S
    save(crop(entry_full, (0, bottom - 560, 1440, 560), pad=20),
         "view-6-bottom", "entry screen, closing band and footer")

    # Not a 45-degree perspective: the subject is a screen display, and
    # tilting a flat screen shears the type and reads as a broken layout
    # rather than as the design. A further screen is submitted instead.
    save(load("newjob"), "view-7-submission", "the submission screen")

    mark = Image.open(os.path.join(RAW, "mark.png")).convert("RGBA")
    white = Image.new("RGBA", mark.size, (255, 255, 255, 255))
    white.alpha_composite(mark)
    save(white.convert("RGB"), "view-8-device-mark", "the mark, on white")

    for src, dst, note in [
        ("library", "extra-1-library-screen", "the library screen"),
        ("notebooks", "extra-2-notebooks-screen", "the notebooks screen"),
        ("work_dark", "extra-3-dark-colourway", "working screen, dark"),
        ("newjob_dark", "extra-4-dark-submission", "submission screen, dark"),
    ]:
        save(load(src), dst, note)

    print("\nNow rebuild the forms:  python ip/make_idf.py")


if __name__ == "__main__":
    main()
