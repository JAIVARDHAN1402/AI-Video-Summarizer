"""Run the LLM understanding layer on an already-processed video.

    python scripts/test_understand.py <video_id> [--type coding] [--topic "..."]

Needs GEMINI_API_KEY set (env or .env). Prints the structured result so the
explanation quality can be judged without a full pipeline run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video_id")
    ap.add_argument("--type", default="lecture")
    ap.add_argument("--topic", default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from core import understand

    if not understand.is_available():
        print("GEMINI_API_KEY not set — cannot test the LLM path.")
        print("Get a free key at https://aistudio.google.com/apikey, then:")
        print('  echo GEMINI_API_KEY=your_key_here > .env')
        return 1

    u = understand.generate(
        args.video_id, video_type=args.type, topic=args.topic, force=args.force
    )

    print(f"\n=== {u['video_type'].upper()}  |  model={u['model']}  |  {u['generated_in_sec']}s ===\n")
    print("OVERVIEW\n" + textwrap.fill(u["overview"], 88))

    print(f"\nEXPLANATION ({len(u['explanation'])} sections)")
    for i, b in enumerate(u["explanation"], 1):
        tag = "" if b["from_video"] else "  [added context]"
        print(f"\n{i}. [{int(b['start_sec'])//60}:{int(b['start_sec'])%60:02d}] {b['heading']}{tag}")
        print(textwrap.fill(b["body"], 88, initial_indent="   ", subsequent_indent="   "))

    print("\nIN SUMMARY\n" + textwrap.fill(u["summary"], 88))

    if u.get("key_points"):
        print("\nKEY POINTS")
        for p in u["key_points"]:
            print("  - " + p)

    if u.get("glossary"):
        print("\nGLOSSARY")
        for g in u["glossary"]:
            print(f"  {g['term']}: {g['definition']}")

    if u.get("coding"):
        c = u["coding"]
        print(f"\nALGORITHM: {c['algorithm_name']}")
        print(textwrap.fill(c.get("intuition", ""), 88))
        for s in c.get("steps", []):
            print("  - " + s)
        print(f"  time={c.get('time_complexity')}  space={c.get('space_complexity')}")
        for impl in c.get("implementations", []):
            print(f"\n--- {impl['language']} ---\n{impl['code']}")

    if u.get("meeting"):
        m = u["meeting"]
        print("\nDECISIONS")
        for d in m.get("decisions", []):
            print("  - " + d)
        print("ACTION ITEMS")
        for a in m.get("action_items", []):
            print(f"  - [{a.get('owner', '?')}] {a['task']} (due {a.get('due', '-')})")

    if u.get("scene_notes"):
        print(f"\nSCENE NOTES ({len(u['scene_notes'])})")
        for n in u["scene_notes"]:
            print(f"  [{int(n['start'])//60}:{int(n['start'])%60:02d}] {n['what_it_shows']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
