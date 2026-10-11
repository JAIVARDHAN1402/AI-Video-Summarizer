"""Audio Overview: two hosts talking through the video, as an audio file.

Two stages. First a provider writes a dialogue from the understanding document
— not a reading of it, a conversation that gets the ideas across in the order a
listener can follow. Then each turn is synthesised in its own voice and the
turns are concatenated into one track.

Speech synthesis is Microsoft Edge's free endpoint via ``edge-tts``: no key, no
account, and voices good enough that the result is listenable, including Indian
English. It needs network access but nothing else, so it does not change the
deployment's shape the way a local neural TTS would.

Per-turn durations are measured after synthesis and stored with the script, so
the player can highlight the line being spoken without guessing.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import config
from core import cache, llm

logger = logging.getLogger(__name__)

# One voice per host, deliberately different in gender and accent so a listener
# can tell them apart without being told which is speaking.
VOICE_A = "en-US-AndrewNeural"   # the one who asks and steers
VOICE_B = "en-US-AvaNeural"      # the one who explains

# edge-tts talks to a remote endpoint. Turns are independent so they synthesise
# concurrently, but not without limit — too many at once gets throttled and
# ends up slower than a smaller batch.
_MAX_CONCURRENT_TTS = 6

_TARGET_TURNS = 26
_MAX_SOURCE_CHARS = 48_000

_PERSONA = """You write a two-person audio conversation that teaches what a \
video covers, for someone who will listen without watching it.

HOST_A asks, steers and pushes back. HOST_B explains.

What makes this good:
- It teaches. A listener who knows nothing should finish understanding the \
actual ideas, not just that the video exists.
- HOST_A's questions are the ones a real learner has: "why does that matter", \
"what breaks if you don't", "how is that different from X". Never "wow" or \
"that's fascinating".
- HOST_B answers in plain speech. Concrete before abstract. One idea per turn.
- Build in order: each turn should depend on the one before it.
- Disagreement and correction are allowed and make it better.

What ruins it, do not do any of these:
- Enthusiasm as filler. No "absolutely", "great question", "exactly right".
- Reading lists aloud. Turn structure into sentences.
- Meta-talk about the video, the episode, or "today we're discussing".
- Sign-offs, intros, music cues, or naming the hosts.
- Stage directions, emoji, or anything a voice cannot say.

Open on the actual first idea. Close on the thing worth remembering. \
Write only what is spoken."""

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "turns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "speaker": {"type": "string", "enum": ["HOST_A", "HOST_B"]},
                    "text": {
                        "type": "string",
                        "description": "What is spoken. One to four sentences.",
                    },
                },
                "required": ["speaker", "text"],
            },
        }
    },
    "required": ["turns"],
}


def is_available() -> bool:
    """A provider writes the script and ffmpeg joins the audio; both required."""
    return llm.get_client().available and shutil.which("ffmpeg") is not None


def _source_text(u: dict[str, Any]) -> str:
    parts: list[str] = []
    if u.get("overview"):
        parts.append(f"OVERVIEW\n{u['overview']}")
    if u.get("key_points"):
        parts.append("KEY POINTS\n" + "\n".join(f"- {p}" for p in u["key_points"]))
    if u.get("glossary"):
        parts.append(
            "TERMS\n"
            + "\n".join(f"- {g.get('term')}: {g.get('definition')}" for g in u["glossary"])
        )
    sections = u.get("explanation") or []
    if sections:
        body = [
            f"{s.get('heading') or ''}\n{(s.get('body') or '')[:1100]}" for s in sections
        ]
        parts.append("WALKTHROUGH\n" + "\n\n".join(body))
    return "\n\n".join(parts)[:_MAX_SOURCE_CHARS]


def _clean_turns(raw: dict[str, Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for t in raw.get("turns") or []:
        text = " ".join(str(t.get("text") or "").split())
        if not text:
            continue
        speaker = "HOST_A" if str(t.get("speaker")).upper() == "HOST_A" else "HOST_B"
        # A model that emits two turns in a row for the same host produces a
        # monologue read in one voice; merging them keeps the alternation.
        if out and out[-1]["speaker"] == speaker:
            out[-1]["text"] = f"{out[-1]['text']} {text}"
        else:
            out.append({"speaker": speaker, "text": text})
    return out


async def _synthesise(turns: list[dict[str, str]], workdir: Path) -> list[Path]:
    import edge_tts

    sem = asyncio.Semaphore(_MAX_CONCURRENT_TTS)

    async def one(i: int, turn: dict[str, str]) -> Path:
        path = workdir / f"{i:03d}.mp3"
        voice = VOICE_A if turn["speaker"] == "HOST_A" else VOICE_B
        async with sem:
            await edge_tts.Communicate(turn["text"], voice).save(str(path))
        return path

    return await asyncio.gather(*(one(i, t) for i, t in enumerate(turns)))


def _duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def _concat(parts: list[Path], dest: Path) -> None:
    """Join the turns into one track, re-encoding so the result is uniform.

    The concat demuxer can stream-copy when every input shares a codec and
    sample rate, which these do — but re-encoding once is cheap next to the
    synthesis and removes a whole class of "plays in one player, not another".
    """
    listing = dest.parent / "concat.txt"
    listing.write_text(
        "\n".join(f"file '{p.as_posix()}'" for p in parts), encoding="utf-8"
    )
    result = subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
         "-c:a", "libmp3lame", "-b:a", "96k", "-ar", "24000", str(dest)],
        capture_output=True, text=True,
    )
    listing.unlink(missing_ok=True)
    if result.returncode != 0 or not dest.exists():
        raise RuntimeError(f"ffmpeg could not join the audio: {result.stderr[-400:]}")


def audio_path(video_id: str) -> Path:
    return cache.get_cache_dir(video_id) / "audio_overview.mp3"


def get(video_id: str) -> Optional[dict[str, Any]]:
    """The cached overview's script and timings, if one has been made."""
    doc = cache.read_json(video_id, "audio_overview")
    if doc and audio_path(video_id).exists():
        return doc
    return None


def generate(video_id: str, *, force: bool = False) -> dict[str, Any]:
    """Write, speak and assemble the overview. Returns the script with timings.

    Raises:
        FileNotFoundError: the video has no understanding document.
        llm.LLMUnavailable / llm.LLMError: the script could not be written.
        RuntimeError: synthesis or assembly failed.
    """
    if not force:
        existing = get(video_id)
        if existing:
            return existing

    u = cache.read_json(video_id, "understanding")
    if not u:
        raise FileNotFoundError(
            "an audio overview is written from the explanation, and this video "
            "has none"
        )

    client = llm.get_client()
    if not client.available:
        raise llm.LLMUnavailable("no LLM provider configured")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is needed to assemble the audio and was not found")

    started = time.perf_counter()
    cache.set_state(video_id, "audio", 0.1, "Writing the conversation...")

    result = client.generate_json(
        _PERSONA,
        f"Write about {_TARGET_TURNS} turns of conversation covering this "
        f"material.\n\n{_source_text(u)}",
        schema=_SCHEMA,
        temperature=0.75,  # dialogue wants more variety than an explanation does
        max_output_tokens=config.UNDERSTANDING_MAX_OUTPUT_TOKENS,
    )
    turns = _clean_turns(result.data)
    if len(turns) < 4:
        raise llm.LLMError("the model did not produce a usable conversation")

    cache.set_state(video_id, "audio", 0.45, f"Speaking {len(turns)} turns...")
    dest = audio_path(video_id)
    dest.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        parts = asyncio.run(_synthesise(turns, workdir))
        cache.set_state(video_id, "audio", 0.85, "Joining the audio...")

        at = 0.0
        for turn, part in zip(turns, parts):
            turn["start_sec"] = round(at, 2)
            at += _duration(part)
            turn["end_sec"] = round(at, 2)

        _concat(parts, dest)

    elapsed = time.perf_counter() - started
    doc = {
        "video_id": video_id,
        "model": result.model,
        "voices": {"HOST_A": VOICE_A, "HOST_B": VOICE_B},
        "duration_sec": round(at, 1),
        "generated_in_sec": round(elapsed, 1),
        "turns": turns,
    }
    cache.write_json(video_id, "audio_overview", doc)
    cache.set_state(video_id, "audio", 1.0, "Audio overview ready")
    logger.info(
        "audio overview for %s: %d turns, %.0fs of audio, built in %.0fs",
        video_id, len(turns), at, elapsed,
    )
    return doc
