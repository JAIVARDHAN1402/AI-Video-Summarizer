"""The highlight endpoint must not build a long reel inside a request.

A ten-hour tutorial whose reel was never produced used to hold the connection
open while it built one; the browser reported a socket hang-up and the studio
tile never loaded. Past a configured length the request is refused with an
explanation instead.
"""
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from core import pipeline  # noqa: E402
from server import app as app_mod  # noqa: E402
from server import config  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(config, "API_KEYS", set())
    monkeypatch.setattr(config, "JWT_SECRET", None)
    monkeypatch.setattr(config, "AUTH_ENABLED", False)
    monkeypatch.setattr(app_mod, "_ensure_local_artifacts", lambda *a, **k: None)
    return TestClient(app_mod.app)


def _no_reel(monkeypatch, built=None):
    """No cached reel; record whether a build was attempted."""
    calls = []

    def fake(video_id, *, build=False):
        calls.append(build)
        return built if build else None

    monkeypatch.setattr(pipeline, "get_highlight", fake)
    return calls


def test_long_video_is_refused_rather_than_built(client, monkeypatch):
    monkeypatch.setattr(pipeline, "get_meta",
                        lambda v: {"duration_sec": 602 * 60})
    monkeypatch.setattr(config, "HIGHLIGHT_ONDEMAND_MAX_MIN", 45.0)
    calls = _no_reel(monkeypatch)

    r = client.get("/api/videos/abc/highlight")

    assert r.status_code == 409
    assert "too long to build while you wait" in r.json()["detail"]
    # It must not have tried to build one.
    assert True not in calls


def test_short_video_still_builds_on_demand(client, monkeypatch, tmp_path):
    reel = tmp_path / "highlight.mp4"
    reel.write_bytes(b"\x00\x00\x00\x18ftypisom")
    monkeypatch.setattr(pipeline, "get_meta",
                        lambda v: {"duration_sec": 11 * 60})
    monkeypatch.setattr(config, "HIGHLIGHT_ONDEMAND_MAX_MIN", 45.0)
    calls = _no_reel(monkeypatch, built=str(reel))

    r = client.get("/api/videos/abc/highlight")

    assert r.status_code == 200
    assert True in calls, "a short video should still be built on demand"


def test_cached_reel_is_served_whatever_the_length(client, monkeypatch,
                                                  tmp_path):
    """A long video that already has a reel is served, not refused."""
    reel = tmp_path / "highlight.mp4"
    reel.write_bytes(b"\x00\x00\x00\x18ftypisom")
    monkeypatch.setattr(pipeline, "get_meta",
                        lambda v: {"duration_sec": 602 * 60})
    monkeypatch.setattr(config, "HIGHLIGHT_ONDEMAND_MAX_MIN", 45.0)
    monkeypatch.setattr(pipeline, "get_highlight",
                        lambda v, *, build=False: str(reel))

    r = client.get("/api/videos/abc/highlight")

    assert r.status_code == 200
