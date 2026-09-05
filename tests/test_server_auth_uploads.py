"""Auth dependency, file uploads, and the storage seam."""

import io

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from server import app as app_mod  # noqa: E402
from server import auth, config, schemas, storage  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(config, "STORAGE_BACKEND", "local")
    monkeypatch.setattr(storage, "_backend", None)
    # auth off unless a test turns it on
    monkeypatch.setattr(config, "API_KEYS", set())
    monkeypatch.setattr(config, "JWT_SECRET", None)
    monkeypatch.setattr(config, "AUTH_ENABLED", False)
    return TestClient(app_mod.app)


# --- auth ------------------------------------------------------------------


def _enable_key_auth(monkeypatch, key="secret-key"):
    monkeypatch.setattr(config, "API_KEYS", {key})
    monkeypatch.setattr(config, "AUTH_ENABLED", True)
    return key


def test_open_when_auth_disabled(client):
    assert client.get("/api/jobs").status_code == 200


def test_health_is_public_even_with_auth_on(client, monkeypatch):
    _enable_key_auth(monkeypatch)
    assert client.get("/api/health").status_code == 200


def test_protected_route_needs_token(client, monkeypatch):
    key = _enable_key_auth(monkeypatch)
    assert client.get("/api/jobs").status_code == 401
    assert client.get("/api/jobs", headers={"Authorization": "Bearer wrong"}).status_code == 401
    ok = client.get("/api/jobs", headers={"Authorization": f"Bearer {key}"})
    assert ok.status_code == 200


def test_keyframe_path_stays_public_for_img_tags(client, monkeypatch):
    _enable_key_auth(monkeypatch)
    # 404 (no such video) rather than 401 — the auth dependency let it through.
    r = client.get("/api/videos/deadbeef/keyframes/scene_000.jpg")
    assert r.status_code == 404


def test_jwt_token_accepted(client, monkeypatch):
    jwt = pytest.importorskip("jwt")
    monkeypatch.setattr(config, "API_KEYS", set())
    monkeypatch.setattr(config, "JWT_SECRET", "topsecret")
    monkeypatch.setattr(config, "JWT_ALGORITHM", "HS256")
    monkeypatch.setattr(config, "JWT_AUDIENCE", None)
    monkeypatch.setattr(config, "AUTH_ENABLED", True)
    token = jwt.encode({"sub": "user-1"}, "topsecret", algorithm="HS256")
    r = client.get("/api/jobs", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    bad = client.get("/api/jobs", headers={"Authorization": "Bearer " + token + "x"})
    assert bad.status_code == 401


# --- uploads -------------------------------------------------------------


def test_upload_rejects_unknown_extension(client):
    r = client.post("/api/uploads", files={"file": ("notes.txt", b"hi", "text/plain")})
    assert r.status_code == 415


def test_upload_stores_and_dedups_by_content_hash(client):
    payload = b"\x00\x01\x02fake-mp4-bytes" * 100
    r1 = client.post("/api/uploads", files={"file": ("a.mp4", io.BytesIO(payload), "video/mp4")})
    assert r1.status_code == 201, r1.text
    body = r1.json()
    assert len(body["upload_id"]) == 64 and body["bytes"] == len(payload)

    # same bytes, different name -> same id, no duplicate file
    r2 = client.post("/api/uploads", files={"file": ("b.mp4", io.BytesIO(payload), "video/mp4")})
    assert r2.json()["upload_id"] == body["upload_id"]
    assert len(list(config.UPLOAD_DIR.glob("*.mp4"))) == 1


def test_upload_enforces_size_limit(client, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 16)
    r = client.post("/api/uploads", files={"file": ("big.mp4", b"x" * 64, "video/mp4")})
    assert r.status_code == 413


def test_create_job_requires_exactly_one_source():
    with pytest.raises(ValueError):
        schemas.CreateJobRequest(url="http://x", upload_id="abc")
    with pytest.raises(ValueError):
        schemas.CreateJobRequest()
    schemas.CreateJobRequest(url="https://youtu.be/x")  # ok
    schemas.CreateJobRequest(upload_id="abc123")        # ok


def test_create_job_with_unknown_upload_id_is_404(client):
    r = client.post("/api/jobs", json={"upload_id": "0" * 64, "video_type": "lecture"})
    assert r.status_code == 404
