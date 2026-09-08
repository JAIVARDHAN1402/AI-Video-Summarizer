"""Storage seam: where uploads land and where finished artifacts are kept.

Two backends behind one interface. ``LocalStorage`` (the default) keeps
everything on disk — uploads under ``config.UPLOAD_DIR``, per-video artifacts
under the pipeline's own ``cache/<video_id>/``. ``S3Storage`` mirrors both to
an S3-compatible bucket (AWS S3, Cloudflare R2, MinIO) so workers on different
machines share them.

The pipeline itself is untouched: it always reads and writes
``cache/<video_id>/`` locally. S3Storage pulls an upload back down to a local
path before a job runs, and pushes the cache tree up after it finishes.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path
from typing import BinaryIO, Optional

from server import config

logger = logging.getLogger("server.storage")

_CHUNK = 1024 * 1024


class StorageError(RuntimeError):
    pass


def _sha256_stream(src: BinaryIO, dest: BinaryIO, limit: int) -> tuple[str, int]:
    """Copy ``src`` -> ``dest`` in chunks, hashing as we go, enforcing ``limit``."""
    h = hashlib.sha256()
    total = 0
    while True:
        chunk = src.read(_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise StorageError(f"upload exceeds {limit} bytes")
        h.update(chunk)
        dest.write(chunk)
    return h.hexdigest(), total


class LocalStorage:
    backend = "local"

    def __init__(self) -> None:
        self.upload_dir = Path(config.UPLOAD_DIR)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    # --- uploads ---------------------------------------------------------

    def save_upload(self, fileobj: BinaryIO, filename: str) -> dict:
        ext = Path(filename).suffix.lower() or ".mp4"
        tmp = self.upload_dir / f".incoming-{id(fileobj):x}{ext}"
        try:
            with tmp.open("wb") as out:
                digest, size = _sha256_stream(fileobj, out, config.MAX_UPLOAD_BYTES)
            final = self.upload_dir / f"{digest}{ext}"
            if final.exists():
                tmp.unlink(missing_ok=True)
            else:
                tmp.rename(final)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        return {"upload_id": digest, "filename": filename, "bytes": size, "ext": ext}

    def upload_path(self, upload_id: str) -> Path:
        for p in self.upload_dir.glob(f"{upload_id}.*"):
            return p
        raise StorageError(f"no upload with id {upload_id}")

    # --- artifacts -----------------------------------------------------------

    def mirror_artifacts(self, video_id: str, local_dir: Path) -> None:
        # Already local — nothing to do.
        return

    def fetch_artifacts(self, video_id: str, local_dir: Path) -> bool:
        return local_dir.is_dir()


class S3Storage:
    backend = "s3"

    def __init__(self) -> None:
        if not config.S3_BUCKET:
            raise StorageError("VS_STORAGE_BACKEND=s3 but VS_S3_BUCKET is unset")
        try:
            import boto3  # type: ignore
        except ImportError as e:  # pragma: no cover - depends on optional dep
            raise StorageError("boto3 not installed — `pip install boto3`") from e
        self._s3 = boto3.client(
            "s3", endpoint_url=config.S3_ENDPOINT_URL, region_name=config.S3_REGION,
        )
        self.bucket = config.S3_BUCKET
        self.prefix = config.S3_PREFIX
        self.upload_dir = Path(config.UPLOAD_DIR)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, *parts: str) -> str:
        return "/".join([p for p in (self.prefix, *parts) if p])

    # --- uploads ---------------------------------------------------------

    def save_upload(self, fileobj: BinaryIO, filename: str) -> dict:
        ext = Path(filename).suffix.lower() or ".mp4"
        local = LocalStorage().save_upload(fileobj, filename)
        src = self.upload_dir / f"{local['upload_id']}{ext}"
        self._s3.upload_file(str(src), self.bucket, self._key("uploads", src.name))
        return local

    def upload_path(self, upload_id: str) -> Path:
        # Serve from local cache if we still have it, else pull it down.
        for p in self.upload_dir.glob(f"{upload_id}.*"):
            return p
        for ext in config.UPLOAD_ALLOWED_EXTS:
            key = self._key("uploads", f"{upload_id}{ext}")
            dest = self.upload_dir / f"{upload_id}{ext}"
            try:
                self._s3.download_file(self.bucket, key, str(dest))
                return dest
            except Exception:
                continue
        raise StorageError(f"upload {upload_id} not found locally or in s3://{self.bucket}")

    # --- artifacts -----------------------------------------------------------

    def mirror_artifacts(self, video_id: str, local_dir: Path) -> None:
        if not local_dir.is_dir():
            return
        for path in local_dir.rglob("*"):
            if path.is_file():
                rel = path.relative_to(local_dir).as_posix()
                self._s3.upload_file(str(path), self.bucket, self._key("artifacts", video_id, rel))
        logger.info("mirrored %s -> s3://%s/%s", video_id, self.bucket, self._key("artifacts", video_id))

    def fetch_artifacts(self, video_id: str, local_dir: Path) -> bool:
        if local_dir.is_dir() and any(local_dir.iterdir()):
            return True
        prefix = self._key("artifacts", video_id) + "/"
        paginator = self._s3.get_paginator("list_objects_v2")
        found = False
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                found = True
                rel = obj["Key"][len(prefix):]
                dest = local_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                self._s3.download_file(self.bucket, obj["Key"], str(dest))
        return found


_backend: Optional[LocalStorage | S3Storage] = None


def get_storage() -> LocalStorage | S3Storage:
    global _backend
    if _backend is None:
        _backend = S3Storage() if config.STORAGE_BACKEND == "s3" else LocalStorage()
        logger.info("storage backend: %s", _backend.backend)
    return _backend
