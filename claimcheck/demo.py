"""Hosted-demo safety, switched on with DEMO_MODE=1.

- Replay always works and needs no key.
- Live checks are limited per IP per UTC day (DEMO_DAILY_LIMIT, default 3),
  questions too (DEMO_ASK_LIMIT, default 10).
- Uploads are capped (DEMO_MAX_MB, default 25) and so is clip length
  (DEMO_MAX_SECONDS, default 60).
- The original upload is deleted as soon as the check ends; the browser-safe
  copy used for playback is deleted DEMO_KEEP_SECONDS later (default 600).
- The API key stays on the server: no endpoint returns it.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException, Request

USAGE: dict[tuple[str, str, str], int] = defaultdict(int)
LOCK = threading.Lock()


def enabled() -> bool:
    return os.getenv("DEMO_MODE", "").strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def client_ip(request: Request) -> str:
    if os.getenv("DEMO_TRUST_PROXY") == "1" and request.headers.get("x-forwarded-for"):
        return request.headers["x-forwarded-for"].split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _key(request: Request, kind: str) -> tuple[str, str, str]:
    return client_ip(request), datetime.now(timezone.utc).date().isoformat(), kind


def _take(request: Request, kind: str, limit: int, what: str):
    key = _key(request, kind)
    with LOCK:
        if USAGE[key] >= limit:
            raise HTTPException(429, f"The demo allows {limit} {what} per day from one address. The recorded replay is always available.")
        USAGE[key] += 1


def public_limits(request: Request) -> dict | None:
    if not enabled():
        return None
    limit = _int("DEMO_DAILY_LIMIT", 3)
    used = USAGE.get(_key(request, "check"), 0)
    return {"checks_per_day": limit, "checks_left": max(0, limit - used), "max_mb": _int("DEMO_MAX_MB", 25),
            "max_seconds": _int("DEMO_MAX_SECONDS", 60)}


def check_upload(request: Request, video: bytes, suffix: str, image_bytes: int = 0):
    """Refuse an upload that breaks the demo caps, then count it against the address."""
    if not enabled():
        return
    max_bytes = _int("DEMO_MAX_MB", 25) * 1024 * 1024
    if len(video) + image_bytes > max_bytes:
        raise HTTPException(413, f"The demo accepts uploads up to {_int('DEMO_MAX_MB', 25)} MB.")
    if video:
        from .media import MediaError, probe
        with tempfile.TemporaryDirectory(prefix="claimcheck-demo-") as tmp:
            path = Path(tmp) / f"clip{suffix}"
            path.write_bytes(video)
            try:
                duration = probe(path)["duration"]
            except MediaError as error:
                raise HTTPException(415, str(error))
        if duration > _int("DEMO_MAX_SECONDS", 60):
            raise HTTPException(413, f"The demo checks clips up to {_int('DEMO_MAX_SECONDS', 60)} seconds; this one is {duration:.0f} s.")
    _take(request, "check", _int("DEMO_DAILY_LIMIT", 3), "live checks")


def check_ask(request: Request):
    if enabled():
        _take(request, "ask", _int("DEMO_ASK_LIMIT", 10), "questions")


def cleanup(job) -> None:
    """Delete the upload when the check ends (demo mode); the playback copy follows after a short delay."""
    if not enabled():
        return
    original = getattr(job, "original", None)
    if original and Path(original).exists() and original != getattr(job, "video_path", None):
        Path(original).unlink(missing_ok=True)
    folder = getattr(job, "folder", None)
    if folder:
        delay = _int("DEMO_KEEP_SECONDS", 600)
        if delay <= 0:
            shutil.rmtree(folder, ignore_errors=True)
        else:
            timer = threading.Timer(delay, shutil.rmtree, args=(folder,), kwargs={"ignore_errors": True})
            timer.daemon = True
            timer.start()
