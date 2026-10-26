"""ffmpeg media extraction: 16 kHz mono WAV plus 3 to 6 JPEG frames."""
from __future__ import annotations

import base64
import subprocess
import tempfile
from pathlib import Path


class MediaError(RuntimeError):
    pass


def frame_times(duration: float, count: int | None = None) -> list[float]:
    """Pick 3 to 6 frame times.

    Paid-partnership labels usually sit in the first seconds, so two frames
    come from the opening (1.0 s and 2.5 s) and the rest spread evenly over the
    remaining video.
    """
    duration = max(0.5, float(duration))
    count = count or max(3, min(6, round(duration / 2)))
    if duration <= 4:
        step = duration / (count + 1)
        return [round(step * (i + 1), 2) for i in range(count)]
    head = [1.0, 2.5]
    rest = count - len(head)
    start, end = 4.0, max(4.0, duration - 0.5)
    if rest == 1:
        tail = [round((start + end) / 2, 2)]
    else:
        tail = [round(start + (end - start) * i / (rest - 1), 2) for i in range(rest)]
    return head + tail


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise MediaError("Could not read the video. Upload an MP4, MOV or WebM file.")
    return float(result.stdout.strip())


def extract_media(video: bytes) -> dict:
    """Return ``{"wav": bytes, "duration": float, "frames": [{"time", "image"(base64 jpeg)}]}``."""
    with tempfile.TemporaryDirectory(prefix="claimcheck-") as tmp:
        root = Path(tmp)
        source = root / "upload"
        source.write_bytes(video)
        duration = probe_duration(source)
        wav = root / "audio.wav"
        audio = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(source), "-vn", "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(wav)],
            capture_output=True,
        )
        frames = []
        for index, at in enumerate(frame_times(duration)):
            target = root / f"frame-{index}.jpg"
            run = subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-ss", f"{at:.2f}", "-i", str(source), "-frames:v", "1",
                 "-vf", "scale=640:-2", "-q:v", "4", str(target)],
                capture_output=True,
            )
            if run.returncode == 0 and target.exists():
                frames.append({"time": at, "image": base64.b64encode(target.read_bytes()).decode("ascii")})
        return {
            "wav": wav.read_bytes() if audio.returncode == 0 and wav.exists() else b"",
            "duration": round(duration, 2),
            "frames": frames,
        }
