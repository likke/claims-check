"""Video media extraction helpers."""
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def extract_media(video: bytes) -> tuple[bytes, list[dict[str, Any]]]:
    """Extract mono 16 kHz audio and up to six evenly spaced video frames."""
    with tempfile.TemporaryDirectory(prefix="claim-check-") as tmp:
        root = Path(tmp)
        source = root / "upload.bin"
        audio_path = root / "audio.wav"
        source.write_bytes(video)
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(source), "-vn", "-ac", "1", "-ar", "16000", str(audio_path)], check=True, capture_output=True)
        duration_text = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(source)], check=True, capture_output=True, text=True).stdout.strip()
        duration = max(1, int(float(duration_text) + .5))
        count = min(6, max(3, duration))
        times = sorted(set(round(i * max(0, duration - 1) / max(1, count - 1), 2) for i in range(count)))
        frames = []
        for index, at in enumerate(times):
            frame = root / f"frame-{index}.jpg"
            subprocess.run(["ffmpeg", "-v", "error", "-ss", str(at), "-i", str(source), "-frames:v", "1", "-q:v", "3", str(frame)], check=True, capture_output=True)
            frames.append({"time": at, "image": frame.read_bytes()})
        return audio_path.read_bytes(), frames
