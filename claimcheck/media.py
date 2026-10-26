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


MAX_CHECK_SECONDS = 90
FORMATS = {".mp4", ".mov", ".m4v", ".webm"}
BROWSER_VIDEO = {"h264"}
BROWSER_AUDIO = {"aac", "mp3", None}
# Longest edge of sampled frames sent to the model; the short edge follows the displayed aspect.
FRAME_EDGE = 640


def probe(path: Path) -> dict:
    """Codec, stored size, rotation and the displayed (rotated) size of a clip.

    iPhone clips often store a portrait video as landscape pixels plus a rotation
    in the display matrix, so the displayed size swaps width and height for 90 or
    270 degrees.
    """
    import json

    result = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise MediaError("Could not read the video. Upload an MP4, MOV, M4V or WebM file.")
    data = json.loads(result.stdout or "{}")
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    audio = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
    if video is None:
        raise MediaError("The file has no video stream.")
    rotation = 0
    for side in video.get("side_data_list") or []:
        if "rotation" in side:
            rotation = int(round(float(side["rotation"])))
    if not rotation and (video.get("tags") or {}).get("rotate"):
        rotation = int(video["tags"]["rotate"])
    rotation %= 360
    width, height = int(video.get("width") or 0), int(video.get("height") or 0)
    display_w, display_h = (height, width) if rotation in (90, 270) else (width, height)
    ratio = display_w / display_h if display_h else 16 / 9
    orientation = "vertical" if ratio < 0.9 else "landscape" if ratio > 1.1 else "square"
    duration = float((data.get("format") or {}).get("duration") or video.get("duration") or 0)
    return {
        "width": width, "height": height, "rotation": rotation, "display_width": display_w, "display_height": display_h,
        "orientation": orientation, "video_codec": video.get("codec_name"), "audio_codec": audio.get("codec_name") if audio else None,
        "has_audio": audio is not None, "duration": round(duration, 2),
    }


def probe_duration(path: Path) -> float:
    return probe(path)["duration"]


def needs_proxy(info: dict) -> bool:
    return info["video_codec"] not in BROWSER_VIDEO or info["audio_codec"] not in BROWSER_AUDIO


def make_proxy(source: Path, target: Path, info: dict) -> Path:
    """A browser-safe H.264/AAC mp4: rotation applied (ffmpeg autorotates), orientation kept, short edge at most 1080."""
    short = "iw" if info["display_width"] <= info["display_height"] else "ih"
    scale = (f"scale='min(iw,1080)':-2" if short == "iw" else f"scale=-2:'min(ih,1080)'")
    command = ["ffmpeg", "-v", "error", "-y", "-i", str(source), "-vf", f"{scale},format=yuv420p", "-c:v", "libx264",
               "-preset", "veryfast", "-crf", "23", "-movflags", "+faststart", "-metadata:s:v:0", "rotate=0"]
    command += ["-c:a", "aac", "-b:a", "128k"] if info["has_audio"] else ["-an"]
    run = subprocess.run(command + [str(target)], capture_output=True)
    if run.returncode != 0 or not target.exists():
        raise MediaError("Could not make a playable copy of the video: " + run.stderr.decode(errors="replace")[-300:])
    return target


def _frame_scale(info: dict) -> str:
    if info["display_width"] >= info["display_height"]:
        return f"scale='min(iw,{FRAME_EDGE})':-2"
    return f"scale=-2:'min(ih,{FRAME_EDGE})'"


def extract_media(video: bytes, info: dict | None = None) -> dict:
    """Audio and frames for the agent.

    Returns ``{"wav", "duration", "frames": [{"time", "image", "width", "height"}], "info",
    "trimmed", "full_duration"}``. Frames are decoded with rotation applied, so their
    size is the displayed size. Clips over 90 seconds are checked for their first 90.
    """
    with tempfile.TemporaryDirectory(prefix="claimcheck-") as tmp:
        root = Path(tmp)
        source = root / "upload"
        source.write_bytes(video)
        info = info or probe(source)
        full = info["duration"]
        duration = min(full, MAX_CHECK_SECONDS) if full else MAX_CHECK_SECONDS
        wav = root / "audio.wav"
        audio = None
        if info["has_audio"]:
            audio = subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-i", str(source), "-t", f"{duration:.2f}", "-vn", "-ac", "1", "-ar", "16000",
                 "-sample_fmt", "s16", str(wav)], capture_output=True)
        frames = []
        for index, at in enumerate(frame_times(duration)):
            target = root / f"frame-{index}.jpg"
            run = subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-ss", f"{at:.2f}", "-i", str(source), "-frames:v", "1",
                 "-vf", _frame_scale(info), "-q:v", "4", str(target)],
                capture_output=True,
            )
            if run.returncode == 0 and target.exists():
                size = probe(target)
                frames.append({"time": at, "image": base64.b64encode(target.read_bytes()).decode("ascii"),
                               "width": size["width"], "height": size["height"]})
        return {
            "wav": wav.read_bytes() if audio is not None and audio.returncode == 0 and wav.exists() else b"",
            "duration": round(duration, 2),
            "full_duration": full,
            "trimmed": bool(full and full > MAX_CHECK_SECONDS),
            "frames": frames,
            "info": info,
        }


def timeline_art(path: Path, duration: float) -> dict:
    """A frame strip (one thumbnail per second, max 30) and an audio waveform for the editor ruler.

    Both are cheap ffmpeg filters. Returns base64 images, or {} when ffmpeg fails.
    """
    duration = max(1.0, float(duration))
    count = max(1, min(30, int(round(duration))))
    strip = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"fps={count / duration:.4f},scale=-2:54,tile={count}x1",
         "-frames:v", "1", "-q:v", "5", "-f", "image2pipe", "-vcodec", "mjpeg", "-"], capture_output=True)
    wave = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-filter_complex", "aformat=channel_layouts=mono,showwavespic=s=1200x60:colors=0x6b6b6b",
         "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True)
    art = {}
    if strip.returncode == 0 and strip.stdout:
        art["strip"] = base64.b64encode(strip.stdout).decode("ascii")
        art["strip_count"] = count
    if wave.returncode == 0 and wave.stdout:
        art["wave"] = base64.b64encode(wave.stdout).decode("ascii")
    return art


IMAGE_EDGE = 1024


def image_frame(content: bytes, number: int, folder: Path) -> dict:
    """One post image as a frame: EXIF orientation applied, longest edge at most 1024, JPEG."""
    source = folder / f"image-{number}.in"
    target = folder / f"image-{number}.jpg"
    source.write_bytes(content)
    info = probe(source)
    scale = f"scale='min(iw,{IMAGE_EDGE})':-2" if info["display_width"] >= info["display_height"] else f"scale=-2:'min(ih,{IMAGE_EDGE})'"
    run = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(source), "-frames:v", "1", "-vf", scale, "-q:v", "3", str(target)],
                         capture_output=True)
    if run.returncode != 0 or not target.exists():
        raise MediaError(f"Could not read image {number}. Use JPG or PNG.")
    size = probe(target)
    source.unlink(missing_ok=True)
    return {"time": float(number), "kind": "image", "image": base64.b64encode(target.read_bytes()).decode("ascii"),
            "width": size["width"], "height": size["height"]}
