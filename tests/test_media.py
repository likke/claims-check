"""Clip formats and orientations: vertical HEVC .mov with a rotation tag, square
WebM, landscape MP4 with no audio, and a clip over 90 seconds."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from claimcheck.agent import run_review
from claimcheck.media import extract_media, make_proxy, needs_proxy, probe
from claimcheck.packs import get_pack
from helpers import Scripted

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is not installed")


def ff(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


@pytest.fixture(scope="session")
def clips(tmp_path_factory):
    root = tmp_path_factory.mktemp("clips")
    # iPhone-style: landscape pixels, HEVC, display matrix rotated 90 degrees -> a 9:16 picture
    ff("-f", "lavfi", "-i", "testsrc2=s=640x360:d=3:r=25", "-f", "lavfi", "-i", "sine=f=440:d=3",
       "-c:v", "libx265", "-x265-params", "log-level=error", "-tag:v", "hvc1", "-c:a", "aac", "-shortest", str(root / "raw.mov"))
    ff("-display_rotation:v:0", "90", "-i", str(root / "raw.mov"), "-c", "copy", str(root / "vertical.mov"))
    ff("-f", "lavfi", "-i", "testsrc2=s=480x480:d=2:r=25", "-f", "lavfi", "-i", "sine=f=330:d=2",
       "-c:v", "libvpx-vp9", "-b:v", "300k", "-c:a", "libopus", "-shortest", str(root / "square.webm"))
    ff("-f", "lavfi", "-i", "testsrc2=s=640x360:d=2:r=25", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(root / "landscape.mp4"))
    ff("-f", "lavfi", "-i", "color=c=gray:s=160x90:d=95:r=1", "-f", "lavfi", "-i", "sine=f=220:d=95",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(root / "long.mp4"))
    return root


def test_probe_reads_orientation_rotation_and_codecs(clips):
    vertical, square, landscape = probe(clips / "vertical.mov"), probe(clips / "square.webm"), probe(clips / "landscape.mp4")
    assert (vertical["width"], vertical["height"], vertical["rotation"]) == (640, 360, 90)
    assert (vertical["display_width"], vertical["display_height"], vertical["orientation"]) == (360, 640, "vertical")
    assert vertical["video_codec"] == "hevc" and needs_proxy(vertical)
    assert square["orientation"] == "square" and square["video_codec"] == "vp9" and needs_proxy(square)
    assert landscape["orientation"] == "landscape" and not landscape["has_audio"] and not needs_proxy(landscape)


def test_proxy_is_h264_with_rotation_applied(clips, tmp_path):
    info = probe(clips / "vertical.mov")
    proxy = probe(make_proxy(clips / "vertical.mov", tmp_path / "v.mp4", info))
    assert proxy["video_codec"] == "h264" and proxy["audio_codec"] == "aac"
    assert (proxy["display_width"], proxy["display_height"]) == (360, 640) and proxy["orientation"] == "vertical"
    square = probe(make_proxy(clips / "square.webm", tmp_path / "s.mp4", probe(clips / "square.webm")))
    assert square["video_codec"] == "h264" and (square["display_width"], square["display_height"]) == (480, 480)


def test_frames_use_the_displayed_size(clips):
    media = extract_media((clips / "vertical.mov").read_bytes())
    assert media["frames"] and all((f["width"], f["height"]) == (360, 640) for f in media["frames"])
    assert media["wav"]
    silent = extract_media((clips / "landscape.mp4").read_bytes())
    assert silent["wav"] == b"" and silent["info"]["has_audio"] is False
    assert all(f["width"] > f["height"] for f in silent["frames"])


def test_long_clip_checks_the_first_90_seconds(clips):
    media = extract_media((clips / "long.mp4").read_bytes())
    assert media["trimmed"] and media["duration"] == 90 and media["full_duration"] >= 94
    assert max(f["time"] for f in media["frames"]) <= 90
    assert 2_800_000 <= len(media["wav"]) <= 2_900_000  # 90 s of 16 kHz 16-bit mono


def test_no_audio_skips_transcription_with_a_note():
    provider = Scripted([[{"tool": "finish", "args": {}}]], {
        "extract_claims": {"claims": [], "on_screen_claims": []},
        "check_disclosure": {"spoken": {"found": False}, "on_screen": {"found": False}, "frame_text": []},
        "check_coverage": {"messages": []}})
    result = run_review(provider, frames=[{"time": 1.0, "image": "aW1n"}], sources="Fictional product docs.", key_messages=[],
                        pack=get_pack("tech"), no_audio=True, duration=2.0, media={"orientation": "landscape"})
    first = result["trace"][0]
    assert first["kind"] == "transcribed" and first["summary"].startswith("No audio track")
    assert result["timing"] == "no audio" and result["duration"] == 2.0 and result["media"]["orientation"] == "landscape"


def read_events(client, review_id):
    events = []
    with client.stream("GET", f"/api/reviews/{review_id}/events") as response:
        for line in response.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


@pytest.mark.parametrize("name,layout,proxy", [("vertical.mov", "layout-vertical", True), ("square.webm", "layout-square", True),
                                               ("landscape.mp4", "layout-landscape", False)])
def test_upload_reports_layout_and_serves_a_playable_copy(clips, monkeypatch, tmp_path, name, layout, proxy):
    from claimcheck import fixture
    from claimcheck.web import app
    monkeypatch.setenv("CLAIMCHECK_PROVIDER", "fixture")
    monkeypatch.setenv("FIXTURE_STEP_DELAY", "0")
    client = TestClient(app)
    response = client.post("/api/reviews", data={"sources": fixture.SOURCE_PACK, "key_messages": "x"},
                           files={"video": (name, (clips / name).read_bytes(), "application/octet-stream")})
    assert response.status_code == 200
    review_id = response.json()["id"]
    media = next(e for e in read_events(client, review_id) if e.get("kind") == "media_info")
    assert media["layout"] == layout and media["proxy"] is proxy
    served = tmp_path / "served"
    served.write_bytes(client.get(f"/api/reviews/{review_id}/video").content)
    assert probe(served)["video_codec"] == ("h264" if proxy else probe(clips / name)["video_codec"])


def test_unsupported_format_is_refused(monkeypatch):
    from claimcheck.web import app
    monkeypatch.setenv("CLAIMCHECK_PROVIDER", "fixture")
    response = TestClient(app).post("/api/reviews", data={"sources": "docs"}, files={"video": ("clip.avi", b"RIFF0000AVI ", "video/x-msvideo")})
    assert response.status_code == 415
