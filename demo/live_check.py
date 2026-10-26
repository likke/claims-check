#!/usr/bin/env python3
"""Live end-to-end check against the Meta Model API.

Runs the full review (ffmpeg, Muse Voice Transcribe, Muse Spark agent,
vision, guard) on demo/live-check.mp4, prints each trace event as it
happens, saves the result JSON and checks six expected outcomes.

    set -a; . /path/to/meta-model-api.env; set +a
    python demo/live_check.py --out demo/live-run-2026-10-27.json

A call budget (default 26 Meta calls) stops the run before credits run away.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from claimcheck.agent import run_review  # noqa: E402
from claimcheck.media import extract_media  # noqa: E402
from claimcheck.packs import get_pack  # noqa: E402
from claimcheck.providers import MetaProvider, ProviderError  # noqa: E402

SOURCES = ("Dasho checks creator content before it posts. Each post gets a draft first and a named human approver. "
           "Dasho flags possible claim issues for a human reviewer; it does not guarantee regulatory compliance.")
KEY_MESSAGES = ["Named approver before posting"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=str(ROOT / "demo" / "live-check.mp4"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--budget", type=int, default=int(os.getenv("LIVE_CALL_BUDGET", "26")))
    parser.add_argument("--product", default="Dasho")
    parser.add_argument("--pack", default="Tech claims")
    parser.add_argument("--sources-file", help="text file with the source pack (default: the built-in Dasho pack)")
    parser.add_argument("--key-message", action="append", help="repeat for each key message")
    parser.add_argument("--update-replay", action="store_true", help="point demo/replay.json at this run and video")
    args = parser.parse_args()

    provider = MetaProvider()
    original = provider._responses

    def budgeted(payload, task):
        if len(provider.calls) >= args.budget:
            raise ProviderError(f"live call budget of {args.budget} reached")
        return original(payload, task)

    provider._responses = budgeted

    def show(event):
        who = event.get("by") or ""
        line = f"[{event['t']:6.2f}s] {event['kind']:<16} {who:<6} {event.get('tool', ''):<16} {event.get('summary', '')}"
        if event["kind"] == "plan":
            line += f"  <- {event['mode']}: {event.get('reasoning', '')[:160]}"
        print(line[:400], flush=True)

    media = extract_media(Path(args.video).read_bytes())
    print(f"video {media['duration']} s, frames at {[f['time'] for f in media['frames']]}")
    sources = Path(args.sources_file).read_text(encoding="utf-8") if args.sources_file else SOURCES
    result = run_review(provider, wav=media["wav"], frames=media["frames"], sources=sources, key_messages=args.key_message or KEY_MESSAGES,
                        pack=get_pack(args.pack), product=args.product, emit=show)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if provider.api_key in text:
        raise SystemExit("refusing to save: the API key appears in the result")
    Path(args.out).write_text(text, encoding="utf-8")
    if args.update_replay:
        relative = lambda path: str(Path(path).resolve().relative_to(ROOT)) if Path(path).resolve().is_relative_to(ROOT) else str(Path(path).resolve())
        (ROOT / "demo" / "replay.json").write_text(json.dumps({"run": relative(args.out), "clip": relative(args.video)}, indent=2) + "\n")
        print("demo/replay.json now points at this run and clip")
    print(f"\nsaved {args.out}; usage {result['usage']}; agent {result['agent']}")
    for claim in result["claims"]:
        print(f"{claim['id']} {claim['status']:<6} \"{claim['quote']}\" -> {claim['suggested_line']}  evidence={claim['evidence']}")
    print("disclosure", json.dumps({k: result["disclosure"].get(k) for k in ("disclosed", "spoken", "on_screen")}))
    print("coverage", json.dumps(result["coverage"]))


if __name__ == "__main__":
    main()
