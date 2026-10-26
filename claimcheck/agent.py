"""Plan, act, and observe review workflow."""
import re
import uuid

from .packs import pack_guidance
from .providers import get_provider
from .tools import search_source


def extract_claims(transcript):
    return [{"quote": match.group().strip(), "timestamp": round(match.start() / 12, 1), "rounds": []} for match in re.finditer(r"[^.!?]+[.!?]?", transcript) if re.search(r"\b(guarantee|guarantees|cure|treat|prevents|eliminates|100%|zero)\b", match.group(), re.I)]


def run_review(video, sources, messages, pack, provider=None, emit=None):
    provider = provider or get_provider()
    trace = []

    def log(tool, inputs, output, mode=None):
        if len(trace) >= 12:
            return
        event = {"step": len(trace) + 1, "tool": tool, "input": inputs, "output": output}
        if mode:
            event["mode"] = mode
        trace.append(event)
        if emit:
            emit(event)

    transcript = provider.transcribe(video)
    log("transcribe", {"video_bytes": len(video)}, transcript)
    frames = provider.frames(video)
    public_frames = [{key: value for key, value in frame.items() if key != "image"} for frame in frames]
    log("vision_frames", {"sampled_frames": len(frames)}, public_frames)
    guidance = pack_guidance(pack)
    if pack.lower() == "fda beauty":
        log("FDA_rule_pack", {"pack": pack}, guidance)
    claims = extract_claims(transcript)
    log("extract_claims", {}, claims)
    for claim in claims:
        evidence = search_source(claim["quote"], sources + "\n" + guidance)
        log("search_source", {"query": claim["quote"]}, evidence)
        verdict = provider.judge_claim(claim["quote"], evidence)
        log("judge_claim", {"claim": claim["quote"]}, verdict)
        claim["rounds"].append(verdict)
        if verdict.get("verdict") == "fail":
            revised = provider.rewrite_line(claim["quote"], verdict.get("reason", ""), evidence, 1)
            log("rewrite_line", {"claim": claim["quote"], "failure_reason": verdict.get("reason")}, revised)
            retry = provider.judge_claim(revised, evidence)
            log("judge_claim", {"claim": revised, "round": 2, "failure_reason": verdict.get("reason")}, retry)
            claim["rounds"].append(retry)
            claim["suggested_line"] = revised
        claim["verdict"] = claim["rounds"][-1].get("verdict", "fail")
        claim["evidence"] = claim["rounds"][-1].get("evidence", "")
    disclosed = provider.check_disclosure(transcript, frames)
    log("check_disclosure", {}, disclosed)
    coverage = provider.check_coverage(transcript, messages)
    log("check_coverage", {}, coverage)
    log("finish", {}, {"claim_count": len(claims)})
    return {"id": str(uuid.uuid4()), "transcript": transcript, "claims": claims, "disclosure": disclosed, "coverage": coverage, "trace": trace, "frames": public_frames}
