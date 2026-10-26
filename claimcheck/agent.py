"""Plan, act, and observe review workflow."""
import uuid
import inspect
import re

from .packs import pack_guidance
from .providers import get_provider
from .tools import search_source


def extract_claims(transcript):
    """Fixture-compatible claim extraction; live providers use their model."""
    from .providers import FixtureProvider
    return FixtureProvider().extract_claims(transcript)


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

    transcription = provider.transcribe(video)
    transcript = transcription.get("text", "") if isinstance(transcription, dict) else transcription
    log("transcribe", {"video_bytes": len(video)}, transcript)
    frames = provider.frames(video)
    public_frames = [{key: value for key, value in frame.items() if key != "image"} for frame in frames]
    log("vision_frames", {"sampled_frames": len(frames)}, public_frames)
    guidance = pack_guidance(pack)
    if pack.lower() == "fda beauty":
        log("FDA_rule_pack", {"pack": pack}, guidance)
    claims = provider.extract_claims(transcription)
    log("extract_claims", {}, claims)
    for claim in claims:
        quote = claim["quote"]
        position = transcript.find(quote)
        before = transcript[:position].strip() if position >= 0 else ""
        after = transcript[position + len(quote):].strip() if position >= 0 else ""
        neighbours = " ".join(([before.split(".")[-1].strip()] if before else []) + ([after.split(".")[0].strip()] if after else []))
        evidence = search_source(claim["quote"], sources + "\n" + guidance)
        log("search_source", {"query": claim["quote"]}, evidence)
        verdict = provider.judge_claim(claim["quote"], evidence)
        log("judge_claim", {"claim": claim["quote"]}, verdict)
        claim["rounds"].append(verdict)
        if verdict.get("verdict") == "fail":
            rewrite_args = (claim["quote"], verdict.get("reason", ""), evidence, 1)
            if "neighbouring_lines" in inspect.signature(provider.rewrite_line).parameters:
                revised = provider.rewrite_line(*rewrite_args, neighbouring_lines=neighbours)
            else:
                revised = provider.rewrite_line(*rewrite_args)
            if not _keeps_product_claim(claim["quote"], revised):
                revised = ""
            log("rewrite_line", {"claim": claim["quote"], "failure_reason": verdict.get("reason")}, revised)
            if revised:
                judge_parameters = inspect.signature(provider.judge_claim).parameters
                retry = provider.judge_claim(revised, evidence, failure_reason=verdict.get("reason", "")) if "failure_reason" in judge_parameters else provider.judge_claim(revised, evidence)
            else:
                retry = {"verdict": "fail", "reason": "Rewrite removed the product claim.", "evidence": ""}
            log("judge_claim", {"claim": revised, "round": 2, "failure_reason": verdict.get("reason")}, retry)
            claim["rounds"].append(retry)
            claim["suggested_line"] = revised
        claim["verdict"] = claim["rounds"][-1].get("verdict", "fail")
        claim["evidence"] = claim["rounds"][-1].get("evidence", "")
    disclosure_result = provider.check_disclosure(transcript, frames)
    disclosed = _disclosure_details(disclosure_result, transcript, frames)
    log("check_disclosure", {}, disclosed)
    coverage = _coverage_details(provider.check_coverage(transcript, messages), messages)
    log("check_coverage", {}, coverage)
    log("finish", {}, {"claim_count": len(claims)})
    provider_label = "meta" if provider.__class__.__name__ == "MetaProvider" else "local" if provider.__class__.__name__ == "OpenAICompatibleProvider" else "fixture"
    return {"id": str(uuid.uuid4()), "transcript": transcript, "claims": claims, "disclosure": disclosed, "coverage": coverage, "trace": trace, "frames": public_frames, "provider": provider_label}


def _disclosure_details(result, transcript, frames):
    """Return a consistent disclosure finding, including its source and time."""
    if isinstance(result, dict):
        return {"disclosed": bool(result.get("disclosed")), "how": result.get("how", "speech or on-screen text"), "timestamp": result.get("timestamp")}
    if not result:
        return {"disclosed": False, "how": "not found", "timestamp": None}
    import re
    speech = re.search(r"paid partnership|sponsored|pakikipagtulungan bayad|bayad na partnership", transcript, re.I)
    if speech:
        return {"disclosed": True, "how": "spoken", "timestamp": None}
    for frame in frames:
        if re.search(r"paid partnership|sponsored|\bad\b", frame.get("text", ""), re.I):
            return {"disclosed": True, "how": "on screen", "timestamp": frame.get("time")}
    return {"disclosed": True, "how": "speech or on-screen text", "timestamp": None}


def _coverage_details(result, messages):
    """Map provider coverage output to a status and supporting quote per message."""
    if isinstance(result, dict) and "messages" in result:
        return result
    covered = result.get("covered", []) if isinstance(result, dict) else []
    missing = result.get("missing", []) if isinstance(result, dict) else messages.splitlines()
    return {"messages": [{"message": item, "status": "covered" if item in covered else "missing", "quote": item if item in covered else ""} for item in messages.splitlines() if item.strip()]}


def _keeps_product_claim(original, revised):
    """Reject empty or purely conversational rewrites that drop the product claim."""
    if not isinstance(revised, str) or not revised.strip():
        return False
    words = re.findall(r"[\w]+", revised)
    return len(words) >= 4 and bool(re.search(r"\b(help|support|flag|review|product|service|care|claim|use|work)\b", revised, re.I))
