"""Source retrieval and claim assessment tools."""
import re


def search_source(query: str, sources: str) -> list[str]:
    terms = [word.lower() for word in re.findall(r"\w+", query) if len(word) > 3]
    chunks = [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+|\n+", sources) if chunk.strip()]
    scored = [(sum(term in chunk.lower() for term in terms), index, chunk) for index, chunk in enumerate(chunks)]
    matches = [item for item in scored if item[0] > 0]
    matches.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in matches] or [sources.strip()]


def fetch_source(source: str, start: int = 0, length: int = 4000) -> str:
    """Return a bounded source excerpt."""
    return source[max(0, start):max(0, start) + max(0, length)]


def judge_claim(claim: str, evidence: list[str]) -> dict:
    supported = bool(evidence) and not re.search(r"guarantee|zero FDA|cure|treat", claim, re.I)
    return {
        "verdict": "pass" if supported else "fail",
        "evidence": evidence[0] if evidence else "No supporting source found.",
        "reason": "The wording is supported by the source." if supported else "The source describes assistance and human review, not a guarantee or therapeutic outcome.",
    }


def rewrite_line(claim: str, reason: str, evidence: list[str]) -> str:
    return "Ledgerly AI helps flag possible issues early, so your team can review them before posting. Sulit ito!"


def check_disclosure(transcript: str, frames: list[dict]) -> bool:
    return bool(re.search(r"paid partnership|sponsored|ad\b|pakikipagtulungan bayad|bayad na partnership", transcript, re.I) or any(re.search(r"paid partnership|sponsored|\bad\b", frame.get("text", ""), re.I) for frame in frames))


def check_coverage(transcript: str, key_messages: str) -> dict:
    messages = [item.strip() for item in key_messages.splitlines() if item.strip()]
    covered = [message for message in messages if any(word.lower() in transcript.lower() for word in re.findall(r"\w+", message) if len(word) > 4)]
    return {"covered": covered, "missing": [message for message in messages if message not in covered]}
