"""Offline fixture: a fictional product ("Ledgerly AI") so the whole flow runs
with no API key. The fixture provider answers each model task with plain
heuristics and plans with a fixed policy. The UI labels it as offline.
"""
from __future__ import annotations

import os
import re
import time

from .providers import BaseProvider, PlanResult
from .text import content_terms, has_tagalog, is_taglish, tokens

PRODUCT = "Ledgerly AI"

SOURCE_PACK = """Ledgerly AI reads a typical invoice in under 10 seconds.
Ledgerly AI flags unusual charges for a human reviewer before any payment is approved.
Ledgerly AI connects with 12 partner banks in the Philippines.
Ledgerly AI does not guarantee fraud prevention or regulatory compliance.
fixture://ledgerly-ai/pricing"""

KEY_MESSAGES = ["Flags unusual charges for human review", "Works with Philippine banks", "14-day free trial"]

TURNS = [
    {"start": 0.0, "end": 3.6, "text": "Uy guys, bayad na partnership ito with Ledgerly AI."},
    {"start": 3.6, "end": 8.2, "text": "Ledgerly AI reads a typical invoice in under 10 seconds, sobrang bilis."},
    {"start": 8.2, "end": 13.5, "text": "Grabe, it guarantees zero fraud, one hundred percent."},
    {"start": 13.5, "end": 18.0, "text": "Every flagged charge goes to a human reviewer bago ma-approve ang payment."},
    {"start": 18.0, "end": 22.4, "text": "Connected na siya sa lahat ng bank sa Pilipinas."},
    {"start": 22.4, "end": 25.0, "text": "Try n'yo na, sulit talaga!"},
]

FRAMES = [
    {"time": 1.0, "text": "Paid partnership with Ledgerly AI"},
    {"time": 2.5, "text": ""},
    {"time": 9.0, "text": "ZERO FRAUD. 100%."},
    {"time": 14.0, "text": ""},
    {"time": 19.0, "text": ""},
    {"time": 24.5, "text": "ledgerly.example"},
]

# Scripted rewrites for the fixture claims, keyed by a word in the original.
# The first fraud rewrite still overclaims on purpose, so round 2 is visible.
REWRITES = {
    "guarantees zero fraud": [
        "Grabe, it catches all fraud bago pa mangyari.",
        "Grabe, it flags unusual charges para ma-check ng human reviewer ang fraud risk.",
    ],
    "lahat ng bank": ["Connected na siya sa 12 partner banks dito sa Pilipinas."],
    "zero fraud. 100%": ["Flags unusual charges to fight fraud"],
}

ABSOLUTE = re.compile(r"\b(guarantee[sd]?|zero|100%?|hundred percent|lahat|all|never|always)\b", re.I)
CLAIM_VERB = re.compile(
    r"\b(reads?|connect\w*|flags?|guarantee\w*|goes to|works?|cures?|treats?|heals?|prevents?|removes?|checks?|gets|catch\w*|approve\w*|zero|100)\b",
    re.I,
)
DISCLOSURE = re.compile(r"(paid partnership|sponsored|bayad na partnership|partnership with|#ad\b)", re.I)
NEGATION = re.compile(r"\b(does not|doesn't|do not|never|not|no)\b", re.I)


def _overlap(a: str, b: str) -> float:
    terms = set(content_terms(a))
    return len(terms & set(content_terms(b))) / max(1, len(terms))


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+", text.replace(",", "")))


def judge(line: str, chunks: list[dict], target: str, original: str) -> dict:
    product_chunks = [c for c in chunks if not c["id"].startswith("R")] or chunks
    best = max(product_chunks, key=lambda c: _overlap(line, c["text"]), default=None)
    if best is None:
        return {"verdict": "not_supported", "chunk_id": "", "evidence_quote": "", "reason": "The source pack is empty."}
    score = _overlap(line, best["text"])
    absolute = ABSOLUTE.findall(line)
    out = {"chunk_id": best["id"], "evidence_quote": best["text"]}
    if absolute and NEGATION.search(best["text"]) and score > 0:
        out.update(verdict="contradicted", reason=f"The source limits this: it says \"{best['text']}\"")
    elif absolute and not any(re.search(re.escape(word), best["text"], re.I) for word in absolute):
        out.update(verdict="not_supported", reason=f"The source does not back \"{absolute[0]}\"; it says \"{best['text']}\"")
    elif _numbers(line) - _numbers(best["text"]):
        out.update(verdict="not_supported", reason="A number in the line does not appear in the source.")
    elif score >= 0.3:
        out.update(verdict="supported", reason="The source states this.")
    else:
        out.update(verdict="not_supported", reason="No source chunk states this.")
    if target == "rewrite":
        out["keeps_point"] = len(tokens(line)) >= 3
        out["keeps_point_reason"] = "The line still describes the same product feature."
    return out


class FixtureProvider(BaseProvider):
    name = "fixture"
    label = "Offline fixture · Ledgerly AI (fictional) · heuristic planner"
    needs_media = False

    def __init__(self, step_delay: float | None = None):
        super().__init__()
        self.step_delay = float(os.getenv("FIXTURE_STEP_DELAY", "0.35")) if step_delay is None else step_delay

    def _pause(self):
        if self.step_delay:
            time.sleep(self.step_delay)

    def transcribe(self, wav, keywords):
        return {"turns": [dict(t) for t in TURNS], "duration": 25.0, "timing": "fixture"}

    def frames(self) -> list[dict]:
        return [dict(f) for f in FRAMES]

    def complete_json(self, task, prompt, images=None, context=None):
        started = time.monotonic()
        self._pause()
        context = context or {}
        handler = getattr(self, "_" + task)
        result = handler(context)
        self._record("fixture", task, started, True)
        return result

    def _extract_claims(self, ctx):
        claims = [{"sentence": s["index"], "quote": s["text"], "kind": "other"}
                  for s in ctx["sentences"] if CLAIM_VERB.search(s["text"]) and not DISCLOSURE.search(s["text"])]
        screen = [{"frame_time": f["time"], "text": f["text"], "kind": "guarantee"}
                  for f in ctx["frames"] if f.get("text") and CLAIM_VERB.search(f["text"]) and not DISCLOSURE.search(f["text"])]
        return {"claims": claims, "on_screen_claims": screen}

    def _judge_claim(self, ctx):
        return judge(ctx["line"], ctx["chunks"], ctx["target"], ctx["original"])

    def _rewrite_line(self, ctx):
        claim = ctx["claim"]
        number = ctx["round"]
        quote = claim["quote"].lower()
        for key, lines in REWRITES.items():
            if key in quote and number <= len(lines):
                return {"line": lines[number - 1], "chunk_ids": [], "what_changed": "Matched the claim to the source."}
        best = max((c for c in ctx["chunks"] if not c["id"].startswith("R")), key=lambda c: _overlap(claim["quote"], c["text"]), default=None)
        line = best["text"] if best else claim["quote"]
        if is_taglish(claim["quote"]) and not has_tagalog(line):
            line = line.rstrip(".") + ", talaga."
        return {"line": line, "chunk_ids": [best["id"]] if best else [], "what_changed": "Used the source wording."}

    def _check_disclosure(self, ctx):
        spoken = next((s for s in ctx["sentences"] if DISCLOSURE.search(s["text"])), None)
        screen = next((f for f in ctx["frames"] if DISCLOSURE.search(f.get("text", ""))), None)
        return {
            "spoken": {"found": bool(spoken), "sentence": spoken["index"] if spoken else None, "quote": spoken["text"] if spoken else ""},
            "on_screen": {"found": bool(screen), "frame_time": screen["time"] if screen else None, "text": screen["text"] if screen else ""},
            "frame_text": [{"frame_time": f["time"], "text": f.get("text", "")} for f in ctx["frames"]],
            "disclaimer": {"found": False, "where": "none", "quote": ""},
        }

    def _check_coverage(self, ctx):
        out = []
        for message in ctx["messages"]:
            best = max(ctx["sentences"], key=lambda s: _overlap(message, s["text"]), default=None)
            score = _overlap(message, best["text"]) if best else 0
            status = "covered" if score >= 0.5 else "partly" if score >= 0.25 else "missing"
            out.append({"message": message, "status": status, "sentence": best["index"] if best and status != "missing" else None,
                        "quote": best["text"] if best and status != "missing" else ""})
        return {"messages": out}

    def _plan(self, ctx):  # never used: the fixture plans with plan() below
        return {"actions": []}

    def plan(self, prompt, tools, context=None):
        """A fixed policy that reads the same state the model sees."""
        started = time.monotonic()
        self._pause()
        state = context["state"]
        unread = [u for u in state.sources.pending_urls if u not in state.sources.fetched_urls]
        if unread:
            actions, why = [{"tool": "fetch_source", "args": {"url": u}} for u in unread], f"The source pack lists {len(unread)} URL I have not read."
        elif not state.claims_extracted:
            actions, why = [{"tool": "extract_claims", "args": {}}], "First I need the claims from speech and frames."
        elif any(c.needs() == "judge" for c in state.claims):
            todo = [c for c in state.claims if c.needs() == "judge"][:3]
            actions, why = [{"tool": "judge_claim", "args": {"claim_id": c.id, "target": "original"}} for c in todo], f"Judging {', '.join(c.id for c in todo)} against the sources."
        elif any(c.needs() == "judge_rewrite" for c in state.claims):
            todo = [c for c in state.claims if c.needs() == "judge_rewrite"]
            actions, why = [{"tool": "judge_claim", "args": {"claim_id": c.id, "target": "rewrite"}} for c in todo], "Re-checking the proposed lines."
        elif any(c.needs() == "rewrite" for c in state.claims):
            todo = [c for c in state.claims if c.needs() == "rewrite"]
            actions, why = [{"tool": "rewrite_line", "args": {"claim_id": c.id}} for c in todo], f"{', '.join(c.id for c in todo)} failed; proposing corrected lines."
        elif state.disclosure is None or state.coverage is None:
            actions = ([{"tool": "check_disclosure", "args": {}}] if state.disclosure is None else []) + (
                [{"tool": "check_coverage", "args": {}}] if state.coverage is None else [])
            why = "Claims are settled; checking disclosure and the brief."
        else:
            fixed = sum(c.status == "fixed" for c in state.claims)
            actions, why = [{"tool": "finish", "args": {"summary": f"{len(state.claims)} claims checked, {fixed} fixed with sourced lines."}}], "All work is done."
        self._record("fixture", "plan", started, True)
        return PlanResult("fixture_planner", actions, reasoning=why)
