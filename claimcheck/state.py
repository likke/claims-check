"""Review state shared by the agent, the tools and the guard."""
from __future__ import annotations

from dataclasses import dataclass, field

from .packs import Pack
from .sources import SourceIndex
from .text import Sentence

MAX_REWRITES = 2


@dataclass
class Claim:
    id: str
    quote: str
    start: float | None  # None for a claim from the post caption
    end: float | None
    estimated: bool
    origin: str  # "speech" or "on_screen"
    kind: str = "other"
    sentence_index: int | None = None
    context: str = ""
    rounds: list[dict] = field(default_factory=list)
    searched: list[str] = field(default_factory=list)
    status: str = "pending"  # pending, pass, fail, fixed
    needs_human_review: bool = False  # the language is not one the speech model is evaluated for
    suggested_line: str | None = None

    def rewrites(self) -> list[dict]:
        return [r for r in self.rounds if r["round"] > 0]

    def needs(self) -> str | None:
        """The next piece of work this claim still needs, or None."""
        if not self.rounds:
            return "judge"
        latest = self.rounds[-1]
        if latest["verdict"] == "pending":
            return "judge_rewrite"
        if latest["verdict"] == "pass":
            return None
        if len(self.rewrites()) < MAX_REWRITES:
            return "rewrite"
        return None

    def refresh_status(self) -> None:
        if not self.rounds:
            self.status = "pending"
            return
        if self.rounds[0]["verdict"] == "pass":
            self.status = "pass"
            self.suggested_line = None
            return
        passed = [r for r in self.rewrites() if r["verdict"] == "pass"]
        if passed:
            self.status = "fixed"
            self.suggested_line = passed[-1]["line"]
        else:
            self.status = "fail"
            self.suggested_line = None

    def as_dict(self) -> dict:
        return {
            "id": self.id, "quote": self.quote, "start": None if self.start is None else round(self.start, 2),
            "end": None if self.end is None else round(self.end, 2),
            "estimated": self.estimated, "origin": self.origin, "kind": self.kind, "status": self.status,
            "suggested_line": self.suggested_line, "rounds": self.rounds, "needs_human_review": self.needs_human_review,
            "evidence": _final_evidence(self),
        }


def _final_evidence(claim: Claim) -> dict | None:
    judged = [r for r in claim.rounds if r.get("chunk_id")]
    if not judged:
        return None
    chosen = judged[-1]
    return {"chunk_id": chosen["chunk_id"], "quote": chosen.get("evidence", ""), "origin": chosen.get("evidence_origin", "")}


@dataclass
class ReviewState:
    sentences: list[Sentence]
    duration: float
    timing: str  # "asr", "estimated" or "fixture"
    frames: list[dict]
    sources: SourceIndex
    key_messages: list[str]
    pack: Pack
    product: str = ""
    supplement: bool = False
    sources_text: str = ""
    caption: str = ""
    post_kind: str = "video"  # "video" or "images"
    language: dict | None = None
    claims: list[Claim] = field(default_factory=list)
    claims_extracted: bool = False
    disclosure: dict | None = None
    coverage: dict | None = None
    finished: bool = False
    finish_summary: str = ""

    def claim(self, claim_id: str) -> Claim | None:
        wanted = str(claim_id or "").strip().upper()
        return next((c for c in self.claims if c.id == wanted), None)

    @property
    def transcript(self) -> str:
        return " ".join(s.text for s in self.sentences)

    def open_work(self) -> list[str]:
        work = [f"fetch_source {url}" for url in self.sources.pending_urls if url not in self.sources.fetched_urls]
        if not self.claims_extracted:
            work.append("extract_claims")
        for claim in self.claims:
            need = claim.needs()
            if need == "judge":
                work.append(f"judge_claim {claim.id} original")
            elif need == "judge_rewrite":
                work.append(f"judge_claim {claim.id} rewrite")
            elif need == "rewrite":
                work.append(f"rewrite_line {claim.id} (round {len(claim.rewrites()) + 1})")
        if self.disclosure is None:
            work.append("check_disclosure")
        if self.coverage is None:
            work.append("check_coverage")
        return work

    def status_text(self, actions_used: int, cap: int) -> str:
        lines = []
        pending = [u for u in self.sources.pending_urls if u not in self.sources.fetched_urls]
        lines.append(
            f"Sources: {sum(1 for c in self.sources.chunks if not c.id.startswith('R'))} chunks indexed"
            + (f"; URLs not read yet: {', '.join(pending)}" if pending else "; no unread URLs")
        )
        lines.append(f"Speech: {len(self.sentences)} sentences, {self.duration:.1f} s, timing {self.timing}; {len(self.frames)} frames sampled")
        if self.caption.strip():
            lines.append(f"Post caption: {len(self.caption.split())} words")
        lines.append(f"Key messages: {len(self.key_messages)}; pack: {self.pack.name}" + ("; product is a food supplement" if self.supplement else ""))
        if not self.claims_extracted:
            lines.append("Claims: not extracted yet")
        elif not self.claims:
            lines.append("Claims: extracted, none found")
        for claim in self.claims:
            latest = claim.rounds[-1] if claim.rounds else None
            place = "caption" if claim.start is None else f"image {int(claim.start)}" if self.post_kind == "images" else f"{claim.origin} {claim.start:.1f}s"
            detail = f"{claim.id} [{place}] \"{claim.quote}\" status={claim.status}"
            detail += f" rewrites={len(claim.rewrites())}/{MAX_REWRITES}"
            if latest and latest["verdict"] == "fail":
                detail += f" last_failure=\"{latest.get('reason', '')}\""
            if claim.needs():
                detail += f" NEXT={claim.needs()}"
            lines.append("  " + detail)
        lines.append("Disclosure: " + ("checked" if self.disclosure is not None else "not checked"))
        lines.append("Coverage: " + ("checked" if self.coverage is not None else "not checked"))
        lines.append(f"Actions used: {actions_used} of {cap}")
        work = self.open_work()
        lines.append("Open work: " + ("; ".join(work) if work else "none, call finish"))
        return "\n".join(lines)
