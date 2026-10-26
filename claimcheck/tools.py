"""The eight agent tools.

Model-backed tools (extract_claims, judge_claim, rewrite_line,
check_disclosure, check_coverage) call Muse Spark, then code checks every
answer against the inputs: quotes must exist in the transcript, citations
must exist in the cited chunk, frame times must be frames we sampled, and a
rewrite may not delete the claim. search_source and fetch_source are code.
"""
from __future__ import annotations

import re

from . import prompts
from .packs import hard_rule_hits
from .sources import Chunk, fetch_url
from .state import MAX_REWRITES, Claim, ReviewState
from .text import TAGALOG_MARKERS, containment, content_terms, has_tagalog, is_taglish, locate, normalise, quote_in, quote_span, stem, tokens

DISCLOSURE = re.compile(
    r"(paid partnership|sponsored|bayad na partnership|may bayad|in partnership with|partnership with|#ad\b|\bad:|binayaran|#bayad)",
    re.I,
)
PLEASANTRY = re.compile(
    r"^\W*(thanks?( you)?( so much)? for watching|like and subscribe|follow (me|us)( for more)?|salamat( po)?|see you( next time)?|bye|check it out|link in bio)\W*$",
    re.I,
)

TOOL_SPECS = [
    {"name": "extract_claims", "description": "Read the timestamped speech and the sampled frames and list every factual product claim (spoken or on screen). Run once.",
     "parameters": {"type": "object", "properties": {}, "required": []}},
    {"name": "search_source", "description": "Search the brand's source pack for chunks about a topic. Optional; pass claim_id to attach the hits to that claim's evidence.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "claim_id": {"type": "string"}}, "required": ["query"]}},
    {"name": "fetch_source", "description": "Read one URL listed in the source pack and add its text to the index.",
     "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}},
    {"name": "judge_claim", "description": "Verify a claim (target original) or its latest rewrite (target rewrite) against source chunks. Returns pass or fail with the cited chunk.",
     "parameters": {"type": "object", "properties": {"claim_id": {"type": "string"}, "target": {"type": "string", "enum": ["original", "rewrite"]},
                                                     "chunk_ids": {"type": "array", "items": {"type": "string"}}}, "required": ["claim_id", "target"]}},
    {"name": "rewrite_line", "description": f"Propose a corrected line for a failed claim in the creator's voice. Round 2 receives round 1's failure reason. Max {MAX_REWRITES} per claim. Judge each rewrite afterwards.",
     "parameters": {"type": "object", "properties": {"claim_id": {"type": "string"}}, "required": ["claim_id"]}},
    {"name": "check_disclosure", "description": "Check for a paid-partnership disclosure, spoken (English or Tagalog) or on screen, plus any required disclaimer for the rule pack.",
     "parameters": {"type": "object", "properties": {}, "required": []}},
    {"name": "check_coverage", "description": "Check each key message from the brief: covered, partly or missing, with the quote.",
     "parameters": {"type": "object", "properties": {}, "required": []}},
    {"name": "finish", "description": "End the review with a one-sentence summary.",
     "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}},
]
TOOL_NAMES = [t["name"] for t in TOOL_SPECS]


# Function words that are Cebuano (Bisaya) and not Tagalog.
CEBUANO_MARKERS = {"nga", "ug", "kaayo", "jud", "gyud", "dili", "unsa", "karon", "lagi", "bitaw", "ganahan", "nindot", "imong",
                   "akong", "naa", "wala'y", "kini", "kana", "niini", "pud", "pod", "bai", "bay", "gani", "diay"}
TAGALOG_ONLY = {"ng", "mga", "hindi", "talaga", "kasi", "yung", "naman", "lang", "ito", "iyan", "ako", "sobrang", "grabe", "po"}
VERIFIED_LANGUAGES = {"english", "tagalog", "taglish"}


def detect_language(state, model_language: str) -> dict:
    """Is the clip in a language the speech model is evaluated for (Tagalog, English, Taglish)?

    Code counts Cebuano-only and Tagalog-only function words; the model's own
    reading is the second signal. Either one saying Cebuano or another
    language means the claims need a person.
    """
    words = tokens(state.transcript + " " + state.caption)
    cebuano = {w for w in words if w in CEBUANO_MARKERS}
    tagalog = {w for w in words if w in TAGALOG_ONLY}
    model_language = model_language.strip().lower()
    code_says = "cebuano" if len(cebuano) >= 2 and len(cebuano) > len(tagalog) else ""
    detected = code_says or model_language or ("taglish" if tagalog else "english")
    verified = not code_says and (not model_language or model_language in VERIFIED_LANGUAGES)
    how = []
    if code_says:
        how.append("Cebuano words: " + ", ".join(sorted(cebuano)[:5]))
    if model_language:
        how.append(f"model reads {model_language}")
    return {"detected": detected, "verified": verified, "how": "; ".join(how) or "default"}


class ToolError(Exception):
    """A tool refused the call. The message goes back to the agent as its observation."""


def _snap_frame(state: ReviewState, at) -> dict | None:
    try:
        at = float(at)
    except (TypeError, ValueError):
        return None
    best = min(state.frames, key=lambda f: abs(f["time"] - at), default=None)
    if best is None or abs(best["time"] - at) > 0.6:
        return None
    return best


def _valid_box(box):
    """A model-drawn box [left, top, width, height] in frame fractions, or None if it is not a sane box."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        left, top, width, height = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    if min(left, top) < 0 or width <= 0.01 or height <= 0.01 or left + width > 1.001 or top + height > 1.001:
        return None
    return {"x": round(left, 4), "y": round(top, 4), "w": round(width, 4), "h": round(height, 4), "source": "Muse Spark vision"}


def _images(state: ReviewState) -> list[dict]:
    return [{"time": f["time"], "image": f["image"]} for f in state.frames if f.get("image")]


def _frames_for_context(state: ReviewState) -> list[dict]:
    return [{"time": f["time"], "text": f.get("text", "")} for f in state.frames]


def rewrite_floor(original: str, line: str, pack) -> list[str]:
    """Code checks a rewrite must pass before the judge sees it."""
    problems = []
    if not line.strip():
        return ["The rewrite is empty, which deletes the claim."]
    if PLEASANTRY.match(line.strip()):
        problems.append("The rewrite replaces the claim with a pleasantry, which deletes the claim.")
    original_words, new_words = len(tokens(original)), len(tokens(line))
    if new_words < max(3, round(0.4 * original_words)):
        problems.append(f"The rewrite has {new_words} words against {original_words}; it is too short to keep the creator's point.")
    if normalise(line).strip(" .!") == normalise(original).strip(" .!"):
        problems.append("The rewrite repeats the failed line.")
    filler = {stem(word) for word in TAGALOG_MARKERS} | {"grabe", "guys", "uy", "really", "so", "super", "talaga"}
    topic = set(content_terms(original)) - filler
    if len(topic) >= 2 and not topic & set(content_terms(line)):
        words = []
        for word in tokens(original):
            if stem(word) in topic and word not in words:
                words.append(word)
        problems.append("The rewrite drops the creator's topic: it shares no key word with the original line ("
                        + ", ".join(words[:6]) + ").")
    if is_taglish(original) and not has_tagalog(line):
        problems.append("The creator's line is Taglish but the rewrite is English only.")
    for rule in hard_rule_hits(pack, line):
        problems.append(rule.reason)
    return problems


class ToolBox:
    def __init__(self, state: ReviewState, provider, fetcher=fetch_url):
        self.state = state
        self.provider = provider
        self.fetcher = fetcher

    def run(self, name: str, args: dict | None) -> tuple[dict, str]:
        args = args if isinstance(args, dict) else {}
        handler = getattr(self, "tool_" + str(name), None)
        if name not in TOOL_NAMES or handler is None:
            return {"error": f"unknown tool {name!r}; tools are {', '.join(TOOL_NAMES)}"}, f"Unknown tool {name}"
        try:
            return handler(**{k: v for k, v in args.items() if k in handler.__code__.co_varnames})
        except ToolError as refused:
            return {"error": str(refused)}, f"Refused: {refused}"

    # ---- source tools -------------------------------------------------
    def tool_search_source(self, query: str = "", claim_id: str | None = None):
        if not str(query).strip():
            raise ToolError("search_source needs a query")
        hits = self.state.sources.search(str(query), k=4)
        claim = self.state.claim(claim_id) if claim_id else None
        if claim:
            for chunk, _ in hits:
                if chunk.id not in claim.searched:
                    claim.searched.append(chunk.id)
        observation = {"hits": [{"id": c.id, "score": s, "text": c.text, "origin": c.origin} for c, s in hits]}
        if claim:
            observation["attached_to"] = claim.id
        return observation, f"{len(hits)} chunks for \"{query}\": " + ", ".join(c.id for c, _ in hits)

    def tool_fetch_source(self, url: str = ""):
        url = str(url).strip()
        sources = self.state.sources
        if url not in sources.pending_urls:
            raise ToolError("only URLs listed in the source pack can be fetched")
        if url in sources.fetched_urls:
            raise ToolError(f"{url} was already read ({sources.fetched_urls[url]})")
        try:
            text = self.fetcher(url)
        except Exception as error:  # network, HTTP, refused host
            sources.fetched_urls[url] = f"error: {error}"
            raise ToolError(f"could not read {url}: {error}") from None
        added = sources.add_document(url, text)
        sources.fetched_urls[url] = f"ok, {len(added)} chunks"
        return ({"url": url, "chunks_added": [c.id for c in added], "first": added[0].text if added else ""},
                f"Read {url}: {len(added)} chunks added ({', '.join(c.id for c in added[:6])})")

    # ---- claims --------------------------------------------------------
    def tool_extract_claims(self):
        state = self.state
        if state.claims_extracted:
            raise ToolError(f"claims were already extracted ({len(state.claims)} claims)")
        prompt = prompts.extract_claims(state.pack, state.product, state.sentences, state.frames, state.caption)
        data = self.provider.complete_json(
            "extract_claims", prompt, images=_images(state),
            context={"sentences": [s.as_dict() for s in state.sentences], "frames": _frames_for_context(state), "product": state.product,
                     "caption": state.caption},
        )
        rejected = []
        seen = set()
        for item in data.get("claims") or []:
            quote = str(item.get("quote", "")).strip()
            index = item.get("sentence")
            matched = []
            if isinstance(index, int) and 0 <= index < len(state.sentences) and containment(quote, state.sentences[index].text) >= 0.8:
                matched = [state.sentences[index]]
            elif quote:
                matched, _ = locate(quote, state.sentences)
            if not matched:
                rejected.append({"quote": quote, "why": "quote not found in the transcript"})
                continue
            key = normalise(quote)
            if key in seen:
                continue
            seen.add(key)
            first, last = matched[0], matched[-1]
            around = [s.text for s in state.sentences if s.index in (first.index - 1, last.index + 1)]
            start, end, estimated = quote_span(quote, matched)
            state.claims.append(Claim(
                id=f"C{len(state.claims) + 1}", quote=quote, start=start, end=end,
                estimated=estimated or state.timing == "estimated", origin="speech",
                kind=str(item.get("kind", "other")), sentence_index=first.index, context=" ".join(around),
            ))
        for item in data.get("on_screen_claims") or []:
            text = str(item.get("text", "")).strip()
            frame = _snap_frame(state, item.get("frame_time"))
            if not text or frame is None:
                rejected.append({"quote": text, "why": "frame time is not one of the sampled frames"})
                continue
            if normalise(text) in seen:
                continue
            seen.add(normalise(text))
            state.claims.append(Claim(
                id=f"C{len(state.claims) + 1}", quote=text, start=frame["time"], end=frame["time"], estimated=False,
                origin="on_screen", kind=str(item.get("kind", "other")), context="On-screen text",
            ))
        for item in data.get("caption_claims") or []:
            text = str(item.get("quote", "")).strip()
            if not text or not quote_in(text, state.caption):
                rejected.append({"quote": text, "why": "quote not found in the post caption"})
                continue
            if normalise(text) in seen:
                continue
            seen.add(normalise(text))
            state.claims.append(Claim(
                id=f"C{len(state.claims) + 1}", quote=text, start=None, end=None, estimated=False,
                origin="caption", kind=str(item.get("kind", "other")), context="Post caption: " + state.caption[:300],
            ))
        state.language = detect_language(state, str(data.get("language", "")))
        if not state.language["verified"]:
            for claim in state.claims:
                claim.needs_human_review = True
        state.claims_extracted = True
        observation = {"language": state.language, "claims": [{"id": c.id, "quote": c.quote, "origin": c.origin, "start": None if c.start is None else round(c.start, 2),
                                   "estimated": c.estimated, "needs_human_review": c.needs_human_review} for c in state.claims], "rejected": rejected}
        summary = f"{len(state.claims)} claims: " + "; ".join(f"{c.id} \"{c.quote}\"" for c in state.claims)
        if rejected:
            summary += f" ({len(rejected)} model quotes rejected: not in transcript or frames)"
        return observation, summary

    def _evidence(self, claim: Claim, line: str, chunk_ids, extra: list[str] = ()) -> list[Chunk]:
        sources = self.state.sources
        wanted: list[str] = []
        for cid in list(chunk_ids or []) + list(claim.searched) + list(extra):
            if sources.get(str(cid)) and str(cid) not in wanted:
                wanted.append(str(cid))
        for chunk, _ in sources.search(line, k=4, include_rules=False):
            if chunk.id not in wanted:
                wanted.append(chunk.id)
        if claim.quote != line:
            for chunk, _ in sources.search(claim.quote, k=2, include_rules=False):
                if chunk.id not in wanted:
                    wanted.append(chunk.id)
        wanted += [c.id for c in sources.chunks if c.id.startswith("R") and c.id not in wanted]
        return [sources.get(cid) for cid in wanted[:10]]

    def tool_judge_claim(self, claim_id: str = "", target: str = "original", chunk_ids=None):
        state = self.state
        claim = state.claim(claim_id)
        if claim is None:
            raise ToolError(f"no claim {claim_id!r}; claims are {', '.join(c.id for c in state.claims) or 'not extracted yet'}")
        target = "rewrite" if str(target).lower().startswith("rewrite") else "original"
        if target == "original":
            if claim.rounds:
                raise ToolError(f"{claim.id} original was already judged ({claim.rounds[0]['verdict']})")
            line, extra = claim.quote, []
        else:
            if claim.needs() != "judge_rewrite":
                raise ToolError(f"{claim.id} has no unjudged rewrite (status {claim.status}, next step {claim.needs() or 'none'})")
            line, extra = claim.rounds[-1]["line"], claim.rounds[-1].get("cited_chunks", []) + [claim.rounds[0].get("chunk_id") or ""]
        chunks = self._evidence(claim, line, chunk_ids if isinstance(chunk_ids, list) else [], extra)
        original_reason = claim.rounds[0].get("reason") if claim.rounds else None
        prompt = prompts.judge_claim(state.pack, line, chunks, target, claim.quote, original_reason, state.product, claim.context)
        data = self.provider.complete_json(
            "judge_claim", prompt,
            context={"line": line, "original": claim.quote, "target": target, "chunks": [c.as_dict() for c in chunks], "pack": state.pack.key},
        )
        result = self._validate_verdict(data, chunks, line, target)
        if target == "original":
            claim.rounds.append({"round": 0, "line": line, **result})
        else:
            claim.rounds[-1].update(result)
        claim.refresh_status()
        label = "rewrite " + str(claim.rounds[-1]["round"]) if target == "rewrite" else "original"
        observation = {"claim_id": claim.id, "target": label, "verdict": result["verdict"], "judgement": result["judgement"],
                       "reason": result["reason"], "chunk_id": result["chunk_id"], "evidence": result["evidence"],
                       "evidence_origin": result.get("evidence_origin", ""), "line": line, "keeps_point": result.get("keeps_point"),
                       "claim_status": claim.status, "suggested_line": claim.suggested_line, "next": claim.needs()}
        cite = f" · cites {result['chunk_id']}" if result["chunk_id"] else ""
        return observation, f"{claim.id} {label}: {result['verdict'].upper()} ({result['judgement']}){cite}. {result['reason']}"

    def _validate_verdict(self, data: dict, chunks: list[Chunk], line: str, target: str) -> dict:
        verdict_word = str(data.get("verdict", "")).strip().lower().replace(" ", "_").replace("-", "_")
        reason = str(data.get("reason", "")).strip() or "No reason given."
        by_id = {c.id: c for c in chunks}
        chunk = by_id.get(str(data.get("chunk_id", "")).strip().strip("[]"))
        quote = str(data.get("evidence_quote", "")).strip()
        checks = []
        verified = bool(chunk and quote and quote_in(quote, chunk.text))
        if verdict_word not in ("supported", "contradicted", "not_supported"):
            checks.append(f"judge returned an unknown verdict {verdict_word!r}; treated as not_supported")
            verdict_word = "not_supported"
        if verdict_word == "supported":
            if chunk is None or chunk.id.startswith("R"):
                checks.append("judge said supported but cited no product source chunk; downgraded")
                verdict_word, reason = "not_supported", "No product source was cited for this line, so it is treated as unsupported."
            elif not verified:
                checks.append(f"cited words were not found in {chunk.id}; downgraded")
                verdict_word, reason = "not_supported", "The quoted evidence is not in the cited product source, so the claim is treated as unsupported."
        result = {
            "verdict": "pass" if verdict_word == "supported" else "fail",
            "judgement": verdict_word,
            "reason": reason,
            "chunk_id": chunk.id if chunk else None,
            "evidence": (quote if verified else chunk.text) if chunk else "",
            "evidence_origin": chunk.origin if chunk else "",
            "citation_verified": verified,
            "checks": checks,
        }
        if target == "rewrite":
            keeps = data.get("keeps_point")
            result["keeps_point"] = bool(keeps) if keeps is not None else None
            if result["verdict"] == "pass" and keeps is False:
                result.update(verdict="fail", judgement="lost_point",
                              reason="The rewrite no longer makes the creator's point. " + str(data.get("keeps_point_reason", "")).strip())
        for rule in hard_rule_hits(self.state.pack, line):
            rule_chunk = self.state.sources.get(rule.rule_chunk)
            result.update(verdict="fail", judgement="rule", reason=rule.reason, chunk_id=rule.rule_chunk,
                          evidence=rule_chunk.text if rule_chunk else "", evidence_origin=rule_chunk.origin if rule_chunk else "",
                          citation_verified=True)
            result["checks"] = result["checks"] + [f"hard rule {rule.id} matched"]
        return result

    def tool_rewrite_line(self, claim_id: str = ""):
        state = self.state
        claim = state.claim(claim_id)
        if claim is None:
            raise ToolError(f"no claim {claim_id!r}")
        need = claim.needs()
        if need != "rewrite":
            why = {"judge": "judge the original first", "judge_rewrite": "judge the latest rewrite first",
                   None: "it passed" if claim.status in ("pass", "fixed") else f"the limit of {MAX_REWRITES} rewrites is reached"}[need]
            raise ToolError(f"{claim.id} cannot be rewritten now: {why}")
        number = len(claim.rewrites()) + 1
        first = claim.rounds[0]
        previous = claim.rounds[-1] if number > 1 else None
        evidence_chunk = state.sources.get(first.get("chunk_id") or "")
        extra = [first.get("chunk_id") or ""] + (previous.get("cited_chunks", []) if previous else [])
        chunks = self._evidence(claim, claim.quote, [], extra)
        reason = first.get("reason", "")
        prompt = prompts.rewrite_line(state.pack, claim, reason, evidence_chunk, chunks, previous)
        data = self.provider.complete_json(
            "rewrite_line", prompt,
            context={"claim": claim.as_dict(), "reason": reason, "round": number, "previous": previous,
                     "chunks": [c.as_dict() for c in chunks], "pack": state.pack.key},
        )
        line = str(data.get("line", "")).strip().strip('"')
        cited = [str(c) for c in (data.get("chunk_ids") or []) if state.sources.get(str(c))]
        round_ = {"round": number, "line": line, "verdict": "pending", "cited_chunks": cited,
                  "what_changed": str(data.get("what_changed", "")), "previous_failure": previous.get("reason") if previous else reason}
        problems = rewrite_floor(claim.quote, line, state.pack)
        if problems:
            round_.update(verdict="fail", judgement="rejected_by_code", reason=" ".join(problems), chunk_id=None, evidence="")
        claim.rounds.append(round_)
        claim.refresh_status()
        observation = {"claim_id": claim.id, "round": number, "line": line, "cited_chunks": cited,
                       "code_check": "rejected: " + " ".join(problems) if problems else "passed", "next": claim.needs()}
        summary = f"{claim.id} rewrite {number}: \"{line}\"" + (f" — rejected by code: {' '.join(problems)}" if problems else "")
        return observation, summary

    # ---- disclosure and coverage --------------------------------------
    def tool_check_disclosure(self):
        state = self.state
        if state.disclosure is not None:
            raise ToolError("disclosure was already checked")
        disclaimers = [d for d in state.pack.disclaimers if d.applies_to != "supplement" or state.supplement]
        label = disclaimers[0].label if disclaimers else None
        prompt = prompts.check_disclosure(state.sentences, state.frames, label, state.caption)
        data = self.provider.complete_json(
            "check_disclosure", prompt, images=_images(state),
            context={"sentences": [s.as_dict() for s in state.sentences], "frames": _frames_for_context(state), "disclaimer": label,
                     "caption": state.caption},
        )
        notes = []
        # Frame text read by the model, pinned to the frames we actually sampled.
        frame_text = []
        for item in data.get("frame_text") or []:
            frame = _snap_frame(state, item.get("frame_time"))
            if frame is not None:
                frame["text"] = str(item.get("text", "")).strip()
                frame_text.append({"frame_time": frame["time"], "text": frame["text"]})
        if not frame_text:
            frame_text = [{"frame_time": f["time"], "text": f.get("text", "")} for f in state.frames if f.get("text")]
        # Spoken disclosure: the model's quote must be in the transcript.
        spoken = {"found": False, "quote": "", "time": None, "how": "not found"}
        model_spoken = data.get("spoken") or {}
        if model_spoken.get("found"):
            quote = str(model_spoken.get("quote", ""))
            matched, _ = locate(quote, state.sentences)
            if matched and quote.strip():
                spoken = {"found": True, "quote": quote, "time": quote_span(quote, matched)[0], "how": "model, quote verified in transcript"}
            else:
                notes.append("model reported a spoken disclosure whose quote is not in the transcript; ignored")
        if not spoken["found"]:
            for sentence in state.sentences:
                if DISCLOSURE.search(sentence.text):
                    spoken = {"found": True, "quote": sentence.text, "time": round(sentence.start, 2), "how": "phrase match in transcript"}
                    break
        # On-screen disclosure: must name a sampled frame.
        on_screen = {"found": False, "text": "", "time": None, "how": "not found"}
        model_screen = data.get("on_screen") or {}
        if model_screen.get("found"):
            frame = _snap_frame(state, model_screen.get("frame_time"))
            text = str(model_screen.get("text", "")).strip()
            if frame is not None and text:
                on_screen = {"found": True, "text": text, "time": frame["time"], "how": "model vision, frame time verified",
                             "box": _valid_box(model_screen.get("box"))}
            else:
                notes.append("model reported an on-screen disclosure without a sampled frame time; ignored")
        if not on_screen["found"]:
            for item in frame_text:
                if DISCLOSURE.search(item["text"]):
                    on_screen = {"found": True, "text": item["text"], "time": item["frame_time"], "how": "phrase match in frame text"}
                    break
        # Caption disclosure: the model's quote must be in the caption, or a known phrase must match.
        caption = {"found": False, "quote": "", "how": "not found" if state.caption.strip() else "no caption given"}
        model_caption = data.get("caption") or {}
        if model_caption.get("found") and quote_in(str(model_caption.get("quote", "")), state.caption):
            caption = {"found": True, "quote": str(model_caption["quote"]), "how": "model, quote verified in caption"}
        elif state.caption.strip():
            match = DISCLOSURE.search(state.caption)
            if match:
                caption = {"found": True, "quote": match.group(0), "how": "phrase match in caption"}
        result = {"disclosed": spoken["found"] or on_screen["found"] or caption["found"], "spoken": spoken, "on_screen": on_screen,
                  "caption": caption, "frame_text": frame_text, "notes": notes, "required_disclaimer": None}
        if disclaimers:
            result["required_disclaimer"] = self._disclaimer(disclaimers[0], data.get("disclaimer") or {}, frame_text)
        state.disclosure = result
        parts = []
        if spoken["found"]:
            parts.append(f"spoken at {spoken['time']}s (\"{spoken['quote']}\")")
        if on_screen["found"]:
            parts.append(f"on screen at {on_screen['time']}s (\"{on_screen['text']}\")")
        if caption["found"]:
            parts.append(f"in the caption (\"{caption['quote']}\")")
        summary = "Disclosure " + ("found: " + "; ".join(parts) if parts else "MISSING")
        if result["required_disclaimer"]:
            summary += f". Required line \"{result['required_disclaimer']['label']}\": " + ("found" if result["required_disclaimer"]["found"] else "MISSING")
        return {k: v for k, v in result.items() if k != "frame_text"} | {"frames_read": len(frame_text)}, summary

    def _disclaimer(self, disclaimer, model: dict, frame_text: list[dict]) -> dict:
        state = self.state
        for sentence in state.sentences:
            if any(p.search(sentence.text) for p in disclaimer.patterns):
                return {"label": disclaimer.label, "found": True, "where": "spoken", "time": round(sentence.start, 2), "quote": sentence.text}
        for item in frame_text:
            if any(p.search(item["text"]) for p in disclaimer.patterns):
                return {"label": disclaimer.label, "found": True, "where": "on_screen", "time": item["frame_time"], "quote": item["text"]}
        if any(p.search(state.caption) for p in disclaimer.patterns):
            match = next(p.search(state.caption) for p in disclaimer.patterns if p.search(state.caption))
            return {"label": disclaimer.label, "found": True, "where": "caption", "time": None, "quote": match.group(0)}
        quote = str(model.get("quote", ""))
        if model.get("found") and quote and (quote_in(quote, state.transcript) or any(quote_in(quote, f["text"]) for f in frame_text)):
            return {"label": disclaimer.label, "found": True, "where": str(model.get("where", "")), "time": None, "quote": quote}
        return {"label": disclaimer.label, "found": False, "where": "none", "time": None, "quote": ""}

    def tool_check_coverage(self):
        state = self.state
        if state.coverage is not None:
            raise ToolError("coverage was already checked")
        if not state.key_messages:
            state.coverage = {"messages": []}
            return {"messages": []}, "No key messages in the brief"
        frame_text = [{"frame_time": f["time"], "text": f.get("text", "")} for f in state.frames]
        prompt = prompts.check_coverage(state.sentences, state.key_messages, frame_text, state.caption)
        data = self.provider.complete_json(
            "check_coverage", prompt,
            context={"sentences": [s.as_dict() for s in state.sentences], "messages": state.key_messages, "frame_text": frame_text},
        )
        answers = data.get("messages") or []
        results = []
        for i, message in enumerate(state.key_messages):
            answer = next((a for a in answers if normalise(str(a.get("message", ""))) == normalise(message)), None)
            if answer is None and i < len(answers):
                answer = answers[i]
            answer = answer or {}
            status = str(answer.get("status", "missing")).lower()
            status = status if status in ("covered", "partly", "missing") else "missing"
            quote = str(answer.get("quote", "")).strip()
            time = None
            note = ""
            if status != "missing":
                matched, _ = locate(quote, state.sentences) if quote else ([], 0)
                if matched:
                    time = quote_span(quote, matched)[0]
                elif quote and quote_in(quote, state.caption):
                    note = "said in the post caption"
                elif quote and any(quote_in(quote, f.get("text", "")) for f in state.frames):
                    frame = next(f for f in state.frames if quote_in(quote, f.get("text", "")))
                    time, note = frame["time"], "shown on screen"
                else:
                    note = "quote not found in transcript, so the status was lowered"
                    status = "partly" if status == "covered" and quote else "missing"
            results.append({"message": message, "status": status, "quote": quote, "time": time, "note": note})
        state.coverage = {"messages": results}
        summary = "; ".join(f"\"{r['message']}\": {r['status']}" for r in results)
        return {"messages": results}, summary

    def tool_finish(self, summary: str = ""):
        self.state.finished = True
        self.state.finish_summary = str(summary or "").strip()
        open_work = self.state.open_work()
        return ({"summary": self.state.finish_summary, "open_work_left_for_guard": open_work},
                "Agent finished" + (f"; {len(open_work)} open items go to the guard" if open_work else "; nothing left open"))
