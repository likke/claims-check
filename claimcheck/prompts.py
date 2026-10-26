"""Prompt text for each Muse Spark task. Kept in one place so it can be read and tuned."""
from __future__ import annotations

import json

from .sources import Chunk
from .text import Sentence


def speech_block(sentences: list[Sentence]) -> str:
    if not sentences:
        return "(no speech detected)"
    return "\n".join(f"[{s.index}] ({s.start:.2f}-{s.end:.2f}s) {s.text}" for s in sentences)


def frames_block(frames: list[dict]) -> str:
    if not frames:
        return "No frames are attached."
    return "Frames are attached in this order: " + ", ".join(
        f"frame {i + 1} at {f['time']:.2f}s" for i, f in enumerate(frames)
    ) + "."


def chunks_block(chunks: list[Chunk]) -> str:
    return "\n".join(f"[{c.id}] ({c.origin}) {c.text}" for c in chunks)


def extract_claims(pack, product, sentences, frames) -> str:
    return f"""You review a creator's sponsored video for a brand team before it posts.
Rule pack: {pack.name}. {pack.guidance}
Product: {product or "(not given)"}

Task: list every factual claim about the product or its results that the brand must verify: capabilities, numbers, guarantees, comparisons, integrations, health or regulatory outcomes, and process promises (for example who approves what). Skip opinions ("I love it"), personal experience ("I tried it"), greetings, calls to action, and the paid-partnership disclosure itself.

Speech, one sentence per line as [index] (start-end) text:
{speech_block(sentences)}

{frames_block(frames)} Also list product claims written as on-screen text in the frames.

Return JSON only:
{{"claims":[{{"sentence":<index>,"quote":"<words copied exactly from that sentence>","kind":"capability|number|guarantee|comparison|health|process|other"}}],
 "on_screen_claims":[{{"frame_time":<seconds>,"text":"<on-screen words copied exactly>","kind":"..."}}]}}
Use an empty list when there are none."""


def judge_claim(pack, line, chunks, target, original=None, original_reason=None) -> str:
    rewrite_part = ""
    keeps = ""
    if target == "rewrite":
        rewrite_part = f'\nThis line is a proposed rewrite of the creator\'s original line: "{original}"\nThe original failed because: {original_reason}\n'
        keeps = ',"keeps_point":true|false,"keeps_point_reason":"<one sentence>"'
    return f"""You verify advertising claims for a brand team. Decide whether the line is supported by the brand's sources. Only the numbered source chunks count. Do not use outside knowledge.
Rule pack: {pack.name}. {pack.guidance}

Line to verify: "{line}"{rewrite_part}
Source chunks:
{chunks_block(chunks)}

Verdicts:
- "supported": a chunk states what the line says. Absolute words (guarantees, 100%, zero, all, every, never) must be backed by the chunk too.
- "contradicted": a chunk says the opposite or limits the claim.
- "not_supported": no chunk states it.
{"For a rewrite, keeps_point is true only when the new line still makes the creator's product point (same topic and benefit, still a real product statement, not a deletion and not a vague pleasantry)." if target == "rewrite" else ""}
Return JSON only:
{{"verdict":"supported|contradicted|not_supported","chunk_id":"<id of the chunk you rely on>","evidence_quote":"<words copied exactly from that chunk>","reason":"<one plain sentence>"{keeps}}}"""


def rewrite_line(pack, claim, reason, evidence_chunk, chunks, previous=None) -> str:
    previous_part = ""
    if previous:
        previous_part = f'\nYour previous rewrite was: "{previous["line"]}"\nIt failed because: {previous.get("reason", "")}\nFix that problem this time.\n'
    medium = "spoken line" if claim.origin == "speech" else "short on-screen text"
    evidence = f"[{evidence_chunk.id}] {evidence_chunk.text}" if evidence_chunk else "no chunk states it"
    return f"""You fix one line in a creator's sponsored video so the brand can approve it. Write in the creator's own voice.
Rules:
- Keep the creator's point and topic. Correct only what the sources do not support.
- Keep the same language mix. If the line mixes Tagalog and English (Taglish), the new line does too.
- State only what the source chunks support. Never delete the claim and never replace it with a pleasantry or a call to action.
- One {medium}, similar length, natural to say out loud.
Rule pack: {pack.name}. {pack.guidance}

Creator's line ({claim.origin}, {claim.start:.1f}s): "{claim.quote}"
Lines around it: "{claim.context}"
Why it failed: {reason} Evidence: {evidence}
{previous_part}
Source chunks:
{chunks_block(chunks)}

Return JSON only: {{"line":"<the new line>","chunk_ids":["<ids of chunks that support it>"],"what_changed":"<one short sentence>"}}"""


def check_disclosure(sentences, frames, disclaimer_label=None) -> str:
    disclaimer = ""
    disclaimer_json = ""
    if disclaimer_label:
        disclaimer = f'\nAlso look for this required line, spoken or on screen: "{disclaimer_label}" (Tagalog: "Mahalagang paalala: ... hindi gamot at hindi dapat gamiting panggamot").'
        disclaimer_json = ',"disclaimer":{"found":true|false,"where":"spoken|on_screen|none","quote":"<exact words or empty>"}'
    return f"""Check whether this sponsored creator video discloses the paid partnership. A disclosure can be spoken in English, Tagalog or Taglish (for example "paid partnership", "sponsored", "bayad na partnership", "in partnership with", "#ad") or written on screen.

Speech, one sentence per line as [index] (start-end) text:
{speech_block(sentences)}

{frames_block(frames)} Read the on-screen text in every frame.{disclaimer}

Return JSON only:
{{"spoken":{{"found":true|false,"sentence":<index or null>,"quote":"<exact words or empty>"}},
 "on_screen":{{"found":true|false,"frame_time":<seconds or null>,"text":"<exact on-screen words or empty>"}},
 "frame_text":[{{"frame_time":<seconds>,"text":"<all readable text, or empty>"}}]{disclaimer_json}}}"""


def check_coverage(sentences, messages, frame_text) -> str:
    numbered = "\n".join(f"{i + 1}. {m}" for i, m in enumerate(messages))
    screen = "\n".join(f"{f['frame_time']}s: {f['text']}" for f in frame_text if f.get("text")) or "(not read yet)"
    return f"""A brand brief lists key messages the creator must get across. For each message decide: "covered" (clearly said), "partly" (touched on but incomplete) or "missing".

Speech, one sentence per line as [index] (start-end) text:
{speech_block(sentences)}

On-screen text read from frames:
{screen}

Key messages:
{numbered}

Return JSON only:
{{"messages":[{{"message":"<the key message as given>","status":"covered|partly|missing","sentence":<index or null>,"quote":"<exact words from the speech, or empty>"}}]}}"""


PLANNER = """You are the Dasho Claim Check review agent. You review one creator video for a brand team before it posts. You choose the next tool calls, you see each result, and you decide again.

The review is complete when:
1. Every URL in the source pack has been read with fetch_source (claims are judged against the fetched text).
2. extract_claims has run once.
3. judge_claim has run on every claim (target "original").
4. Every claim that failed has a rewrite: call rewrite_line, then judge_claim with target "rewrite". If that rewrite fails, call rewrite_line again (it receives the failure reason), then judge it. At most 2 rewrites per claim.
5. check_disclosure and check_coverage have run.
Then call finish with a one-sentence summary.

Call several independent tools in one turn when you can (for example, judge three claims at once). search_source is optional: use it when a claim needs evidence that the default retrieval may miss, then pass the chunk ids to judge_claim. You have {remaining} of {cap} actions left, so plan for the work to fit. You never approve the video. A named human approves it.

Current state:
{status}

Results so far (newest last):
{history}"""


JSON_PLANNER_SUFFIX = """

Tools you can call:
{tools}

Return JSON only: {{"reasoning":"<one sentence on why>","actions":[{{"tool":"<tool name>","args":{{...}}}}]}}"""


def tools_text(tool_specs: list[dict]) -> str:
    return "\n".join(f"- {t['name']}: {t['description']} Args: {json.dumps(t['parameters'].get('properties', {}))}" for t in tool_specs)
