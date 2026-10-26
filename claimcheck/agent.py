"""The review agent.

Loop: the planner (Muse Spark) sees the review state and every result so
far, then chooses one or more tool calls. Each call runs, its observation is
logged and fed back, and the planner decides again. The loop stops when the
agent calls ``finish`` or uses its 20 actions.

After the loop a deterministic guard checks the work. Anything the agent
skipped (an unjudged claim, a failed claim with no rewrite, a missing
disclosure check) is run by code and labelled "completed by guard".

Every event is kept. Nothing is dropped from the trace.
"""
from __future__ import annotations

import json
import time
import uuid
from collections import Counter
from datetime import datetime, timezone

from . import prompts
from .packs import Pack
from .sources import SourceIndex, fetch_url
from .state import MAX_REWRITES, ReviewState
from .text import sentences_from_turns
from .tools import TOOL_SPECS, ToolBox

ACTION_CAP = 20


class Trace:
    def __init__(self, emit=None):
        self.events: list[dict] = []
        self.started = time.monotonic()
        self.emit = emit

    def add(self, kind: str, **fields) -> dict:
        event = {"seq": len(self.events) + 1, "t": round(time.monotonic() - self.started, 2), "kind": kind, **fields}
        self.events.append(event)
        if self.emit:
            self.emit(event)
        return event


def _short(value, limit: int = 160) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def build_state(turns: dict, frames: list[dict], sources_text: str, key_messages: list[str], pack: Pack,
                product: str = "", supplement: bool = False) -> ReviewState:
    sentences = sentences_from_turns(turns.get("turns", []))
    duration = turns.get("duration") or (sentences[-1].end if sentences else 0.0)
    timing = turns.get("timing", "asr")
    if timing == "asr" and sentences and all(s.estimated for s in sentences):
        timing = "estimated"
    elif timing == "asr" and any(s.estimated for s in sentences):
        timing = "asr turns, sentence times estimated"
    return ReviewState(
        sentences=sentences, duration=float(duration or 0.0), timing=timing, frames=frames,
        sources=SourceIndex.from_pack(sources_text, list(pack.rules)), key_messages=[m.strip() for m in key_messages if m.strip()],
        pack=pack, product=product, supplement=supplement, sources_text=sources_text,
    )


class Agent:
    def __init__(self, state: ReviewState, provider, emit=None, cap: int = ACTION_CAP, fetcher=fetch_url, trace: Trace | None = None):
        self.state = state
        self.provider = provider
        self.cap = cap
        self.trace = trace or Trace(emit)
        self.tools = ToolBox(state, provider, fetcher)
        self.actions_used = 0
        self.guard_actions = 0
        self.plan_calls = 0
        self.modes: Counter = Counter()
        self.cap_hit = False
        self.history: list[str] = []

    def _execute(self, tool: str, args: dict, by: str, note: str = "") -> dict:
        calls_before = len(self.provider.calls)
        started = time.monotonic()
        try:
            observation, summary = self.tools.run(tool, args)
        except Exception as error:  # a model or network failure inside a tool
            observation, summary = {"error": f"{type(error).__name__}: {error}"}, f"Failed: {error}"
        event = self.trace.add(
            "action", by=by, step=self.actions_used if by == "agent" else None, tool=tool, args=args,
            input_summary=_short(args) if args else "", observation=observation, summary=summary,
            error="error" in observation, note=note, duration_ms=int((time.monotonic() - started) * 1000),
            model_calls=len(self.provider.calls) - calls_before,
        )
        return event

    def run(self) -> dict:
        state = self.state
        self.trace.add("start", summary=f"{len(state.sentences)} sentences, {len(state.frames)} frames, "
                                        f"{len(state.sources.chunks)} source chunks, {len(state.sources.pending_urls)} URLs, pack {state.pack.name}",
                       provider=self.provider.info())
        idle = 0
        while not state.finished:
            if self.actions_used >= self.cap:
                self._cap_reached([])
                break
            prompt = prompts.PLANNER.format(
                remaining=self.cap - self.actions_used, cap=self.cap,
                status=state.status_text(self.actions_used, self.cap),
                history="\n".join(self.history[-16:]) or "(nothing yet)",
            )
            started = time.monotonic()
            try:
                plan = self.provider.plan(prompt, TOOL_SPECS, context={"state": state, "history": self.history,
                                                                       "actions_used": self.actions_used, "cap": self.cap})
            except Exception as error:
                self.trace.add("planner_error", summary=f"Planner failed: {error}; the guard takes over")
                break
            self.plan_calls += 1
            self.modes[plan.mode] += 1
            if plan.fallback_reason:
                self.trace.add("planner_fallback", mode=plan.mode,
                               summary="Native tool calling was rejected; switched to the JSON planner",
                               reason=_short(plan.fallback_reason, 300))
            self.trace.add("plan", mode=plan.mode, reasoning=_short(plan.reasoning, 400),
                           actions=[{"tool": a["tool"], "args": a["args"]} for a in plan.actions],
                           summary=", ".join(a["tool"] for a in plan.actions) or "no tool call",
                           duration_ms=int((time.monotonic() - started) * 1000))
            if not plan.actions:
                idle += 1
                self.history.append("(you returned no tool call; call a tool or finish)")
                if idle >= 2:
                    self.trace.add("planner_stalled", summary="Planner returned no tool call twice; the guard takes over")
                    break
                continue
            idle = 0
            for position, action in enumerate(plan.actions):
                if state.finished:
                    self.trace.add("skipped", summary=f"Skipped {len(plan.actions) - position} calls planned after finish",
                                   dropped=plan.actions[position:])
                    break
                if self.actions_used >= self.cap:
                    self._cap_reached(plan.actions[position:])
                    break
                self.actions_used += 1
                event = self._execute(action["tool"], action["args"], by="agent")
                self.history.append(f"#{self.actions_used} {action['tool']}({_short(action['args'], 80)}) -> {_short(event['summary'], 300)}")
            if self.cap_hit:
                break
        finished_by = "agent" if state.finished else "guard"
        self._guard()
        return self._result(finished_by)

    def _cap_reached(self, dropped: list[dict]):
        if self.cap_hit:
            return
        self.cap_hit = True
        self.trace.add("cap_reached", summary=f"Action cap of {self.cap} reached; {len(self.state.open_work())} open items go to the guard",
                       dropped=dropped, open_work=self.state.open_work())

    def _guard(self):
        """Run by code whatever the agent left open."""
        state = self.state
        open_before = state.open_work()
        self.trace.add("guard", summary="Guard check: " + ("nothing left open" if not open_before else f"{len(open_before)} open items: " + "; ".join(open_before)),
                       open_work=open_before)

        def do(tool, args, why):
            self.guard_actions += 1
            return self._execute(tool, args, by="guard", note="completed by guard: " + why)

        for url in list(state.sources.pending_urls):
            if url not in state.sources.fetched_urls:
                do("fetch_source", {"url": url}, "URL in the source pack was not read")
        if not state.claims_extracted:
            do("extract_claims", {}, "claims were not extracted")
            state.claims_extracted = True
        for claim in state.claims:
            for _ in range(2 * MAX_REWRITES + 1):
                need = claim.needs()
                if need is None:
                    break
                if need == "judge":
                    event = do("judge_claim", {"claim_id": claim.id, "target": "original"}, "claim was not judged")
                elif need == "judge_rewrite":
                    event = do("judge_claim", {"claim_id": claim.id, "target": "rewrite"}, "rewrite was not judged")
                else:
                    event = do("rewrite_line", {"claim_id": claim.id}, f"failed claim had {len(claim.rewrites())} of {MAX_REWRITES} rewrites")
                if event["error"]:
                    break
        if state.disclosure is None:
            event = do("check_disclosure", {}, "disclosure was not checked")
            if event["error"]:
                state.disclosure = {"disclosed": False, "error": event["summary"], "spoken": {"found": False}, "on_screen": {"found": False},
                                    "frame_text": [], "required_disclaimer": None, "notes": ["check failed"]}
        if state.coverage is None:
            event = do("check_coverage", {}, "key messages were not checked")
            if event["error"]:
                state.coverage = {"messages": [], "error": event["summary"]}
        if not state.finished:
            state.finished = True
            self.trace.add("finish", by="guard", summary="Finished by guard (the agent did not call finish)")

    def _result(self, finished_by: str) -> dict:
        state = self.state
        return {
            "id": uuid.uuid4().hex[:12],
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "provider": self.provider.info(),
            "pack": state.pack.name,
            "inputs": {"product": state.product, "pack": state.pack.name, "supplement": state.supplement,
                       "sources": state.sources_text, "key_messages": state.key_messages},
            "product": state.product,
            "supplement": state.supplement,
            "duration": round(state.duration, 2),
            "timing": state.timing,
            "transcript": [s.as_dict() for s in state.sentences],
            "frames": [{"time": f["time"], "text": f.get("text", "")} for f in state.frames],
            "claims": [c.as_dict() for c in state.claims],
            "disclosure": state.disclosure,
            "coverage": state.coverage,
            "sources": [c.as_dict() for c in state.sources.chunks],
            "fetched_urls": state.sources.fetched_urls,
            "agent": {
                "actions": self.actions_used, "cap": self.cap, "cap_hit": self.cap_hit, "plan_calls": self.plan_calls,
                "planner_modes": dict(self.modes), "guard_actions": self.guard_actions, "finished_by": finished_by,
                "summary": state.finish_summary,
            },
            "usage": self.provider.usage(),
            "trace": self.trace.events,
            "approval": {"status": "awaiting human review", "note": "The agent never approves. A named reviewer decides."},
        }


def run_review(provider, *, wav: bytes = b"", frames: list[dict] | None = None, sources: str, key_messages: list[str],
               pack: Pack, product: str = "", supplement: bool = False, emit=None, cap: int = ACTION_CAP, fetcher=fetch_url) -> dict:
    """Transcribe, build the state, run the agent and the guard. Returns the review."""
    trace = Trace(emit)
    calls_before = len(provider.calls)
    transcription = provider.transcribe(wav, [product] if product else [])
    turns = transcription.get("turns", [])
    trace.add("transcribed", summary=f"Speech transcribed: {len(turns)} turns, timing {transcription.get('timing')}",
              model_calls=len(provider.calls) - calls_before, text=" ".join(t.get("text", "") for t in turns))
    state = build_state(transcription, frames or [], sources, key_messages, pack, product, supplement)
    agent = Agent(state, provider, cap=cap, fetcher=fetcher, trace=trace)
    return agent.run()
