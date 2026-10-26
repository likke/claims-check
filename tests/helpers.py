"""Mock providers for tests. No network."""
from __future__ import annotations

import json
from collections import defaultdict

import httpx

from claimcheck.agent import Agent, build_state
from claimcheck.fixture import FixtureProvider
from claimcheck.packs import get_pack
from claimcheck.providers import BaseProvider, MetaProvider, PlanResult


class Scripted(BaseProvider):
    """A provider whose model answers are scripted per task.

    ``plans`` is a list of action lists (popped in order) or a callable taking
    the planner context. ``responses[task]`` is a dict or a callable
    ``(context, prompt) -> dict``.
    """
    name = "mock"
    label = "mock"
    needs_media = False

    def __init__(self, plans=None, responses=None, mode="native_tools"):
        super().__init__()
        self.plans = plans if callable(plans) else list(plans or [])
        self.responses = responses or {}
        self.prompts = defaultdict(list)
        self.images = defaultdict(list)
        self.mode = mode

    def transcribe(self, wav, keywords):
        raise AssertionError("tests build state directly")

    def complete_json(self, task, prompt, images=None, context=None):
        self.prompts[task].append(prompt)
        self.images[task].append(images)
        self.calls.append({"kind": "model", "task": task, "ms": 0, "ok": True, "tokens": None})
        answer = self.responses[task]
        return answer(context, prompt) if callable(answer) else answer

    def plan(self, prompt, tools, context=None):
        self.prompts["plan"].append(prompt)
        self.calls.append({"kind": "model", "task": "plan", "ms": 0, "ok": True, "tokens": None})
        if callable(self.plans):
            actions = self.plans(context)
        else:
            actions = self.plans.pop(0) if self.plans else [{"tool": "finish", "args": {"summary": "done"}}]
        return PlanResult(self.mode, actions, reasoning="scripted plan")


def policy(context):
    """Reuse the fixture's fixed policy as a well-behaved planner."""
    return FixtureProvider(step_delay=0).plan("", [], context).actions


def state(turns, sources, messages=(), pack="Tech claims", frames=None, supplement=False, product="Ledgerly AI"):
    return build_state({"turns": turns, "duration": turns[-1]["end"] if turns else 0, "timing": "asr"},
                       frames or [], sources, list(messages), get_pack(pack), product, supplement)


def run(provider, st, cap=20):
    return Agent(st, provider, cap=cap, fetcher=lambda url: "Fetched text about invoices.").run()


def extraction(*quotes):
    """An extract_claims answer for sentences 0..n."""
    return {"claims": [{"sentence": i, "quote": q, "kind": "other"} for i, q in quotes], "on_screen_claims": []}


def supported(chunk_id, quote, keeps=True):
    return {"verdict": "supported", "chunk_id": chunk_id, "evidence_quote": quote, "reason": "The source states this.", "keeps_point": keeps}


def meta_with(handler) -> MetaProvider:
    """A MetaProvider whose HTTP calls go to ``handler(request) -> httpx.Response``."""
    requests = []

    def record(request):
        requests.append(request)
        return handler(request)

    provider = MetaProvider(api_key="test-key", client=httpx.Client(transport=httpx.MockTransport(record)))
    provider.requests = requests
    return provider


def responses_text(text: str) -> httpx.Response:
    return httpx.Response(200, json={"output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
                                     "usage": {"total_tokens": 10}})


def body(request: httpx.Request) -> dict:
    return json.loads(request.content)
