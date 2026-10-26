"""Model providers.

- MetaProvider: Meta Model API (https://api.meta.ai/v1). Muse Voice Transcribe
  for speech, Muse Spark /responses for text, vision and native tool calling.
- OpenAICompatibleProvider: Muse Spark Open (open weights) served by any
  OpenAI-compatible endpoint, for example a local server or Fireworks.
- FixtureProvider (in fixture.py): offline demo with a fictional product.

Every provider exposes the same four calls: transcribe, complete_json, plan
and info. The agent never knows which one it is talking to.
"""
from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass

import httpx

from .prompts import JSON_PLANNER_SUFFIX, tools_text


# Muse Voice Transcribe's evaluated languages include Tagalog and English but
# not Cebuano (dev.meta.ai/docs/models, checked 27 Oct 2026). The bias stays
# Tagalog and English; other languages are flagged for a person (see tools.py).
LANGUAGE_BIAS = ["Tagalog", "English"]


class ProviderError(RuntimeError):
    pass


class ToolsRejected(ProviderError):
    """The endpoint refused a request that carried a tools array."""


@dataclass
class PlanResult:
    mode: str  # "native_tools", "json_planner" or "fixture_planner"
    actions: list[dict]
    reasoning: str = ""
    fallback_reason: str | None = None
    calls: int = 1


def parse_json(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in model output")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("model output is not a JSON object")
    return value


def normalise_actions(raw, tool_names: list[str]) -> list[dict]:
    actions = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("tool") or item.get("name") or "").strip()
        args = item.get("args") if "args" in item else item.get("arguments", item.get("input", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args or "{}")
            except json.JSONDecodeError:
                args = {}
        actions.append({"tool": name, "args": args if isinstance(args, dict) else {}})
    return actions


class BaseProvider:
    name = "base"
    label = ""
    needs_media = True

    def __init__(self):
        self.calls: list[dict] = []
        self.tool_mode = "native"

    # -- bookkeeping ------------------------------------------------------
    def _record(self, kind: str, task: str, started: float, ok: bool, tokens: int | None = None):
        self.calls.append({"kind": kind, "task": task, "ms": int((time.monotonic() - started) * 1000), "ok": ok, "tokens": tokens})

    def usage(self) -> dict:
        tasks = Counter(c["task"] for c in self.calls)
        return {
            "model_calls": sum(1 for c in self.calls if c["kind"] == "model"),
            "asr_calls": sum(1 for c in self.calls if c["kind"] == "asr"),
            "tokens": sum(c["tokens"] or 0 for c in self.calls),
            "by_task": dict(tasks),
        }

    def info(self) -> dict:
        return {"name": self.name, "label": self.label}

    # -- shared behaviour -------------------------------------------------
    def complete_json(self, task: str, prompt: str, images: list[dict] | None = None, context: dict | None = None) -> dict:
        text = self._generate(prompt, images, task)
        try:
            return parse_json(text)
        except (ValueError, json.JSONDecodeError):
            text = self._generate("Your previous answer was not valid JSON. Return one JSON object only.\n\n" + prompt, images, task + "_retry")
            return parse_json(text)

    def plan(self, prompt: str, tools: list[dict], context: dict | None = None) -> PlanResult:
        names = [t["name"] for t in tools]
        fallback = None
        if self.tool_mode == "native":
            try:
                calls, text = self._native_tools(prompt, tools)
                actions = normalise_actions(calls, names)
                if not actions and text:
                    try:
                        actions = normalise_actions(parse_json(text).get("actions"), names)
                    except (ValueError, json.JSONDecodeError):
                        pass
                return PlanResult("native_tools", actions, reasoning=text.strip())
            except ToolsRejected as rejected:
                self.tool_mode = "json"
                fallback = str(rejected)
        data = self.complete_json("plan", prompt + JSON_PLANNER_SUFFIX.format(tools=tools_text(tools)), context=context)
        return PlanResult("json_planner", normalise_actions(data.get("actions"), names),
                          reasoning=str(data.get("reasoning", "")), fallback_reason=fallback)

    # -- provider specific ------------------------------------------------
    def transcribe(self, wav: bytes, keywords: list[str]) -> dict:
        raise NotImplementedError

    def _generate(self, prompt: str, images: list[dict] | None, task: str) -> str:
        raise NotImplementedError

    def _native_tools(self, prompt: str, tools: list[dict]) -> tuple[list[dict], str]:
        raise NotImplementedError


def _post(client: httpx.Client, url: str, **kwargs) -> httpx.Response:
    """POST with one retry on rate limits, server errors and timeouts."""
    for attempt in range(2):
        try:
            response = client.post(url, **kwargs)
        except (httpx.TimeoutException, httpx.TransportError) as error:
            if attempt:
                raise ProviderError(f"request to {url} failed: {error}") from None
            time.sleep(2)
            continue
        if response.status_code in (429, 500, 502, 503, 504) and not attempt:
            time.sleep(2)
            continue
        return response
    raise ProviderError(f"request to {url} failed")


def _error_text(response: httpx.Response) -> str:
    return f"HTTP {response.status_code}: {response.text[:300]}"


class MetaProvider(BaseProvider):
    name = "meta"

    def __init__(self, api_key: str | None = None, client: httpx.Client | None = None):
        super().__init__()
        self.api_key = api_key or os.environ["MODEL_API_KEY"]
        self.base_url = os.getenv("MODEL_BASE_URL", "https://api.meta.ai/v1").rstrip("/")
        self.model = os.getenv("MODEL_NAME", "muse-spark-1.3")
        self.asr_model = os.getenv("ASR_MODEL", "muse-voice-transcribe-1.0")
        self.planner_effort = os.getenv("PLANNER_REASONING", "low")
        self.task_effort = os.getenv("TASK_REASONING", "")
        self.client = client or httpx.Client(timeout=180)
        self.label = f"Meta cloud · {self.model} + {self.asr_model}"

    def info(self) -> dict:
        return {"name": self.name, "label": self.label, "model": self.model, "asr_model": self.asr_model, "base_url": self.base_url}

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _responses(self, payload: dict, task: str) -> dict:
        started = time.monotonic()
        response = _post(self.client, self.base_url + "/responses", json=payload, headers=self._headers())
        if response.status_code >= 400:
            self._record("model", task, started, False)
            if "tools" in payload and response.status_code in (400, 404, 422):
                raise ToolsRejected(_error_text(response))
            raise ProviderError(_error_text(response))
        data = response.json()
        self._record("model", task, started, True, (data.get("usage") or {}).get("total_tokens"))
        return data

    @staticmethod
    def output_text(data: dict) -> str:
        if isinstance(data.get("output_text"), str):
            return data["output_text"]
        parts = []
        for item in data.get("output", []):
            if item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") in ("output_text", "text"):
                        parts.append(content.get("text", ""))
        return "\n".join(parts)

    @staticmethod
    def _content(prompt: str, images: list[dict] | None) -> list[dict]:
        content = [{"type": "input_text", "text": prompt}]
        for image in images or []:
            content.append({"type": "input_image", "image_url": "data:image/jpeg;base64," + image["image"]})
        return content

    def _generate(self, prompt, images, task):
        payload = {"model": self.model, "input": [{"type": "message", "role": "user", "content": self._content(prompt, images)}]}
        if self.task_effort:
            payload["reasoning"] = {"effort": self.task_effort}
        return self.output_text(self._responses(payload, task))

    def _native_tools(self, prompt, tools):
        payload = {
            "model": self.model,
            "input": [{"type": "message", "role": "user", "content": self._content(prompt, None)}],
            "tools": [{"type": "function", "name": t["name"], "description": t["description"], "parameters": t["parameters"]} for t in tools],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        }
        if self.planner_effort:
            payload["reasoning"] = {"effort": self.planner_effort}
        data = self._responses(payload, "plan")
        calls = [{"tool": item.get("name"), "args": item.get("arguments", "{}")} for item in data.get("output", []) if item.get("type") == "function_call"]
        return calls, self.output_text(data)

    def transcribe(self, wav, keywords):
        settings = {"model": self.asr_model, "mode": "ENDPOINTING", "audioEncoding": "WAV",
                    "languageBias": LANGUAGE_BIAS, "keywords": [k for k in keywords if k][:10]}
        started = time.monotonic()
        response = _post(
            self.client, self.base_url + "/asr/transcribe", headers=self._headers(),
            files={"request": (None, json.dumps(settings), "application/json"), "audio": ("audio.wav", wav, "audio/wav")},
        )
        if response.status_code >= 400:
            self._record("asr", "transcribe", started, False)
            raise ProviderError("speech transcription failed: " + _error_text(response))
        data = response.json()
        self._record("asr", "transcribe", started, True)
        turns = [{"start": t.get("startMs", 0) / 1000, "end": t.get("endMs", 0) / 1000, "text": t.get("transcript", "")}
                 for t in data.get("turns") or [] if t.get("transcript", "").strip()]
        duration = (data.get("audioDurationMs") or 0) / 1000 or None
        if not turns and data.get("transcript", "").strip():
            turns = [{"start": 0.0, "end": duration or 0.0, "text": data["transcript"], "estimated": True}]
        return {"turns": turns, "duration": duration, "timing": "asr"}


class OpenAICompatibleProvider(BaseProvider):
    """Muse Spark Open behind an OpenAI-compatible /chat/completions endpoint."""
    name = "open"

    def __init__(self, client: httpx.Client | None = None):
        super().__init__()
        self.base_url = os.getenv("OPEN_BASE_URL", "http://localhost:8000/v1").rstrip("/")
        self.api_key = os.getenv("OPEN_API_KEY", "local")
        self.model = os.getenv("OPEN_MODEL", "muse-spark-open")
        self.asr_model = os.getenv("OPEN_ASR_MODEL", "whisper-1")
        self.client = client or httpx.Client(timeout=180)
        self.label = f"Muse Spark Open · {self.model} at {self.base_url}"

    def info(self) -> dict:
        return {"name": self.name, "label": self.label, "model": self.model, "base_url": self.base_url}

    def _chat(self, payload: dict, task: str) -> dict:
        started = time.monotonic()
        response = _post(self.client, self.base_url + "/chat/completions", json=payload, headers={"Authorization": f"Bearer {self.api_key}"})
        if response.status_code >= 400:
            self._record("model", task, started, False)
            if "tools" in payload and response.status_code in (400, 404, 422):
                raise ToolsRejected(_error_text(response))
            raise ProviderError(_error_text(response))
        data = response.json()
        self._record("model", task, started, True, (data.get("usage") or {}).get("total_tokens"))
        return data

    def _generate(self, prompt, images, task):
        if images:
            content = [{"type": "text", "text": prompt}] + [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + i["image"]}} for i in images]
        else:
            content = prompt
        data = self._chat({"model": self.model, "messages": [{"role": "user", "content": content}]}, task)
        return ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""

    def _native_tools(self, prompt, tools):
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "tools": [{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}} for t in tools],
            "tool_choice": "auto",
        }
        message = ((self._chat(payload, "plan").get("choices") or [{}])[0].get("message")) or {}
        calls = [{"tool": c.get("function", {}).get("name"), "args": c.get("function", {}).get("arguments", "{}")} for c in message.get("tool_calls") or []]
        return calls, message.get("content") or ""

    def transcribe(self, wav, keywords):
        if os.getenv("MODEL_API_KEY"):
            meta = MetaProvider(client=self.client)
            result = meta.transcribe(wav, keywords)
            self.calls.extend(meta.calls)
            return result
        started = time.monotonic()
        url = os.getenv("OPEN_ASR_URL", self.base_url + "/audio/transcriptions")
        response = _post(self.client, url, headers={"Authorization": f"Bearer {self.api_key}"},
                         data={"model": self.asr_model, "response_format": "verbose_json"}, files={"file": ("audio.wav", wav, "audio/wav")})
        if response.status_code >= 400:
            self._record("asr", "transcribe", started, False)
            raise ProviderError("speech transcription failed: " + _error_text(response))
        data = response.json()
        self._record("asr", "transcribe", started, True)
        turns = [{"start": s.get("start", 0.0), "end": s.get("end", 0.0), "text": s.get("text", "")} for s in data.get("segments") or []]
        if not turns and data.get("text"):
            turns = [{"start": 0.0, "end": data.get("duration") or 0.0, "text": data["text"], "estimated": True}]
        return {"turns": turns, "duration": data.get("duration"), "timing": "asr"}


def get_provider(name: str | None = None):
    from .fixture import FixtureProvider

    choice = (name or os.getenv("CLAIMCHECK_PROVIDER") or "").strip().lower()
    if not choice:
        choice = "meta" if os.getenv("MODEL_API_KEY") else "fixture"
    if choice == "meta":
        return MetaProvider()
    if choice in ("open", "muse-spark-open", "openai"):
        return OpenAICompatibleProvider()
    return FixtureProvider()
