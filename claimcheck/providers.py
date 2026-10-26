"""Fixture, Meta, and OpenAI-compatible model providers."""
import base64
import json
import os
import re
import urllib.request
from typing import Any

from .media import extract_media
from .tools import check_coverage, check_disclosure, judge_claim, rewrite_line

TOOLS = ["search_source", "fetch_source", "judge_claim", "rewrite_line", "check_disclosure", "check_coverage", "finish"]


class FixtureProvider:
    def transcribe(self, video: bytes) -> str:
        return "Ledgerly AI guarantees zero FDA violations. Sulit ito!"

    def frames(self, video: bytes) -> list[dict[str, Any]]:
        return [{"time": 1.2, "text": "Paid partnership", "claims": []}]

    def plan(self, context, history):
        return {"tool": "search_source", "input": {"query": "capabilities"}, "mode": "fixture"}

    def json_plan(self, prompt):
        return {"tool": "finish", "input": {}, "mode": "json"}

    def judge_claim(self, claim, evidence):
        return judge_claim(claim, evidence)

    def rewrite_line(self, claim, reason, evidence, round_number=1):
        return rewrite_line(claim, reason, evidence)

    def check_disclosure(self, transcript, frames):
        return check_disclosure(transcript, frames)

    def check_coverage(self, transcript, key_messages):
        return check_coverage(transcript, key_messages)


class MetaProvider(FixtureProvider):
    """Meta.ai v1 provider for Muse Voice Transcribe and Muse Spark."""
    def __init__(self):
        self.base_url = os.getenv("MODEL_BASE_URL", "https://api.meta.ai/v1").rstrip("/")
        self.api_key = os.environ["MODEL_API_KEY"]
        self.asr_model = os.getenv("ASR_MODEL", "muse-voice-transcribe")
        self.model = os.getenv("MODEL_NAME", "muse-spark")
        self.mode = "tools"

    def _request(self, path, payload):
        request = urllib.request.Request(self.base_url + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read())

    def transcribe(self, video):
        audio, self._frames = extract_media(video)
        result = self._request("/audio/transcriptions", {"model": self.asr_model, "file": base64.b64encode(audio).decode(), "language": "tl,en", "prompt": "Tagalog and English (Taglish) speech"})
        return result.get("text", result.get("transcript", ""))

    def frames(self, video):
        return [{"time": frame["time"], "image": base64.b64encode(frame["image"]).decode()} for frame in getattr(self, "_frames", [])]

    def _spark(self, instruction, images=None):
        content = [{"type": "input_text", "text": instruction}]
        for image in images or []:
            content.append({"type": "input_image", "image_url": "data:image/jpeg;base64," + image["image"]})
        return self._request("/responses", {"model": getattr(self, "model", "muse-spark"), "input": content})

    @staticmethod
    def _output(result):
        if "output_text" in result:
            return result["output_text"]
        for item in result.get("output", []):
            for content in item.get("content", []):
                if content.get("type") in ("output_text", "text"):
                    return content.get("text", "")
        return ""

    def _json_call(self, prompt, images=None):
        raw = self._output(self._spark(prompt, images))
        for attempt in range(2):
            try:
                match = re.search(r"\{.*\}", raw, re.S)
                return json.loads(match.group() if match else raw)
            except (TypeError, json.JSONDecodeError):
                if attempt:
                    raise
                raw = self._output(self._spark("Return valid JSON only. Previous response was invalid.\n" + prompt, images))

    def plan(self, context, history):
        schema = [{"type": "function", "name": name, "description": f"Run {name}", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "claim": {"type": "string"}}, "additionalProperties": True}} for name in TOOLS]
        try:
            result = self._request("/responses", {"model": getattr(self, "model", "muse-spark"), "input": [{"type": "input_text", "text": json.dumps({"context": context, "history": history})}], "tools": schema})
            for item in result.get("output", []):
                if item.get("type") == "function_call":
                    self.mode = "tools"
                    return {"tool": item.get("name"), "input": json.loads(item.get("arguments", "{}")), "mode": "tools"}
            raise ValueError("No tool call in response")
        except Exception:
            self.mode = "json"
            plan = self._json_call("Choose one tool from " + json.dumps(TOOLS) + ". Return {tool,input}. Context/history: " + json.dumps({"context": context, "history": history}))
            plan["mode"] = "json"
            return plan

    def json_plan(self, prompt):
        return self._json_call(prompt)

    def judge_claim(self, claim, evidence):
        prompt = f"Judge whether this advertising claim is supported by sources. Return JSON verdict (pass/fail), evidence, reason. Claim: {claim}\nSources: {evidence}"
        return self._json_call(prompt)

    def rewrite_line(self, claim, reason, evidence, round_number=1):
        return self._output(self._spark(f"Rewrite this line to be true according to sources. Preserve the creator's point and voice; never delete the claim. Round {round_number}. Failure reason: {reason}\nLine: {claim}\nSources: {evidence}"))

    def check_disclosure(self, transcript, frames):
        return bool(self._json_call("Return JSON {disclosed:boolean}. Is paid disclosure present in the speech or these video frames? Speech: " + transcript, frames).get("disclosed"))

    def check_coverage(self, transcript, key_messages):
        return self._json_call(f"Return JSON covered and missing arrays. Speech: {transcript}\nRequired messages: {key_messages}")


class OpenAICompatibleProvider(FixtureProvider):
    """Chat Completions compatible local provider used for Muse Spark Open."""
    def __init__(self):
        self.base_url = os.getenv("MODEL_BASE_URL", "http://localhost:1234/v1").rstrip("/")
        self.api_key = os.getenv("MODEL_API_KEY", "local")
        self.model = os.getenv("MODEL_NAME", "muse-spark-open")

    def _chat(self, messages, **kwargs):
        payload = {"model": self.model, "messages": messages, **kwargs}
        request = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read())

    def json_plan(self, prompt):
        for attempt in range(2):
            result = self._chat([{"role": "user", "content": prompt if attempt == 0 else "Return valid JSON only. " + prompt}])
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            try:
                match = re.search(r"\{.*\}", content, re.S)
                return json.loads(match.group() if match else content)
            except (TypeError, json.JSONDecodeError):
                if attempt:
                    raise


def get_provider():
    return MetaProvider() if os.getenv("MODEL_API_KEY") else FixtureProvider()
