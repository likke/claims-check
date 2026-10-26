"""Disclosure, coverage, FDA pack, claim extraction and timing."""
import json

import httpx

from claimcheck.text import sentences_from_turns
from claimcheck.tools import ToolBox
from helpers import Scripted, body, extraction, meta_with, policy, responses_text, run, state, supported

SOURCES = "Ledgerly AI flags unusual charges for a human reviewer."
FRAMES = [{"time": 1.0, "image": "aW1n"}, {"time": 2.5, "image": "aW1n"}, {"time": 5.0, "image": "aW1n"}]


def test_taglish_spoken_disclosure_is_found_and_verified():
    turns = [{"start": 0.0, "end": 2.8, "text": "Uy guys, may bayad na partnership kami with Ledgerly AI."},
             {"start": 2.8, "end": 6.0, "text": "Sobrang bilis niya mag-flag ng charges."}]
    answer = {"spoken": {"found": True, "sentence": 0, "quote": "may bayad na partnership kami with Ledgerly AI"},
              "on_screen": {"found": False, "frame_time": None, "text": ""}, "frame_text": []}
    provider = meta_with(lambda request: responses_text(json.dumps(answer)))
    st = state(turns, SOURCES)
    observation, summary = ToolBox(st, provider).run("check_disclosure", {})
    assert st.disclosure["disclosed"] is True
    spoken = st.disclosure["spoken"]
    assert spoken["found"] and spoken["quote"] == answer["spoken"]["quote"] and spoken["how"] == "model, quote verified in transcript"
    assert 0.5 < spoken["time"] < 0.7  # the quote starts at word 3 of a 2.8 s sentence
    assert f"spoken at {spoken['time']}s" in summary


def test_taglish_disclosure_found_by_code_when_the_model_misses_it():
    turns = [{"start": 0.0, "end": 3.0, "text": "Sponsored ito ng Ledgerly AI, mga kaibigan."}]
    provider = Scripted(responses={"check_disclosure": {"spoken": {"found": False}, "on_screen": {"found": False}, "frame_text": []}})
    st = state(turns, SOURCES)
    ToolBox(st, provider).run("check_disclosure", {})
    assert st.disclosure["spoken"]["found"] and st.disclosure["spoken"]["how"] == "phrase match in transcript"


def test_invented_spoken_quote_is_rejected():
    turns = [{"start": 0.0, "end": 3.0, "text": "Ledgerly AI is fast."}]
    provider = Scripted(responses={"check_disclosure": {"spoken": {"found": True, "sentence": 0, "quote": "this is a paid partnership"},
                                                        "on_screen": {"found": False}, "frame_text": []}})
    st = state(turns, SOURCES)
    ToolBox(st, provider).run("check_disclosure", {})
    assert st.disclosure["disclosed"] is False
    assert any("not in the transcript" in note for note in st.disclosure["notes"])


def test_disclosure_found_from_frames_with_a_sampled_timestamp():
    turns = [{"start": 0.0, "end": 4.0, "text": "Ledgerly AI flags unusual charges for review."}]
    answer = {"spoken": {"found": False, "sentence": None, "quote": ""},
              "on_screen": {"found": True, "frame_time": 2.4, "text": "Paid partnership with Ledgerly AI"},
              "frame_text": [{"frame_time": 1.0, "text": ""}, {"frame_time": 2.5, "text": "Paid partnership with Ledgerly AI"},
                             {"frame_time": 5.0, "text": ""}]}
    provider = meta_with(lambda request: responses_text(json.dumps(answer)))
    st = state(turns, SOURCES, frames=[dict(f) for f in FRAMES])
    ToolBox(st, provider).run("check_disclosure", {})
    assert st.disclosure["disclosed"] is True and st.disclosure["spoken"]["found"] is False
    assert st.disclosure["on_screen"]["time"] == 2.5  # snapped to the frame we sampled
    sent = body(provider.requests[0])
    content = sent["input"][0]["content"]
    assert [c["type"] for c in content].count("input_image") == 3
    assert content[1]["image_url"].startswith("data:image/jpeg;base64,")
    assert "frame 2 at 2.50s" in content[0]["text"]


def test_on_screen_disclosure_at_a_frame_we_never_sampled_is_ignored():
    turns = [{"start": 0.0, "end": 4.0, "text": "Ledgerly AI flags unusual charges for review."}]
    answer = {"spoken": {"found": False}, "on_screen": {"found": True, "frame_time": 9.0, "text": "Ad"}, "frame_text": []}
    provider = Scripted(responses={"check_disclosure": answer})
    st = state(turns, SOURCES, frames=[dict(f) for f in FRAMES])
    ToolBox(st, provider).run("check_disclosure", {})
    assert st.disclosure["disclosed"] is False


def test_fda_pack_fails_cure_claims_and_flags_missing_supplement_line():
    turns = [{"start": 0.0, "end": 3.0, "text": "This serum cures acne in three days."},
             {"start": 3.0, "end": 6.0, "text": "Ang galing talaga ng serum na ito."}]
    sources = "Glowa serum moisturizes dry skin.\nGlowa serum is a fictional product."
    rewrites = iter([{"line": "This serum heals acne in three days, promise.", "chunk_ids": []},
                     {"line": "This serum moisturizes dry skin in three days.", "chunk_ids": ["S1"]}])

    def judge(ctx, prompt):
        if "moisturizes" in ctx["line"]:
            return supported("S1", "Glowa serum moisturizes dry skin")
        return supported("S1", "Glowa serum moisturizes dry skin")  # the model wrongly passes the cure claim

    responses = {
        "extract_claims": extraction((0, turns[0]["text"])),
        "judge_claim": judge,
        "rewrite_line": lambda ctx, prompt: next(rewrites),
        "check_disclosure": {"spoken": {"found": False}, "on_screen": {"found": False}, "frame_text": [],
                             "disclaimer": {"found": False, "where": "none", "quote": ""}},
        "check_coverage": {"messages": []},
    }
    provider = Scripted(policy, responses)
    result = run(provider, state(turns, sources, pack="FDA beauty", supplement=True, product="Glowa serum"))
    claim = result["claims"][0]
    assert claim["rounds"][0]["verdict"] == "fail" and claim["rounds"][0]["judgement"] == "rule"
    assert claim["rounds"][0]["chunk_id"] == "R1" and "treat" in claim["rounds"][0]["evidence"]
    assert claim["rounds"][1]["judgement"] == "rejected_by_code"  # "heals" is still a therapeutic claim
    assert claim["status"] == "fixed" and "moisturizes" in claim["suggested_line"]
    required = result["disclosure"]["required_disclaimer"]
    assert required["label"] == "No approved therapeutic claims" and required["found"] is False
    assert "Food supplement" in provider.prompts["judge_claim"][0] or "R2" in provider.prompts["judge_claim"][0]


def test_fda_supplement_line_found_on_screen():
    turns = [{"start": 0.0, "end": 3.0, "text": "Masarap inumin araw-araw."}]
    answer = {"spoken": {"found": False}, "on_screen": {"found": False},
              "frame_text": [{"frame_time": 1.0, "text": "MAHALAGANG PAALALA: No approved therapeutic claims"}]}
    provider = Scripted(responses={"check_disclosure": answer})
    st = state(turns, "Fictional tea.", pack="FDA beauty", frames=[dict(f) for f in FRAMES], supplement=True)
    ToolBox(st, provider).run("check_disclosure", {})
    assert st.disclosure["required_disclaimer"]["found"] is True
    assert st.disclosure["required_disclaimer"]["where"] == "on_screen" and st.disclosure["required_disclaimer"]["time"] == 1.0


def test_tech_pack_does_not_require_the_supplement_line():
    provider = Scripted(responses={"check_disclosure": {"spoken": {"found": False}, "on_screen": {"found": False}, "frame_text": []}})
    st = state([{"start": 0, "end": 2, "text": "Ledgerly AI is fast."}], SOURCES)
    ToolBox(st, provider).run("check_disclosure", {})
    assert st.disclosure["required_disclaimer"] is None


def test_extraction_keeps_only_quotes_found_in_speech_or_sampled_frames():
    turns = [{"start": 0.0, "end": 6.0, "text": "I tried Ledgerly AI. It guarantees zero fraud. Every charge gets a human reviewer."}]
    answer = {"claims": [{"sentence": 1, "quote": "It guarantees zero fraud."},
                         {"sentence": 0, "quote": "It doubles your revenue."}],
              "on_screen_claims": [{"frame_time": 2.5, "text": "ZERO FRAUD"}, {"frame_time": 7.7, "text": "FREE FOREVER"}]}
    provider = Scripted(responses={"extract_claims": answer})
    st = state(turns, SOURCES, frames=[dict(f) for f in FRAMES])
    observation, _ = ToolBox(st, provider).run("extract_claims", {})
    assert [c.quote for c in st.claims] == ["It guarantees zero fraud.", "ZERO FRAUD"]
    spoken, screen = st.claims
    assert spoken.estimated is True and 1.0 < spoken.start < 3.5  # sentence time estimated inside the ASR turn
    assert screen.origin == "on_screen" and screen.start == 2.5 and screen.estimated is False
    assert len(observation["rejected"]) == 2
    assert provider.images["extract_claims"][0] and len(provider.images["extract_claims"][0]) == 3


def test_sentence_timing_single_sentence_turns_are_exact():
    sentences = sentences_from_turns([{"start": 1.0, "end": 2.0, "text": "One."}, {"start": 2.0, "end": 5.0, "text": "Two words. And three more words."}])
    assert (sentences[0].start, sentences[0].end, sentences[0].estimated) == (1.0, 2.0, False)
    assert sentences[1].estimated and sentences[2].estimated
    assert sentences[1].start == 2.0 and abs(sentences[2].end - 5.0) < 1e-9 and sentences[1].end < sentences[2].end


def test_coverage_needs_a_real_quote():
    turns = [{"start": 0.0, "end": 3.0, "text": "Every draft gets a named approver before it goes live."}]
    answer = {"messages": [{"message": "Named approver before posting", "status": "covered", "sentence": 0, "quote": "Every draft gets a named approver"},
                           {"message": "Free trial", "status": "covered", "sentence": None, "quote": "try it free for a month"}]}
    provider = Scripted(responses={"check_coverage": answer})
    st = state(turns, SOURCES, ["Named approver before posting", "Free trial"])
    ToolBox(st, provider).run("check_coverage", {})
    first, second = st.coverage["messages"]
    assert first["status"] == "covered" and first["time"] == 0.0
    assert second["status"] == "partly" and "lowered" in second["note"]


def test_fetch_source_reads_pack_urls_only():
    st = state([{"start": 0, "end": 2, "text": "Ledgerly AI is fast."}], SOURCES + "\nfixture://ledgerly-ai/pricing")
    tools = ToolBox(st, Scripted())
    observation, _ = tools.run("fetch_source", {"url": "fixture://ledgerly-ai/pricing"})
    assert observation["chunks_added"] and st.sources.fetched_urls["fixture://ledgerly-ai/pricing"].startswith("ok")
    assert "already read" in tools.run("fetch_source", {"url": "fixture://ledgerly-ai/pricing"})[0]["error"]
    assert "only URLs" in tools.run("fetch_source", {"url": "http://169.254.169.254/"})[0]["error"]
    hits = st.sources.search("14-day free trial")
    assert hits[0][0].origin == "fixture://ledgerly-ai/pricing"


def test_private_hosts_are_refused():
    from claimcheck.sources import fetch_url
    try:
        fetch_url("http://127.0.0.1/admin")
    except ValueError as error:
        assert "private" in str(error)
    else:
        raise AssertionError("expected refusal")


def test_meta_asr_request_and_turn_parsing():
    def handler(request):
        assert request.url.path.endswith("/asr/transcribe")
        assert request.headers["authorization"] == "Bearer test-key"
        text = request.content.decode("latin-1")
        assert '"languageBias": ["Tagalog", "English"]' in text and "muse-voice-transcribe-1.0" in text
        return httpx.Response(200, json={"transcript": "Hi. Two.", "audioDurationMs": 4000,
                                         "turns": [{"startMs": 0, "endMs": 1500, "transcript": "Hi."},
                                                   {"startMs": 1500, "endMs": 4000, "transcript": "Two."}]})
    provider = meta_with(handler)
    result = provider.transcribe(b"RIFF....", ["Ledgerly AI"])
    assert result["turns"] == [{"start": 0.0, "end": 1.5, "text": "Hi."}, {"start": 1.5, "end": 4.0, "text": "Two."}]
    assert result["duration"] == 4.0 and provider.usage()["asr_calls"] == 1


def test_meta_native_tools_parse_parallel_calls_and_commentary():
    def handler(request):
        sent = body(request)
        assert sent["tools"][0]["type"] == "function" and sent["parallel_tool_calls"] is True
        return httpx.Response(200, json={"output": [
            {"type": "reasoning", "summary": []},
            {"type": "message", "content": [{"type": "output_text", "text": "Judging both claims."}]},
            {"type": "function_call", "name": "judge_claim", "arguments": "{\"claim_id\":\"C1\",\"target\":\"original\"}"},
            {"type": "function_call", "name": "judge_claim", "arguments": "{\"claim_id\":\"C2\",\"target\":\"original\"}"},
        ], "usage": {"total_tokens": 5}})
    provider = meta_with(handler)
    from claimcheck.tools import TOOL_SPECS
    plan = provider.plan("state", TOOL_SPECS)
    assert plan.mode == "native_tools" and plan.reasoning == "Judging both claims."
    assert [a["args"]["claim_id"] for a in plan.actions] == ["C1", "C2"]


def test_meta_falls_back_to_json_planner_when_tools_are_rejected():
    def handler(request):
        if "tools" in body(request):
            return httpx.Response(400, json={"error": {"message": "tools not supported"}})
        return responses_text('```json\n{"reasoning":"start","actions":[{"tool":"extract_claims","args":{}}]}\n```')
    provider = meta_with(handler)
    from claimcheck.tools import TOOL_SPECS
    first = provider.plan("state", TOOL_SPECS)
    second = provider.plan("state", TOOL_SPECS)
    assert first.mode == "json_planner" and "HTTP 400" in first.fallback_reason
    assert second.mode == "json_planner" and second.fallback_reason is None
    assert sum(1 for r in provider.requests if "tools" in body(r)) == 1
    assert first.actions == [{"tool": "extract_claims", "args": {}}]


def test_json_retry_on_invalid_model_output():
    replies = iter([responses_text("Sure! Here you go."), responses_text('{"messages": []}')])
    provider = meta_with(lambda request: next(replies))
    assert provider.complete_json("check_coverage", "prompt") == {"messages": []}
    assert provider.usage()["by_task"] == {"check_coverage": 1, "check_coverage_retry": 1}


def test_open_weights_provider_uses_chat_completions(monkeypatch):
    from claimcheck.providers import OpenAICompatibleProvider
    from claimcheck.tools import TOOL_SPECS
    monkeypatch.setenv("OPEN_BASE_URL", "http://open.test/v1")
    seen = []

    def handler(request):
        sent = body(request)
        seen.append(sent)
        assert request.url.path == "/v1/chat/completions"
        if "tools" in sent:
            assert sent["tools"][0]["function"]["name"] == "extract_claims"
            return httpx.Response(200, json={"choices": [{"message": {"content": "", "tool_calls": [
                {"function": {"name": "extract_claims", "arguments": "{}"}}]}}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": true}'}}]})
    provider = OpenAICompatibleProvider(client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert provider.plan("state", TOOL_SPECS).actions == [{"tool": "extract_claims", "args": {}}]
    assert provider.complete_json("x", "prompt", images=[{"time": 1, "image": "aW1n"}]) == {"ok": True}
    assert seen[1]["messages"][0]["content"][1]["type"] == "image_url"


def test_claim_inside_a_long_unpunctuated_turn_gets_an_estimated_offset():
    # Seen live on 27 Oct: ASR returned one unpunctuated 9 s turn.
    turns = [{"start": 0.0, "end": 9.0, "text": "i tried it for our posts it guarantees zero violations every draft gets an approver"}]
    provider = Scripted(responses={"extract_claims": {"claims": [{"sentence": 0, "quote": "every draft gets an approver"},
                                                                 {"sentence": 0, "quote": "it guarantees zero violations"}], "on_screen_claims": []}})
    st = state(turns, SOURCES)
    ToolBox(st, provider).run("extract_claims", {})
    late, early = st.claims
    assert late.estimated and early.estimated
    assert 6.0 <= late.start < late.end <= 9.0
    assert 2.0 < early.start < 5.0


def test_vision_box_is_kept_only_when_sane():
    turns = [{"start": 0.0, "end": 4.0, "text": "Ledgerly AI flags unusual charges for review."}]
    for box, expected in (([0.2, 0.7, 0.6, 0.15], True), ([0.9, 0.9, 0.5, 0.5], False), ("near the bottom", False)):
        answer = {"spoken": {"found": False}, "on_screen": {"found": True, "frame_time": 1.0, "text": "Paid partnership", "box": box}, "frame_text": []}
        st = state(turns, SOURCES, frames=[dict(f) for f in FRAMES])
        ToolBox(st, Scripted(responses={"check_disclosure": answer})).run("check_disclosure", {})
        assert (st.disclosure["on_screen"]["box"] is not None) is expected
