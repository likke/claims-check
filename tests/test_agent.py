"""The agent loop: planning, the action cap, the guard and rewrite rounds."""
import pytest

from claimcheck.providers import BaseProvider, ToolsRejected
from helpers import Scripted, extraction, policy, run, state, supported

TURNS = [
    {"start": 0.0, "end": 3.0, "text": "Grabe, Ledgerly AI guarantees zero fraud para sa lahat."},
    {"start": 3.0, "end": 6.0, "text": "It reads a typical invoice in under 10 seconds."},
]
SOURCES = ("Ledgerly AI flags unusual charges for a human reviewer.\n"
           "Ledgerly AI reads a typical invoice in under 10 seconds.\n"
           "Ledgerly AI does not guarantee fraud prevention.")
CONTRADICTED = {"verdict": "contradicted", "chunk_id": "S3", "evidence_quote": "does not guarantee fraud prevention",
                "reason": "The source says it does not guarantee fraud prevention."}


def judge_by_line(context, prompt):
    if "under 10 seconds" in context["line"]:
        return supported("S2", "reads a typical invoice in under 10 seconds")
    if "flags unusual charges" in context["line"]:
        return supported("S1", "flags unusual charges for a human reviewer")
    return CONTRADICTED


def base_responses(**overrides):
    responses = {
        "extract_claims": extraction((0, TURNS[0]["text"]), (1, TURNS[1]["text"])),
        "judge_claim": judge_by_line,
        "rewrite_line": {"line": "Grabe, Ledgerly AI flags unusual charges para ma-review ng team.", "chunk_ids": ["S1"]},
        "check_disclosure": {"spoken": {"found": False}, "on_screen": {"found": False}, "frame_text": []},
        "check_coverage": {"messages": [{"message": "Fast invoices", "status": "covered", "sentence": 1, "quote": "reads a typical invoice in under 10 seconds"}]},
    }
    responses.update(overrides)
    return responses


def agent_actions(result, by="agent"):
    return [e["tool"] for e in result["trace"] if e["kind"] == "action" and e["by"] == by]


def test_planner_chooses_the_actions_and_order():
    plans = [
        [{"tool": "check_coverage", "args": {}}],
        [{"tool": "check_disclosure", "args": {}}],
        [{"tool": "extract_claims", "args": {}}],
        [{"tool": "judge_claim", "args": {"claim_id": "C2", "target": "original"}}],
        [{"tool": "finish", "args": {"summary": "stopping early"}}],
    ]
    provider = Scripted(plans, base_responses())
    result = run(provider, state(TURNS, SOURCES, ["Fast invoices"]))
    assert agent_actions(result) == ["check_coverage", "check_disclosure", "extract_claims", "judge_claim", "finish"]
    plan_events = [e for e in result["trace"] if e["kind"] == "plan"]
    assert len(plan_events) == 5 and plan_events[0]["actions"][0]["tool"] == "check_coverage"
    assert result["agent"]["planner_modes"] == {"native_tools": 5}
    # The planner sees each observation: the 4th planning prompt includes the extraction result.
    assert "C1" in provider.prompts["plan"][3] and "extract_claims" in provider.prompts["plan"][3]
    # C1 was never judged by the agent, so the guard did it and labelled it.
    guard = [e for e in result["trace"] if e["kind"] == "action" and e["by"] == "guard"]
    assert guard and all(e["note"].startswith("completed by guard") for e in guard)
    assert {"claim_id": "C1", "target": "original"} in [e["args"] for e in guard]


def test_cap_hit_is_logged_and_no_event_is_dropped():
    provider = Scripted(lambda ctx: [{"tool": "search_source", "args": {"query": "fraud"}}] * 3, base_responses())
    result = run(provider, state(TURNS, SOURCES))
    assert result["agent"]["actions"] == 20 and result["agent"]["cap_hit"] is True
    cap = [e for e in result["trace"] if e["kind"] == "cap_reached"]
    assert len(cap) == 1
    assert cap[0]["dropped"] == [{"tool": "search_source", "args": {"query": "fraud"}}]  # 7th plan: 2 ran, 1 dropped and logged
    assert len(agent_actions(result)) == 20
    seqs = [e["seq"] for e in result["trace"]]
    assert seqs == list(range(1, len(seqs) + 1))
    assert len(result["trace"]) > 20
    # The guard then did the real work the agent never reached.
    assert {"extract_claims", "judge_claim", "rewrite_line", "check_disclosure", "check_coverage"} <= set(agent_actions(result, "guard"))
    assert all(c["status"] in ("pass", "fixed", "fail") for c in result["claims"])


class FallbackProvider(Scripted):
    """Uses BaseProvider.plan so the real fallback path runs."""
    plan = BaseProvider.plan

    def __init__(self, responses):
        super().__init__(None, responses)
        self.native_attempts = 0
        self.json_plans = []

    def _native_tools(self, prompt, tools):
        self.native_attempts += 1
        raise ToolsRejected("HTTP 400: tools are not supported for this model")

    def _generate(self, prompt, images, task):
        raise AssertionError("complete_json is scripted")


def test_tool_calling_fallback_to_json_planner():
    steps = iter([
        {"reasoning": "Extract first.", "actions": [{"tool": "extract_claims", "args": {}}]},
        {"reasoning": "Done.", "actions": [{"tool": "finish", "args": {"summary": "ok"}}]},
    ])
    responses = base_responses(plan=lambda ctx, prompt: next(steps))
    provider = FallbackProvider(responses)
    result = run(provider, state(TURNS, SOURCES))
    assert provider.native_attempts == 1  # tried once, then stayed on the JSON planner
    assert result["agent"]["planner_modes"] == {"json_planner": 2}
    fallback = [e for e in result["trace"] if e["kind"] == "planner_fallback"]
    assert len(fallback) == 1 and "HTTP 400" in fallback[0]["reason"]
    assert "Return JSON only" in provider.prompts["plan"][0] and "judge_claim" in provider.prompts["plan"][0]
    assert agent_actions(result) == ["extract_claims", "finish"]


def test_round_two_rewrite_gets_round_one_reason_and_passes():
    round_lines = ["Grabe, Ledgerly AI catches all fraud para sa team.", "Grabe, Ledgerly AI flags unusual charges para ma-review ng team."]

    def rewrite(context, prompt):
        return {"line": round_lines[context["round"] - 1], "chunk_ids": ["S1"]}

    def judge(context, prompt):
        if context["line"] == round_lines[0]:
            return {"verdict": "not_supported", "chunk_id": "S1", "evidence_quote": "flags unusual charges",
                    "reason": "R1-REASON: the source never says it catches all fraud."}
        return judge_by_line(context, prompt)

    provider = Scripted(policy, base_responses(rewrite_line=rewrite, judge_claim=judge))
    result = run(provider, state(TURNS, SOURCES, ["Fast invoices"]))
    claim = result["claims"][0]
    assert [r["round"] for r in claim["rounds"]] == [0, 1, 2]
    assert [r["verdict"] for r in claim["rounds"]] == ["fail", "fail", "pass"]
    assert claim["status"] == "fixed" and claim["suggested_line"] == round_lines[1]
    assert claim["rounds"][2]["chunk_id"] == "S1" and claim["rounds"][2]["citation_verified"] is True
    assert "R1-REASON" in provider.prompts["rewrite_line"][1]
    assert "R1-REASON" not in provider.prompts["rewrite_line"][0]
    assert result["claims"][1]["status"] == "pass"
    assert result["agent"]["guard_actions"] == 0 and result["agent"]["finished_by"] == "agent"


def test_rewrite_that_deletes_the_claim_is_rejected_by_code():
    provider = Scripted(policy, base_responses(rewrite_line={"line": "Thanks for watching!", "chunk_ids": []}))
    result = run(provider, state(TURNS, SOURCES))
    claim = result["claims"][0]
    rewrites = claim["rounds"][1:]
    assert len(rewrites) == 2
    assert all(r["judgement"] == "rejected_by_code" and r["verdict"] == "fail" for r in rewrites)
    assert "deletes the claim" in rewrites[0]["reason"]
    assert claim["status"] == "fail" and claim["suggested_line"] is None
    judged_lines = [p for p in provider.prompts["judge_claim"] if "Thanks for watching" in p]
    assert judged_lines == []  # a rejected rewrite never reaches the judge


def test_rewrite_dropping_taglish_or_losing_the_point_fails():
    english = "Ledgerly AI flags unusual charges for a human reviewer to check."
    provider = Scripted(policy, base_responses(rewrite_line={"line": english, "chunk_ids": ["S1"]}))
    result = run(provider, state(TURNS, SOURCES))
    assert "Taglish" in result["claims"][0]["rounds"][1]["reason"]

    lost = lambda ctx, prompt: (supported("S1", "flags unusual charges", keeps=False) if ctx["target"] == "rewrite" else judge_by_line(ctx, prompt))
    provider = Scripted(policy, base_responses(judge_claim=lost))
    result = run(provider, state(TURNS, SOURCES))
    assert result["claims"][0]["rounds"][1]["judgement"] == "lost_point"
    assert result["claims"][0]["status"] == "fail"


def test_guard_completes_work_the_agent_skipped():
    provider = Scripted([[{"tool": "finish", "args": {"summary": "looks fine"}}]], base_responses())
    result = run(provider, state(TURNS, SOURCES, ["Fast invoices"]))
    assert agent_actions(result) == ["finish"]
    guard = [e for e in result["trace"] if e["kind"] == "action" and e["by"] == "guard"]
    assert [e["tool"] for e in guard] == ["extract_claims", "judge_claim", "rewrite_line", "judge_claim", "judge_claim",
                                          "check_disclosure", "check_coverage"]
    assert all(e["note"].startswith("completed by guard:") for e in guard)
    assert result["agent"]["guard_actions"] == 7
    assert result["claims"][0]["status"] == "fixed" and result["claims"][1]["status"] == "pass"
    assert result["coverage"]["messages"][0]["status"] == "covered"
    guard_check = next(e for e in result["trace"] if e["kind"] == "guard")
    assert "extract_claims" in guard_check["open_work"]


def test_planner_that_stalls_hands_over_to_the_guard():
    provider = Scripted(lambda ctx: [], base_responses())
    result = run(provider, state(TURNS, SOURCES))
    assert any(e["kind"] == "planner_stalled" for e in result["trace"])
    assert result["agent"]["finished_by"] == "guard"
    assert any(e["kind"] == "finish" and e["by"] == "guard" for e in result["trace"])


def test_tool_refusals_are_observations_not_crashes():
    plans = [[{"tool": "rewrite_line", "args": {"claim_id": "C9"}}, {"tool": "delete_everything", "args": {}},
              {"tool": "fetch_source", "args": {"url": "https://example.com/not-in-pack"}}], [{"tool": "finish", "args": {}}]]
    provider = Scripted(plans, base_responses())
    result = run(provider, state(TURNS, SOURCES))
    errors = [e for e in result["trace"] if e["kind"] == "action" and e["by"] == "agent" and e["error"]]
    assert len(errors) == 3
    assert "only URLs listed in the source pack" in errors[2]["observation"]["error"]
    assert "Refused" in provider.prompts["plan"][1] or "unknown tool" in provider.prompts["plan"][1]


@pytest.mark.parametrize("bad", [
    {"verdict": "supported", "chunk_id": "S1", "evidence_quote": "guarantees zero fraud forever"},
    {"verdict": "supported", "chunk_id": "", "evidence_quote": ""},
    {"verdict": "supported", "chunk_id": "R1", "evidence_quote": "An absolute claim"},
])
def test_a_pass_without_a_real_citation_is_downgraded(bad):
    provider = Scripted([[{"tool": "extract_claims", "args": {}}, {"tool": "judge_claim", "args": {"claim_id": "C1", "target": "original"}}]],
                        base_responses(judge_claim=lambda ctx, prompt: dict(bad, reason="trust me")))
    result = run(provider, state(TURNS[:1], SOURCES))
    first = result["claims"][0]["rounds"][0]
    assert first["verdict"] == "fail" and first["judgement"] == "not_supported"
    assert first["checks"]


def test_rewrite_that_changes_the_subject_is_rejected():
    from claimcheck.packs import get_pack
    from claimcheck.tools import rewrite_floor
    original = "It guarantees zero FDA violations, 100%."
    # Seen live on 27 Oct: true to the source, but no longer about FDA risk at all.
    assert any("drops the creator's topic" in p for p in rewrite_floor(original, "It checks creator content before it posts.", get_pack("tech")))
    assert rewrite_floor(original, "It flags possible FDA claim issues for a human reviewer before posting.", get_pack("tech")) == []
