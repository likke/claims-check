from eval import run_eval
from claimcheck.fixture import FixtureProvider


def test_eval_runs_on_the_fictional_sample(tmp_path):
    out = tmp_path / "results.json"
    payload = run_eval.run("eval/sample.csv", out=out, provider=FixtureProvider(step_delay=0))
    assert len(payload["results"]) == 8 and out.exists()
    b1 = payload["summary"][run_eval.LABELS[0]]
    assert b1["recall"] == 1.0
    assert payload["summary"][run_eval.LABELS[1]]["confusion"]["tp"] == 3
