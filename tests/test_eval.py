from eval import run_eval
from claimcheck.fixture import FixtureProvider


def test_eval_runs_on_the_fictional_sample(tmp_path):
    out = tmp_path / "results.json"
    payload = run_eval.run("eval/sample.csv", out=out, provider=FixtureProvider(step_delay=0))
    assert len(payload["results"]) == 8 and out.exists()
    b1 = payload["summary"][run_eval.LABELS[0]]
    assert b1["recall"] == 1.0
    assert payload["summary"][run_eval.LABELS[1]]["confusion"]["tp"] == 3


def test_ads_eval_checks_quotes_and_scores_reference_labels(tmp_path):
    import csv
    from eval import run_ads_eval
    from helpers import Scripted
    path = tmp_path / "ads.csv"
    rows = [
        ["1", "supplements", "Fictional tea. Gamot sa diabetes, no need opera!", "yes", "not a creator post"],
        ["2", "beauty", "Fictional cream for soft skin. Paid partnership with Glowa.", "no", "shown"],
        ["3", "loans", "Fictional loan, guaranteed approval in 5 minutes!", "yes", "not shown"],
        ["4", "gaming", "Fictional game, play with friends.", "no", "not a creator post"],
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row", "category", "ad_text", "B1_drug_like_or_unproven_claim", "D1_paid_disclosure_creator"])
        writer.writerows(rows)
    answers = {
        "1": {"b1": "no", "b1_quote": ""},  # model misses; the FDA hard rule catches "gamot sa"
        "2": {"b1": "no", "creator_post": True, "disclosed": True, "disclosure_quote": "Paid partnership with Glowa"},
        "3": {"b1": "yes", "b1_quote": "guaranteed approval in 5 minutes", "creator_post": True, "disclosed": False},
        "4": {"b1": "yes", "b1_quote": "win 10,000 pesos daily"},  # invented quote: not counted
    }
    provider = Scripted(responses={"screen_ad": lambda ctx, prompt: answers[ctx["row"]["row"]]})
    payload = run_ads_eval.run(str(path), out=tmp_path / "out.json", provider=provider, workers=1)
    by_row = {r["row"]: r for r in payload["results"]}
    assert by_row["1"]["pred_B1"] == "yes" and by_row["1"]["pred_B1_model_only"] == "no"
    assert by_row["4"]["pred_B1"] == "no" and by_row["4"]["checks"]
    assert payload["summary"]["B1"]["agreement"] == 1.0
    d1 = payload["summary"]["D1_creator_posts_only"]
    assert d1["n"] == 2 and d1["confusion"] == {"tp": 1, "fp": 0, "fn": 0, "tn": 1}
    assert payload["cost"]["calls_per_ad"] == 1.0
    assert "not human" in payload["summary"]["labels"]
