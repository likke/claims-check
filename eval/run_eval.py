#!/usr/bin/env python3
"""Score the claim judge and the disclosure check against labelled ad lines.

Each row is judged as one claim under the FDA beauty pack, with a source pack
that only says the product is fictional. B1 (drug-like or unproven claim) is
predicted when the claim fails; D1 (paid disclosure) when check_disclosure
finds one. Uses the configured provider (fixture offline, Meta with a key).
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from claimcheck.agent import build_state  # noqa: E402
from claimcheck.packs import get_pack  # noqa: E402
from claimcheck.providers import get_provider  # noqa: E402
from claimcheck.state import Claim  # noqa: E402
from claimcheck.tools import ToolBox  # noqa: E402

LABELS = ["B1_drug_like_or_unproven_claim", "D1_paid_disclosure_creator"]


def metrics(rows, label, key):
    tp = fp = fn = tn = agree = 0
    for row in rows:
        actual, predicted = row[label].strip().lower(), row[key]
        agree += predicted == actual
        tp += actual == "yes" and predicted == "yes"
        fp += actual == "no" and predicted == "yes"
        fn += actual == "yes" and predicted == "no"
        tn += actual == "no" and predicted == "no"
    return {"precision": tp / (tp + fp) if tp + fp else 0, "recall": tp / (tp + fn) if tp + fn else 0,
            "agreement": agree / len(rows) if rows else 0,
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "unclear_labels": sum(r[label].strip().lower() == "unclear" for r in rows)}}


def run(path, limit=None, out=None, provider=None):
    provider = provider or get_provider()
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))[:limit]
    results = []
    for row in rows:
        text = row["ad_text"]
        state = build_state({"turns": [{"start": 0.0, "end": 5.0, "text": text}], "duration": 5.0, "timing": "asr"},
                            [], "The product in this line is fictional.", [], get_pack("FDA beauty"))
        state.claims.append(Claim(id="C1", quote=text, start=0.0, end=5.0, estimated=False, origin="speech"))
        state.claims_extracted = True
        tools = ToolBox(state, provider)
        verdict, _ = tools.run("judge_claim", {"claim_id": "C1", "target": "original"})
        tools.run("check_disclosure", {})
        results.append({**row, "pred_B1": "yes" if verdict.get("verdict") == "fail" and _risky(verdict, text) else "no",
                        "pred_D1": "yes" if state.disclosure["disclosed"] else "no", "judge": verdict})
    summary = {LABELS[0]: metrics(results, LABELS[0], "pred_B1"), LABELS[1]: metrics(results, LABELS[1], "pred_D1")}
    payload = {"summary": summary, "results": results}
    if out:
        Path(out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for label, data in summary.items():
        print(f"{label}: precision={data['precision']:.3f} recall={data['recall']:.3f} agreement={data['agreement']:.3f} {json.dumps(data['confusion'])}")
    return payload


RISKY = re.compile(r"\b(guarantee\w*|every|forever|cure\w*|treat\w*|heal\w*|removes?|certainty|100%)\b", re.I)


def _risky(verdict, line):
    """A failed line counts as B1 when a pack rule fired or the line makes a drug-like or absolute promise."""
    return verdict.get("judgement") in ("rule", "contradicted") or bool(RISKY.search(line))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args.csv, args.limit, args.out)
