#!/usr/bin/env python3
"""Evaluate fictional or privately supplied labeled ad copy."""
import argparse, csv, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app

LABELS = ["B1_drug_like_or_unproven_claim", "D1_paid_disclosure_creator"]

def metrics(rows, label, prediction_key):
    tp=fp=fn=tn=agree=0
    for row in rows:
        actual=row[label].strip().lower(); pred=str(row[prediction_key]).lower()
        pred="yes" if pred in ("yes","true","pass") else "no" if pred in ("no","false","fail") else "unclear"
        if pred==actual: agree+=1
        if actual=="yes" and pred=="yes": tp+=1
        elif actual=="no" and pred=="yes": fp+=1
        elif actual=="yes" and pred=="no": fn+=1
        elif actual=="no" and pred=="no": tn+=1
    return {"precision":tp/(tp+fp) if tp+fp else 0,"recall":tp/(tp+fn) if tp+fn else 0,"agreement":agree/len(rows) if rows else 0,"confusion":{"tp":tp,"fp":fp,"fn":fn,"tn":tn,"unclear":sum(str(r[prediction_key]).lower()=="unclear" for r in rows)}}

def run(path, limit=None, out=None, provider=None):
    provider=provider or app.get_provider()
    with open(path,newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
    if limit is not None: rows=rows[:limit]
    results=[]
    for row in rows:
        text=row["ad_text"]
        claim=provider.judge_claim(text,[text])
        disclosure=provider.check_disclosure(text,[])
        results.append({**row,"pred_B1":"yes" if claim.get("verdict")=="fail" else "no","pred_D1":"yes" if disclosure else "no","judge":claim})
    summary={LABELS[0]:metrics(results,LABELS[0],"pred_B1"),LABELS[1]:metrics(results,LABELS[1],"pred_D1")}
    payload={"summary":summary,"results":results}
    if out: Path(out).write_text(json.dumps(payload,indent=2),encoding="utf-8")
    for label,data in summary.items():
        print(f"{label}: precision={data['precision']:.3f} recall={data['recall']:.3f} agreement={data['agreement']:.3f}")
        print("confusion: "+json.dumps(data["confusion"],sort_keys=True))
    return payload

if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--csv",required=True); parser.add_argument("--limit",type=int); parser.add_argument("--out",required=True)
    args=parser.parse_args(); run(args.csv,args.limit,args.out)
