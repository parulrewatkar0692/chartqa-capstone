#!/usr/bin/env python3
"""Step 9: draw the 150-item manual audit sample from the final training split and write it as a CSV
that two reviewers fill in independently (then compute Cohen's kappa with --kappa).

  python3 make_audit_sheet.py --train v1.0/train.jsonl --out audit/audit_sheet.csv
  python3 make_audit_sheet.py --kappa audit/audit_sheet_filled.csv

Each reviewer marks, per item, Y or N for:
  gold_correct        is the gold answer right for this chart input?
  rationale_faithful  does the rationale follow from the evidence and reach the gold answer?
  schema_ok           is the target JSON well formed with the right answer_type and unit?
"""
import argparse
import csv
import json
import random

CHECKS = ("gold_correct", "rationale_faithful", "schema_ok")


def sample(args):
    rows = [json.loads(l) for l in open(args.train, encoding="utf-8")]
    rows = [r for r in rows if r["answer_type"] != "unanswerable"]
    rng = random.Random(args.seed)
    picked = sorted(rng.sample(rows, args.n), key=lambda r: r["id"])
    cols = ["id", "source", "chart_type", "question_type", "question", "target_json", "input_text"]
    cols += [f"{who}_{c}" for who in ("reviewer_A", "reviewer_B") for c in CHECKS] + ["notes"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in picked:
            w.writerow({"id": r["id"], "source": r["source"], "chart_type": r["chart_type"],
                        "question_type": r["question_type"], "question": r["question"],
                        "target_json": r["messages"][2]["content"], "input_text": r["input_text"]})
    print(f"wrote {len(picked)} items to {args.out}")


def kappa(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    for c in CHECKS:
        pairs = [(r[f"reviewer_A_{c}"].strip().upper(), r[f"reviewer_B_{c}"].strip().upper()) for r in rows
                 if r[f"reviewer_A_{c}"].strip() and r[f"reviewer_B_{c}"].strip()]
        if not pairs:
            print(f"{c}: no filled pairs")
            continue
        n = len(pairs)
        po = sum(a == b for a, b in pairs) / n
        pa = sum(a == "Y" for a, _ in pairs) / n
        pb = sum(b == "Y" for _, b in pairs) / n
        pe = pa * pb + (1 - pa) * (1 - pb)
        k = (po - pe) / (1 - pe) if pe < 1 else 1.0
        err = sum(a == "N" or b == "N" for a, b in pairs) / n
        print(f"{c}: n={n} agreement={po:.3f} kappa={k:.3f} flagged_by_either={err:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train")
    ap.add_argument("--out")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kappa", help="filled-in audit CSV")
    args = ap.parse_args()
    if args.kappa:
        kappa(args.kappa)
    else:
        sample(args)


if __name__ == "__main__":
    main()
