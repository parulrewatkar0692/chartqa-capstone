#!/usr/bin/env python3
"""Step 7: add teacher (Claude Opus) evidence + rationales to the v1.0 train/val splits.

The teacher sees each item's input, question and gold answer and returns
{"answer", "evidence", "rationale", "teacher_agrees"}. teacher/check_teacher.py verifies every output:
schema, answer == gold (relaxed), every evidence string points at real content of the input,
rationale <= 40 words, and teacher_agrees is true. The gold answer is never changed.

Items the teacher disputes (it judges the gold answer wrong or unsupported by the input) are dropped
and replaced by an unused item from the same split and stratum (source x chart type x answer type x
question type), drawn from the build's answerable pool. Replacements go through the same teacher and
checker; a disputed replacement is skipped in favour of the next candidate.

Usage:
  # 1. answer-only build + pool dump
  python3 build_dataset.py --chartqa "ChartQA Dataset" --out v1.0/selection --pool-out pool.json
  # 2. (teacher labels v1.0 items -> teacher/outputs/*.jsonl)
  # 3. list replacement candidates for disputed items -> teacher/batches/replace_<split>.jsonl
  python3 add_teacher_labels.py select --data v1.0/selection --pool pool.json --teacher teacher
  # 4. (teacher labels the replacement batches -> teacher/outputs/replace_<split>*.jsonl)
  # 5. merge into the final training files
  python3 add_teacher_labels.py merge --data v1.0/selection --pool pool.json --teacher teacher

Deterministic for a given --seed.
"""
import argparse
import collections
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_dataset as B  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "teacher"))
import check_teacher as C  # noqa: E402

SPLITS = ("train", "val")
OVERSAMPLE = 1.6   # candidates per disputed item (replacements can be disputed too)
KEYS = ("source", "chart_type", "answer_type", "question_type")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


UNRELIABLE = "mark_colours_unreliable"
MARK_LINES = ("Bar colours", "Slice colours", "Point colours")


def unreliable(rec, pool, split):
    """The chart's per-bar/per-slice colour list can't be trusted (build_dataset.mark_colours_unreliable:
    pies with repeated slice labels, or OECD-style bar annotations whose labels and values are out of order)."""
    return pool[split]["charts"][rec["chart_id"]][UNRELIABLE]


def repair(rec, pool, split):
    """Remove untrustworthy per-mark colour lines from the input (input_text and user turn)."""
    if not unreliable(rec, pool, split):
        return rec
    text, msgs = rec["input_text"], [dict(m) for m in rec["messages"]]
    for line in [l for l in text.split("\n") if l.startswith(MARK_LINES)]:
        text = text.replace(line + "\n", "")
        msgs[1]["content"] = msgs[1]["content"].replace(line + "\n", "")
    return dict(rec, input_text=text, messages=msgs, flags=dict(rec["flags"], mark_colours_removed=True))


def colour_dependent(rec, out):
    return bool(rec["flags"].get("mentions_colour")) or any(
        str(e).startswith("Colour:") for e in (out or {}).get("evidence", []))


def batch_item(rec):
    return {"id": rec["id"], "question": rec["question"], "gold_answer": rec["answer"],
            "answer_type": rec["answer_type"], "unit": json.loads(rec["messages"][2]["content"])["unit"],
            "input_text": rec["input_text"]}


def load_teacher(teacher_dir, pattern):
    out = {}
    for p in sorted(glob.glob(os.path.join(teacher_dir, "outputs", pattern))):
        for o in read_jsonl(p):
            out[o["id"]] = o
    return out


def verdicts(items, outs):
    """id -> (ok, reasons) using the same checker the teacher ran."""
    res = {}
    for it in items:
        o = outs.get(it["id"])
        if o is None:
            res[it["id"]] = (False, ["missing teacher output"])
            continue
        errs = C.check(it, o)
        res[it["id"]] = (not errs, errs)
    return res


def pool_records(pool, split):
    """Build v1.0-format records for every pool item (answer-only)."""
    recs = {}
    for it in pool[split]["items"]:
        ch = pool[split]["charts"][it["chart"]]
        recs[it["id"]] = B.build_record(split, it, ch["header"], None, ch["text"], it["chart_type"])
    return recs


def candidates(split, data, pool, disputed, seed, exclude=(), need=None, round_=1):
    rng = random.Random(f"{seed}-{split}" + (f"-round{round_}" if round_ > 1 else ""))
    used = {r["id"].removesuffix("-unans") for r in data} | set(exclude)
    # ("<id>-unans" items reuse pool question <id>, so stripping the suffix keeps those out too.)
    by_stratum = collections.defaultdict(list)
    for it in sorted(pool[split]["items"], key=lambda x: x["id"]):
        if it["domain_match"] and it["id"] not in used:
            by_stratum[tuple(it[k] for k in KEYS)].append(it)
    for v in by_stratum.values():
        rng.shuffle(v)
    taken, plan = set(), []
    if need is None:
        dropped = [r for r in data if r["id"] in disputed]
        need = collections.Counter(tuple(r[k] for k in KEYS) for r in dropped)
    for stratum, n in sorted(need.items()):
        want = int(n * OVERSAMPLE + 0.999) + 1
        got = [x for x in by_stratum.get(stratum, []) if x["id"] not in taken][:want]
        if len(got) < want:   # thin stratum: widen to source x answer type
            wider = [x for k, v in sorted(by_stratum.items()) if k[0] == stratum[0] and k[2] == stratum[2]
                     for x in v if x["id"] not in taken and x not in got]
            got += wider[: want - len(got)]
        for rank, x in enumerate(got):
            taken.add(x["id"])
            plan.append({"stratum": list(stratum), "rank": rank, "id": x["id"], "round": round_})
    return plan


def cmd_select(args):
    pool = json.load(open(args.pool, encoding="utf-8"))
    for split in SPLITS:
        data = read_jsonl(os.path.join(args.data, f"{split}.jsonl"))
        items = [batch_item(r) for r in data if r["answer_type"] != "unanswerable"]
        v = verdicts(items, load_teacher(args.teacher, f"{split}_*.jsonl"))
        bad = {i for i, (ok, _) in v.items() if not ok}
        plan = candidates(split, data, pool, bad, args.seed)
        recs = pool_records(pool, split)
        write_jsonl(os.path.join(args.teacher, "batches", f"replace_{split}.jsonl"),
                    [batch_item(recs[p["id"]]) for p in plan])
        write_jsonl(os.path.join(args.teacher, f"replace_plan_{split}.jsonl"), plan)
        print(f"{split}: {len(bad)} rejected by teacher/checker, {len(plan)} replacement candidates")


def cmd_topup(args):
    """Second round: more candidates for strata whose round-1 candidates were all rejected
    (reads teacher/unreplaced_<split>.jsonl written by merge; appends to the plan)."""
    pool = json.load(open(args.pool, encoding="utf-8"))
    for split in SPLITS:
        data = read_jsonl(os.path.join(args.data, f"{split}.jsonl"))
        plan_path = os.path.join(args.teacher, f"replace_plan_{split}.jsonl")
        plan = read_jsonl(plan_path)
        lost = read_jsonl(os.path.join(args.teacher, f"unreplaced_{split}.jsonl"))
        if not lost:
            continue
        round_ = max(p.get("round", 1) for p in plan) + 1
        need = collections.Counter(tuple(l["stratum"]) for l in lost)
        new = candidates(split, data, pool, set(), args.seed, exclude={p["id"] for p in plan}, need=need, round_=round_)
        recs = pool_records(pool, split)
        new = [p for p in new if not (unreliable(recs[p["id"]], pool, split) and colour_dependent(recs[p["id"]], None))]
        write_jsonl(plan_path, plan + new)
        write_jsonl(os.path.join(args.teacher, "batches", f"replace_{split}_r{round_}.jsonl"),
                    [batch_item(recs[p["id"]]) for p in new])
        print(f"{split}: round {round_}, {sum(need.values())} slots, {len(new)} candidates")


def labelled(rec, out):
    target = json.loads(rec["messages"][2]["content"])
    target["evidence"] = out["evidence"]
    target["rationale"] = out["rationale"]
    rec = dict(rec, target_status="teacher_verified",
               messages=rec["messages"][:2] + [{"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}])
    return rec


def answer_only(rec):
    target = json.loads(rec["messages"][2]["content"])
    if rec["answer_type"] != "unanswerable":
        target["evidence"], target["rationale"] = [], None
    return dict(rec, messages=rec["messages"][:2] + [{"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}])


def cmd_merge(args):
    pool = json.load(open(args.pool, encoding="utf-8"))
    report = {"teacher": "Claude Opus", "splits": {}}
    final_dir = args.out
    os.makedirs(os.path.join(final_dir, "answer_only"), exist_ok=True)
    rejected_log = []
    for split in SPLITS:
        data = read_jsonl(os.path.join(args.data, f"{split}.jsonl"))
        outs = load_teacher(args.teacher, f"{split}_*.jsonl")
        outs.update(load_teacher(args.teacher, f"replace_{split}*.jsonl"))
        items = [batch_item(r) for r in data if r["answer_type"] != "unanswerable"]
        v = verdicts(items, outs)
        recs = pool_records(pool, split)
        # Items that depend on an untrustworthy per-mark colour mapping are rejected like a failed check.
        for r in list(data) + list(recs.values()):
            if r["id"] in v or r["id"] in outs:
                if unreliable(r, pool, split) and colour_dependent(r, outs.get(r["id"])):
                    v[r["id"]] = (False, [UNRELIABLE])
        plan = read_jsonl(os.path.join(args.teacher, f"replace_plan_{split}.jsonl"))
        cand_v = verdicts([batch_item(recs[p["id"]]) for p in plan], outs)
        for p in plan:
            if v.get(p["id"]) == (False, [UNRELIABLE]):
                cand_v[p["id"]] = v[p["id"]]
        queue = collections.defaultdict(list)
        for p in plan:
            queue[tuple(p["stratum"])].append(p["id"])
        final, n_rep, n_lost, lost = [], 0, 0, []
        for r in data:
            if r["answer_type"] == "unanswerable":
                final.append(repair(r, pool, split))
                continue
            ok, why = v[r["id"]]
            if ok:
                final.append(repair(labelled(r, outs[r["id"]]), pool, split))
                continue
            rejected_log.append({"split": split, "id": r["id"], "question": r["question"], "gold": r["answer"],
                                 "reasons": why, "note": outs.get(r["id"], {}).get("note")})
            q = queue[tuple(r[k] for k in KEYS)]
            while q and not cand_v[q[0]][0]:
                c = q.pop(0)
                rejected_log.append({"split": split, "id": c, "question": recs[c]["question"], "gold": recs[c]["answer"],
                                     "reasons": cand_v[c][1], "note": outs.get(c, {}).get("note"), "replacement": True})
            if q:
                c = q.pop(0)
                final.append(dict(repair(labelled(recs[c], outs[c]), pool, split), replaces=r["id"]))
                n_rep += 1
            else:
                n_lost += 1
                lost.append(list(tuple(r[k] for k in KEYS)))
        write_jsonl(os.path.join(args.teacher, f"unreplaced_{split}.jsonl"), [{"stratum": l} for l in lost])
        # Keep the original order (records were shuffled at build time); replacements sit where the
        # item they replace was.
        write_jsonl(os.path.join(final_dir, f"{split}.jsonl"), final)
        write_jsonl(os.path.join(final_dir, "answer_only", f"{split}.jsonl"), [answer_only(r) for r in final])
        n_items = sum(1 for r in data if r["answer_type"] != "unanswerable")
        orig = [r for r in data if r["answer_type"] != "unanswerable"]
        n_bad = sum(1 for r in orig if not v[r["id"]][0])
        by_src = collections.Counter(r["source"] for r in orig if not v[r["id"]][0])
        by_reason = collections.Counter(UNRELIABLE if v[r["id"]][1] == [UNRELIABLE]
                                        else "teacher_disputes_gold" for r in orig if not v[r["id"]][0])
        report["splits"][split] = {
            "labelled_by_teacher": n_items + len(plan),
            "original_items": n_items,
            "original_rejected": n_bad,
            "original_rejection_rate": round(n_bad / n_items, 4),
            "original_rejected_by_source": dict(by_src.most_common()),
            "original_rejected_by_reason": dict(by_reason.most_common()),
            "replacement_candidates_labelled": len(plan),
            "replacement_candidates_rejected": sum(1 for ok, _ in cand_v.values() if not ok),
            "replaced": n_rep,
            "not_replaced": n_lost,
            "final_total": len(final),
            "final_teacher_verified": sum(1 for r in final if r["target_status"] == "teacher_verified"),
            "final_unanswerable": sum(1 for r in final if r["answer_type"] == "unanswerable"),
            "final_by_source": dict(collections.Counter(r["source"] for r in final).most_common()),
            "final_by_chart_type": dict(collections.Counter(r["chart_type"] for r in final).most_common()),
            "final_by_answer_type": dict(collections.Counter(r["answer_type"] for r in final).most_common()),
            "final_by_question_type": dict(collections.Counter(r["question_type"] for r in final).most_common()),
        }
    # The test split is not labelled (it is scored on the answer only). It gets the same colour repair:
    # colour questions on charts with an untrustworthy per-mark colour list are swapped for an unused
    # test-pool item from the same stratum.
    test, report["splits"]["test"] = repair_test(read_jsonl(os.path.join(args.data, "test.jsonl")), pool, args.seed)
    write_jsonl(os.path.join(final_dir, "test.jsonl"), test)
    write_jsonl(os.path.join(final_dir, "answer_only", "test.jsonl"), test)
    write_jsonl(os.path.join(args.teacher, "rejected.jsonl"), rejected_log)
    with open(os.path.join(final_dir, "MANIFEST.txt"), "w") as f:
        for rel in ["train.jsonl", "val.jsonl", "test.jsonl", "answer_only/train.jsonl",
                    "answer_only/val.jsonl", "answer_only/test.jsonl"]:
            f.write(f"{B.sha256(os.path.join(final_dir, rel))}  {rel}\n")
    with open(os.path.join(final_dir, "teacher_stats.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


def repair_test(data, pool, seed):
    split = "test"
    rng = random.Random(f"{seed}-test-repair")
    recs = pool_records(pool, split)
    used = {r["id"].removesuffix("-unans") for r in data}
    by_stratum = collections.defaultdict(list)
    for it in sorted(pool[split]["items"], key=lambda x: x["id"]):
        rec = recs[it["id"]]
        if it["domain_match"] and it["id"] not in used and not (unreliable(rec, pool, split) and colour_dependent(rec, None)):
            by_stratum[tuple(it[k] for k in KEYS)].append(it["id"])
    for v in by_stratum.values():
        rng.shuffle(v)
    out, swapped = [], []
    for r in data:
        if r["answer_type"] != "unanswerable" and unreliable(r, pool, split) and colour_dependent(r, None):
            q = by_stratum[tuple(r[k] for k in KEYS)]
            if q:
                c = q.pop(0)
                out.append(dict(repair(recs[c], pool, split), replaces=r["id"]))
                swapped.append({"removed": r["id"], "added": c})
                continue
        out.append(repair(r, pool, split))
    return out, {"items": len(out), "colour_items_swapped": len(swapped), "swaps": swapped,
                 "inputs_with_mark_colours_removed": sum(1 for r in out if r["flags"].get("mark_colours_removed"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "merge", "topup"])
    ap.add_argument("--data", required=True, help="build_dataset.py output: answer-only {train,val,test}.jsonl")
    ap.add_argument("--out", help="merge: folder for the final files (default: parent of --data)")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--teacher", required=True, help="teacher folder (batches/, outputs/)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.out is None:
        args.out = os.path.dirname(os.path.abspath(args.data))
    {"select": cmd_select, "merge": cmd_merge, "topup": cmd_topup}[args.cmd](args)


if __name__ == "__main__":
    main()
