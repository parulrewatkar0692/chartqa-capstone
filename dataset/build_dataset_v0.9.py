#!/usr/bin/env python3
"""Build the ChartAudit working dataset (v0.9) from ChartQA human-written questions.

Follows Week1_TrackA_ChartQA_Deliverable.md, steps 1-8, with these fixed decisions:
  * ChartQA-H (human-written) questions only, official splits kept (chart-disjoint).
  * Text-only input: the chart's data table serialised as Markdown, plus series names.
  * Targets: 1,450 train / 170 val / 180 frozen test, ~5% built "unanswerable" items each.
  * Questions that cannot be answered from the text representation are excluded,
    and every exclusion is counted by reason in stats.json.

Usage:
  python3 build_dataset.py --chartqa "/path/to/ChartQA/ChartQA Dataset" --out ./v0.9

Standard library only. Deterministic for a given --seed.
"""
import argparse
import collections
import csv
import hashlib
import io
import json
import os
import random
import re

SPLITS = {"train": 1450, "val": 170, "test": 180}
UNANSWERABLE_RATE = 0.05

SYSTEM_PROMPT = (
    "You are a financial reporting analyst. Answer questions about the chart data provided.\n"
    "Return ONLY valid JSON matching the schema. If the data cannot answer the question,\n"
    'set "answer" to null and explain in "rationale".'
)

SCHEMA_HINT = (
    '{"answer": string | number | null, "answer_type": "number" | "text" | "yes_no" | "unanswerable", '
    '"unit": string | null, "evidence": [string], "rationale": string}'
)

# --------------------------------------------------------------------------- lexicons
COLOUR = re.compile(
    r"\b(red|blue|green|yellow|orange|purple|pink|grey|gray|black|brown|violet|navy|maroon|"
    r"cyan|teal|magenta|turquoise|olive|beige|golden|gold|silver|white|lilac|peach|"
    r"colou?rs?|colou?red)\b", re.I)
STRUCTURE = re.compile(
    r"\b(bars?|lines?|slices?|segments?|sections?|pies?|graphs?|plots?|legends?|leftmost|"
    r"rightmost|topmost|bottom|left|right|top|first|last|tallest|shortest|longest|"
    r"smallest|largest|biggest|highest|lowest|peak|x-axis|y-axis|axis|dotted|dashed)\b", re.I)
ARITH = re.compile(
    r"\b(difference|differ|sum|summation|addition|total|add|added|average|mean|median|ratio|times|divide|divided|"
    r"multipl\w*|product|subtract\w*|minus|plus|more than|less than|greater|fewer|higher than|"
    r"lower than|compare|percent(age)? change|increase|decrease|how many|count|gap|twice|half|"
    r"exceed\w*|combined|altogether|together)\b", re.I)

# Questions about how the chart looks rather than what it says: never answerable from the table.
APPEARANCE = re.compile(
    r"(what|which)\s+(is\s+the\s+)?colou?r|colou?r\s+(is|of|does|represents?|indicates?|denotes?)|"
    r"how many colou?rs|what does the [\w\s-]*colou?r\w*\s+(indicate|represent|denote|show|stand)|"
    r"\bgridlines?\b|\blegend\b|\bdotted\b|\bdashed\b|\btitle\b|\bfont\b|\bshaded?\b", re.I)

DOMAIN = {
    "finance": r"revenue|sales|profit|market share|market|price|stock|investment|invest|bank|debt|budget|"
               r"dollar|euro|usd|\$|€|£|financ|earning|asset|capital|interest rate|loan|credit",
    "economy": r"gdp|inflation|unemploy|trade|export|import|income|spending|spend|tax|wage|salary|"
               r"econom|poverty|employ|labou?r|cost|value added|consumer price|per capita",
    "health":  r"hospital|health|mortality|death|vaccin|insurance|disease|covid|life expectancy|patient",
    "business": r"compan|brand|retail|consumer|advertis|employee|industry|revenue|shipment|"
                r"subscriber|user|store|firm|enterprise|startup|manufactur",
}
DOMAIN_RE = {k: re.compile(v, re.I) for k, v in DOMAIN.items()}

UNIT_PATTERNS = [
    (re.compile(r"\bin (billion|million|thousand|trillion)s? ([\w.\s$€£]+?)(\)|$|,)", re.I), None),
    (re.compile(r"\((%|percent|percentage)\)", re.I), "%"),
    (re.compile(r"\b(share|percentage|percent|rate)\b|%", re.I), "%"),
    (re.compile(r"\bin (u\.s\. dollars|us dollars|dollars|euros|pounds|yen|people|tons|"
                r"metric tons|hours|years|days|minutes|kilograms|kg|km|kilometers|degrees)\b", re.I), None),
]


# --------------------------------------------------------------------------- helpers
def source_of(name):
    if name.startswith(("two_col", "multi_col")):
        return "statista"
    if name.startswith("OECD"):
        return "oecd"
    if re.fullmatch(r"\d{14}", name):
        return "owid"
    return "pew"


def read_table(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        rows = [r for r in csv.reader(f) if any(c.strip() for c in r)]
    if len(rows) < 2:
        return None
    header, body = rows[0], rows[1:]
    width = len(header)
    body = [(r + [""] * width)[:width] for r in body]
    return header, body


def to_number(s):
    s = str(s).strip().replace(",", "").replace("%", "").replace("$", "").replace("€", "").replace("£", "")
    s = s.rstrip(".").strip()
    if s.lower().startswith("*"):
        s = s.lstrip("*")
    try:
        v = float(s)
    except ValueError:
        return None
    return v


def numeric_cells(body):
    return [to_number(c) for r in body for c in r[1:] if to_number(c) is not None]


def norm_text(s):
    s = str(s).lower().strip().rstrip(".")
    s = re.sub(r"[\"'`’*]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def normalise_answer(label):
    raw = str(label).strip()
    low = raw.lower().rstrip(".")
    if low in ("yes", "no"):
        return low, "yes_no"
    v = to_number(raw)
    if v is not None and re.fullmatch(r"[\s\-+\d.,%$€£*]+", raw):
        return (int(v) if v == int(v) and "." not in raw.replace(",", "") else v), "number"
    return norm_text(raw), "text"


def md_cell(c):
    return str(c).replace("|", "/").replace("\n", " ").strip()


def linearise(header, body, source):
    series = header[1:]
    lines = [f"Source: {source.upper() if source in ('owid', 'oecd') else source.capitalize()}"]
    lines.append(f"Category axis: {header[0] or 'Category'}")
    if len(series) == 1:
        single = " (single series"
        single += "; all bars/points share one colour)" if source == "statista" else ")"
        lines.append(f"Series: {series[0]}{single}")
    else:
        lines.append("Series: " + " | ".join(series))
    lines.append("Table:")
    lines.append("| " + " | ".join(md_cell(h) for h in header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for r in body:
        lines.append("| " + " | ".join(md_cell(c) for c in r) + " |")
    return "\n".join(lines)


def infer_unit(header):
    text = " ".join(header[1:])
    for pat, fixed in UNIT_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        if fixed:
            return fixed
        return re.sub(r"\s+", " ", m.group(0)).strip(" ,)").removeprefix("in ").strip()
    return None


def domain_tags(header, body, question):
    blob = " ".join(header) + " " + " ".join(r[0] for r in body) + " " + question
    return sorted(k for k, rx in DOMAIN_RE.items() if rx.search(blob))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- filtering
def classify(item, header, body):
    """Return (exclusion_reason | None, flags dict)."""
    q, src = item["query"], item["source"]
    flags = {
        "mentions_colour": bool(COLOUR.search(q)),
        "mentions_structure": bool(STRUCTURE.search(q)),
        "arithmetic_cue": bool(ARITH.search(q)),
    }
    if flags["arithmetic_cue"]:
        qtype = "arithmetic_comparison"
    elif flags["mentions_colour"] or flags["mentions_structure"]:
        qtype = "visual_reference"
    else:
        qtype = "retrieval"
    flags["question_type"] = qtype
    n_series = len(header) - 1
    flags["n_series"] = n_series

    if APPEARANCE.search(q):
        return "asks_about_chart_appearance", flags
    if flags["mentions_colour"]:
        # Without the annotation JSON we only know colours for single-series Statista bar/line
        # charts (all marks share one colour). Pie charts (values summing to ~100) colour each
        # slice differently, and multi-series charts need the legend, so both are ambiguous.
        vals = numeric_cells(body)
        looks_like_pie = n_series == 1 and vals and 95 <= sum(vals) <= 105
        mark_word = re.search(r"\b(bars?|lines?|graphs?|columns?)\b", q, re.I)
        if not (src == "statista" and n_series == 1 and mark_word and not looks_like_pie):
            return "colour_reference_unresolvable_without_annotations", flags

    ans, atype = item["answer"], item["answer_type"]
    table_text = norm_text(" ".join(header) + " " + " ".join(" ".join(r) for r in body))
    if atype == "text":
        # List answers such as "[United States, Spain]" must have every element in the table.
        parts = [p for p in re.split(r"\s*,\s*", ans.strip("[]")) if p] if ans.startswith("[") else [ans]
        flat = re.sub(r"[^\w ]", "", table_text)
        for a in parts:
            a = norm_text(a)
            if a not in table_text and re.sub(r"[^\w ]", "", a) not in flat:
                return "text_answer_not_in_table", flags
    if atype == "number" and qtype != "arithmetic_comparison":
        # Plain look-up of a number: the gold value must be in the table (within rounding)
        # or be a category label (e.g. a year). Otherwise the table cannot support it.
        vals = numeric_cells(body) + [v for v in (to_number(r[0]) for r in body) if v is not None] \
            + [v for v in (to_number(h) for h in header) if v is not None]
        g = float(ans)
        ok = any(abs(v - g) <= max(0.011, abs(g) * 0.005) for v in vals)
        if not ok:
            return "numeric_lookup_not_in_table", flags
    return None, flags


# --------------------------------------------------------------------------- sampling
def stratified_sample(items, n, rng, keys=("source", "answer_type", "question_type")):
    """Proportional allocation over strata (largest remainder), random within stratum."""
    strata = collections.defaultdict(list)
    for it in items:
        strata[tuple(it[k] for k in keys)].append(it)
    total = len(items)
    quotas = {k: n * len(v) / total for k, v in strata.items()}
    alloc = {k: int(q) for k, q in quotas.items()}
    rest = n - sum(alloc.values())
    for k in sorted(quotas, key=lambda k: (quotas[k] - alloc[k], rng.random()), reverse=True)[:rest]:
        alloc[k] += 1
    out = []
    for k in sorted(strata):
        pool = sorted(strata[k], key=lambda x: x["id"])
        rng.shuffle(pool)
        out += pool[: alloc[k]]
    return sorted(out, key=lambda x: x["id"])


def make_unanswerable(pool_items, tables, n, rng, split, used_ids):
    """Pair a question with a different chart's table that shares no content words with it."""
    stop = set("the a an of in on to for and or is are was were what which how many much value "
               "between by from with does do did this that it its at as be".split())
    charts = sorted(tables)
    candidates = sorted((x for x in pool_items if x["id"] not in used_ids), key=lambda x: x["id"])
    rng.shuffle(candidates)
    out = []
    for it in candidates:
        if len(out) >= n:
            break
        qwords = {w for w in re.findall(r"[a-z]{3,}", it["query"].lower()) if w not in stop}
        qnums = set(re.findall(r"\d+(?:\.\d+)?", it["query"]))
        for _ in range(50):
            other = rng.choice(charts)
            if other == it["chart"]:
                continue
            h, b = tables[other]
            ttext = (" ".join(h) + " " + " ".join(" ".join(r) for r in b)).lower()
            twords = set(re.findall(r"[a-z]{3,}", ttext))
            tnums = set(re.findall(r"\d+(?:\.\d+)?", ttext))
            if qwords & twords or qnums & tnums:
                continue
            out.append((it, other))
            break
    return out


def build_record(split, it, header, body, unanswerable=False, swapped_chart=None):
    src = it["source"] if not unanswerable else source_of(swapped_chart)
    text = linearise(header, body, src)
    user = f"Chart data:\n{text}\n\nQuestion: {it['query']}\n\nRespond with JSON: {SCHEMA_HINT}"
    if unanswerable:
        target = {"answer": None, "answer_type": "unanswerable", "unit": None, "evidence": [],
                  "rationale": "The chart data does not contain the information the question asks about."}
    else:
        answer = it["answer"] if it["answer_type"] == "number" else str(it["label"]).strip().rstrip(".")
        if it["answer_type"] == "yes_no":
            answer = answer.capitalize()
        target = {"answer": answer, "answer_type": it["answer_type"],
                  "unit": infer_unit(header) if it["answer_type"] == "number" else None,
                  "evidence": [], "rationale": None}
    rid = f"{split}-{it['chart']}-{it['qidx']}" + ("-unans" if unanswerable else "")
    return {
        "id": rid,
        "split": split,
        "chart_id": swapped_chart if unanswerable else it["chart"],
        "question_source_chart": it["chart"],
        "source": src,
        "question": it["query"],
        "gold_answer_raw": None if unanswerable else it["label"],
        "answer": target["answer"],
        "answer_norm": None if unanswerable else it["answer"],
        "answer_type": target["answer_type"],
        "question_type": "unanswerable" if unanswerable else it["question_type"],
        "domain_tags": [] if unanswerable else it["domain_tags"],
        "flags": {} if unanswerable else it["flags"],
        "target_status": "fixed_refusal" if unanswerable else "answer_only_pending_teacher_rationale",
        "input_text": text,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
        ],
    }


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chartqa", required=True, help='path to "ChartQA Dataset" folder')
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--domain-filter", choices=["off", "prefer", "strict"], default="prefer",
                    help="strict: keep only domain-matching items; prefer: fill from domain items first")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)

    stats = {"seed": args.seed, "domain_filter": args.domain_filter, "splits": {}}
    manifest = {}
    raw = {}
    table_hash = {}
    for split in SPLITS:
        qa_path = os.path.join(args.chartqa, split, f"{split}_human.json")
        manifest[f"{split}/{split}_human.json"] = sha256(qa_path)
        raw[split] = json.load(open(qa_path, encoding="utf-8"))
        for x in raw[split]:
            name = x["imgname"][:-4]
            p = os.path.join(args.chartqa, split, "tables", name + ".csv")
            if os.path.exists(p) and (split, name) not in table_hash:
                table_hash[(split, name)] = hashlib.md5(open(p, "rb").read()).hexdigest()

    eval_hashes = {h for (s, _), h in table_hash.items() if s != "train"}
    test_hashes = {h for (s, _), h in table_hash.items() if s == "test"}

    pools = {}
    for split in SPLITS:
        reasons = collections.Counter()
        seen = set()
        pool, tables = [], {}
        for qidx, x in enumerate(raw[split]):
            name = x["imgname"][:-4]
            p = os.path.join(args.chartqa, split, "tables", name + ".csv")
            if not os.path.exists(p):
                reasons["orphan_no_table"] += 1
                continue
            key = (name, x["query"].strip().lower())
            if key in seen:
                reasons["duplicate"] += 1
                continue
            seen.add(key)
            q = x["query"].strip()
            if len(q.split()) < 3 or not re.search(r"[A-Za-z]", q):
                reasons["malformed_question"] += 1
                continue
            if str(x["label"]).strip() == "":
                reasons["empty_label"] += 1
                continue
            if name not in tables:
                t = read_table(p)
                tables[name] = t
            t = tables[name]
            if t is None or len(numeric_cells(t[1])) < 2:
                reasons["table_unparseable_or_lt2_cells"] += 1
                continue
            h = table_hash[(split, name)]
            if (split == "train" and h in eval_hashes) or (split == "val" and h in test_hashes):
                reasons["identical_table_in_eval_split"] += 1
                continue
            header, body = t
            ans, atype = normalise_answer(x["label"])
            it = {"id": f"{split}-{name}-{qidx}", "chart": name, "qidx": qidx, "query": q,
                  "label": x["label"], "answer": ans, "answer_type": atype, "source": source_of(name)}
            reason, flags = classify(it, header, body)
            if reason:
                reasons[reason] += 1
                continue
            it["flags"] = flags
            it["question_type"] = flags["question_type"]
            it["domain_tags"] = domain_tags(header, body, q)
            it["domain_match"] = bool(it["domain_tags"])
            pool.append(it)
        tables = {k: v for k, v in tables.items() if v is not None and len(numeric_cells(v[1])) >= 2}
        pools[split] = (pool, tables)
        stats["splits"][split] = {
            "raw_human_pairs": len(raw[split]),
            "excluded": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
            "excluded_total": sum(reasons.values()),
            "answerable_pool": len(pool),
            "answerable_pool_domain_match": sum(it["domain_match"] for it in pool),
        }

    for split, n_total in SPLITS.items():
        pool, tables = pools[split]
        n_unans = round(n_total * UNANSWERABLE_RATE)
        n_ans = n_total - n_unans
        if args.domain_filter == "strict":
            cand = [x for x in pool if x["domain_match"]]
            chosen = stratified_sample(cand, n_ans, rng)
        elif args.domain_filter == "prefer":
            dom = [x for x in pool if x["domain_match"]]
            if len(dom) >= n_ans:
                chosen = stratified_sample(dom, n_ans, rng)
            else:
                rest = [x for x in pool if not x["domain_match"]]
                chosen = dom + stratified_sample(rest, n_ans - len(dom), rng)
        else:
            chosen = stratified_sample(pool, n_ans, rng)
        used = {x["id"] for x in chosen}
        unans = make_unanswerable(pool, tables, n_unans, rng, split, used)

        records = [build_record(split, it, *tables[it["chart"]]) for it in chosen]
        records += [build_record(split, it, *tables[other], unanswerable=True, swapped_chart=other)
                    for it, other in unans]
        rng.shuffle(records)
        with open(os.path.join(args.out, f"{split}.jsonl"), "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        s = stats["splits"][split]
        s["final_total"] = len(records)
        s["final_answerable"] = len(chosen)
        s["final_unanswerable"] = len(unans)
        s["final_charts"] = len({r["chart_id"] for r in records})
        for key in ("source", "answer_type", "question_type"):
            s[f"final_by_{key}"] = dict(collections.Counter(r[key] for r in records).most_common())
        s["final_domain_match"] = sum(bool(r["domain_tags"]) for r in records)
        s["final_numeric_with_unit"] = sum(
            1 for r in records if r["answer_type"] == "number" and json.loads(r["messages"][2]["content"])["unit"])

    # Leakage checks across the final files.
    ids = {}
    for split in SPLITS:
        ids[split] = {json.loads(l)["chart_id"] for l in open(os.path.join(args.out, f"{split}.jsonl"))}
    stats["chart_overlap"] = {
        "train_val": len(ids["train"] & ids["val"]),
        "train_test": len(ids["train"] & ids["test"]),
        "val_test": len(ids["val"] & ids["test"]),
    }
    for split in SPLITS:
        manifest[f"OUTPUT {split}.jsonl"] = sha256(os.path.join(args.out, f"{split}.jsonl"))
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(args.out, "MANIFEST.txt"), "w") as f:
        for k, v in manifest.items():
            f.write(f"{v}  {k}\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
