#!/usr/bin/env python3
"""Build the ChartAudit working dataset (v1.0) from ChartQA human-written questions.

Follows Week1_TrackA_ChartQA_Deliverable.md, steps 1-8, with these fixed decisions:
  * ChartQA-H (human-written) questions only, official splits kept (chart-disjoint).
  * Text-only input: the chart's data table serialised as Markdown, plus the visual
    metadata from the ChartQA annotation JSON (chart type, title, series and mark colours).
  * Targets: 1,450 train / 170 val / 180 frozen test, ~5% built "unanswerable" items each.
  * Questions that cannot be answered from the text representation are excluded,
    and every exclusion is counted by reason in stats.json.

Changes from v0.9: annotations are used (v0.9 had tables only), colour questions are kept
whenever every colour they name resolves to exactly one mark in the annotation, "what colour"
and title questions are kept when the serialisation contains the answer, chart type is a
stratification key, and the domain lexicon also runs on chart titles and axis labels.

Source: "ChartQA Dataset.zip" from huggingface.co/datasets/ahmed-masry/ChartQA (it contains
*_human.json, tables/ and annotations/ for each split).

Usage:
  python3 build_dataset.py --chartqa "/path/to/ChartQA Dataset" --out ./v1.0

Standard library only. Deterministic for a given --seed.
"""
import argparse
import collections
import colorsys
import csv
import difflib
import hashlib
import json
import os
import random
import re

VERSION = "1.0"
SPLITS = {"train": 1450, "val": 170, "test": 180}
UNANSWERABLE_RATE = 0.05
STRATA = ("source", "chart_type", "answer_type", "question_type")

SYSTEM_PROMPT = (
    "You are a financial reporting analyst. Answer questions about the chart data provided.\n"
    "Return ONLY valid JSON matching the schema. If the data cannot answer the question,\n"
    'set "answer" to null and explain in "rationale".'
)

SCHEMA_HINT = (
    '{"answer": string | number | null, "answer_type": "number" | "text" | "yes_no" | "unanswerable", '
    '"unit": string | null, "evidence": [string], "rationale": string}'
)

CHART_TYPE_TEXT = {"v_bar": "vertical bar", "h_bar": "horizontal bar", "line": "line", "pie": "pie"}

# --------------------------------------------------------------------------- lexicons
COLOUR_WORDS = (r"red|blue|green|yellow|orange|purple|pink|grey|gray|black|brown|violet|navy|maroon|"
                r"cyan|teal|magenta|turquoise|olive|beige|golden|gold|silver|white|lilac|peach|lime|"
                r"tan|khaki|crimson|scarlet|mustard")
COLOUR = re.compile(r"\b(" + COLOUR_WORDS + r"|colou?rs?|colou?red)\b", re.I)
STRUCTURE = re.compile(
    r"\b(bars?|lines?|slices?|segments?|sections?|pies?|graphs?|plots?|legends?|leftmost|"
    r"rightmost|topmost|bottom|left|right|top|first|last|tallest|shortest|longest|"
    r"smallest|largest|biggest|highest|lowest|peak|x-axis|y-axis|axis|dotted|dashed)\b", re.I)
ARITH = re.compile(
    r"\b(difference|differ|sum|summation|addition|total|add|added|average|mean|median|ratio|times|divide|divided|"
    r"multipl\w*|product|subtract\w*|minus|plus|more than|less than|greater|fewer|higher than|"
    r"lower than|compare|percent(age)? change|increase|decrease|how many|count|gap|twice|half|"
    r"exceed\w*|combined|altogether|together)\b", re.I)

# A colour phrase: optional modifier + colour word ("light blue", "navy blue", "navy", "maroon").
COLOUR_PHRASE = re.compile(
    r"\b(?:(light|dark|navy|sky|pale|bright|deep|lighter|darker)[\s-]+)?(" + COLOUR_WORDS + r")\b", re.I)
# Questions whose answer is a colour, or that ask how many colours there are.
COLOUR_ASK = re.compile(
    r"(what|which)\s+(is\s+the\s+)?colou?r|colou?r\s+(is|of|does)|how many colou?rs", re.I)
TITLE_ASK = re.compile(r"\btitle\b", re.I)
LEGEND = re.compile(r"\blegend\b", re.I)
# Still unanswerable from the serialisation even with annotations.
APPEARANCE_OTHER = re.compile(r"\bgridlines?\b|\bdotted\b|\bdashed\b|\bfont\b|\bshaded?\b|\bshades?\b", re.I)

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

# --------------------------------------------------------------------------- colours
# Colour words people use in questions -> (canonical family, implied modifier).
WORD_FAMILY = {
    "red": ("red", None), "crimson": ("red", None), "scarlet": ("red", None), "maroon": ("red", "dark"),
    "orange": ("orange", None), "peach": ("orange", "light"),
    "yellow": ("yellow", None), "gold": ("yellow", None), "golden": ("yellow", None), "mustard": ("yellow", None),
    "green": ("green", None), "lime": ("green", "light"), "olive": ("green", "dark"),
    "teal": ("teal", None), "turquoise": ("teal", None), "cyan": ("teal", None),
    "blue": ("blue", None), "navy": ("blue", "dark"),
    "purple": ("purple", None), "violet": ("purple", None), "lilac": ("purple", "light"),
    "pink": ("pink", None), "magenta": ("pink", None),
    "brown": ("brown", None), "beige": ("brown", "light"), "tan": ("brown", "light"), "khaki": ("brown", "light"),
    "grey": ("grey", None), "gray": ("grey", None), "silver": ("grey", "light"),
    "black": ("black", None), "white": ("white", None),
}
MODIFIER = {"light": "light", "lighter": "light", "sky": "light", "pale": "light", "bright": "light",
            "dark": "dark", "darker": "dark", "navy": "dark", "deep": "dark"}
# Families a reader might plausibly use for a colour of the key family (used only when no
# colour in the chart is a direct match).
NEIGHBOURS = {
    "red": {"pink", "brown", "orange"}, "orange": {"red", "brown", "yellow"}, "yellow": {"orange"},
    "green": {"teal"}, "teal": {"green", "blue"}, "blue": {"teal", "purple"}, "purple": {"pink", "blue"},
    "pink": {"purple", "red"}, "brown": {"red", "orange"}, "grey": {"black", "white"},
    "black": {"grey"}, "white": {"grey"},
}
PEW_TYPOS = {"dark drown": "dark brown", "light brownn": "light brown", "dark brownn": "dark brown",
             "light brow": "light brown", "brow": "brown", "dark greenn": "dark green",
             "light brown.": "light brown", "hrey": "grey", "orannnge": "orange", "orage": "orange",
             "lemon": "yellow"}


def parse_hex(c):
    c = str(c).strip().lower()
    m = re.fullmatch(r"rgb\((\d+),\s*(\d+),\s*(\d+)\)", c)
    if m:
        return tuple(int(x) for x in m.groups())
    m = re.fullmatch(r"#([0-9a-f]{6})", c)
    if m:
        return tuple(int(m.group(1)[i:i + 2], 16) for i in (0, 2, 4))
    return None


def describe_colour(raw):
    """Map an annotation colour (hex, rgb() or a Pew colour name) to
    {name, family, lightness, extra}. Returns None for unknown colours (Pew "UNK")."""
    s = str(raw).strip().lower()
    s = PEW_TYPOS.get(s, s)
    rgb = parse_hex(s)
    if rgb is None:
        m = re.fullmatch(r"(?:(light|dark)\s+)?([a-z]+)", s)
        if not m or m.group(2) not in WORD_FAMILY:
            return None
        fam, implied = WORD_FAMILY[m.group(2)]
        mod = m.group(1) or implied
        light = {"light": 0.72, None: 0.5, "dark": 0.28}[mod]
        return {"name": (mod + " " if mod else "") + fam, "family": fam, "lightness": light, "extra": set()}
    r, g, b = (x / 255 for x in rgb)
    h, l, sat = colorsys.rgb_to_hls(r, g, b)
    h *= 360
    extra = set()
    if l >= 0.93:
        fam, mod = "white", None
    elif l <= 0.12 or (sat < 0.15 and l < 0.2):
        fam, mod = "black", None
    elif sat < 0.15:
        fam = "grey"
        mod = "light" if l >= 0.68 else ("dark" if l < 0.38 else None)
    else:
        if h < 18 or h >= 345:
            fam = "red"
        elif h < 40:
            fam = "brown" if l < 0.3 else "orange"
        elif h < 70:
            fam = "yellow"
        elif h < 165:
            fam = "green"
        elif h < 192:
            fam = "teal"
        elif h < 258:
            fam = "blue"
        elif h < 300:
            fam = "purple"
        else:
            fam = "pink"
        mod = "light" if l >= 0.65 else ("dark" if l < 0.3 or (sat < 0.35 and l < 0.4) else None)
        if sat < 0.35:
            extra.add("grey")
        if fam == "blue" and l < 0.22:
            extra.add("black")
        if fam == "red" and l < 0.4:
            extra.add("brown")
        if fam == "orange" and l < 0.42:
            extra |= {"red", "brown"}
    return {"name": (mod + " " if mod else "") + fam, "family": fam, "lightness": l, "extra": extra}


def colour_phrases(text):
    """Colour phrases in a question as (modifier, family) pairs, in order."""
    out = []
    for mod, word in COLOUR_PHRASE.findall(text):
        fam, implied = WORD_FAMILY[word.lower()]
        out.append((MODIFIER.get(mod.lower()) if mod else implied, fam))
    return out


def resolve_phrase(phrase, colours):
    """Return the indices of distinct chart colours a phrase can refer to (empty = no match,
    more than one = ambiguous). `colours` is a list of described colours, all distinct by name."""
    mod, fam = phrase
    cand = [i for i, c in enumerate(colours) if c["family"] == fam]
    if not cand:
        cand = [i for i, c in enumerate(colours) if fam in c["extra"]]
    if not cand:
        cand = [i for i, c in enumerate(colours) if fam in NEIGHBOURS.get(c["family"], ())]
    if len(cand) <= 1:
        return cand
    names = [colours[i]["name"] for i in cand]
    want = (mod + " " if mod else "") + fam
    exact = [i for i in cand if colours[i]["name"] == want]
    if len(exact) == 1:
        return exact
    if mod in ("light", "dark"):
        ls = sorted(cand, key=lambda i: colours[i]["lightness"])
        lo, hi = colours[ls[0]]["lightness"], colours[ls[-1]]["lightness"]
        if hi - lo >= 0.08:
            pick = ls[-1] if mod == "light" else ls[0]
            # The extreme must be clearly apart from the next one.
            nxt = ls[-2] if mod == "light" else ls[1]
            if abs(colours[pick]["lightness"] - colours[nxt]["lightness"]) >= 0.06:
                return [pick]
    if mod is None:
        plain = [i for i in cand if colours[i]["name"] == fam]
        if len(plain) == 1:
            return plain
    del names
    return cand


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


def clean(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def best_label(name, labels):
    """Map an annotation series/slice name to the table's spelling when they clearly match."""
    name = clean(name)
    if not labels:
        return name
    low = {clean(l).lower(): clean(l) for l in labels if clean(l)}
    if name.lower() in low:
        return low[name.lower()]
    m = difflib.get_close_matches(name.lower(), list(low), n=1, cutoff=0.8)
    return low[m[0]] if m else name


# --------------------------------------------------------------------------- annotations
def load_annotation(path, header, body):
    """Extract what the serialisation needs from one annotation JSON:
    chart_type, title, y-axis label, per-series colours, per-mark colours."""
    d = json.load(open(path, encoding="utf-8"))
    info = d.get("general_figure_info") or {}
    title = clean((info.get("title") or {}).get("text") or "") or None
    yl = (info.get("y_axis") or {}).get("label") or {}
    y_label = clean(yl.get("text") or "") if isinstance(yl, dict) else ""
    ctype = d.get("type")
    models = d.get("models") or []
    cols = [clean(h) for h in header[1:]]
    rows = [clean(r[0]) for r in body]
    series, marks = [], []   # series: (label, colour_raw); marks: (series_label|None, mark_label, colour_raw)
    if ctype == "pie":
        labels = [m.get("text_label", m.get("name")) for m in models]
        if any(l is None for l in labels):
            labels = rows if len(rows) == len(models) else [None] * len(models)
        for m, lab in zip(models, labels):
            if lab is not None:
                marks.append((None, best_label(lab, rows), m.get("color")))
    else:
        for m in models:
            nm = m.get("name")
            if not isinstance(nm, str) or nm.strip() in ("bars", "line", "", "{}"):
                nm = cols[0] if len(cols) == 1 and len(models) == 1 else None
            label = best_label(nm, cols + rows) if nm else None
            c = m.get("colors", m.get("color"))
            if isinstance(c, list):
                xs = m.get("x") or m.get("labels") or []
                uniq = {str(v).lower() for v in c}
                if len(uniq) == 1:
                    series.append((label, c[0]))
                else:
                    for x, v in zip(xs, c):
                        marks.append((label, best_label(str(x), rows + cols), v))
            else:
                series.append((label, c))
    return {"chart_type": ctype, "title": title, "y_label": y_label or None,
            "series": series, "marks": marks, "n_models": len(models)}


def chart_colours(ann):
    """Distinct described colours in a chart, plus whether any colour is unknown."""
    seen, out, unknown = set(), [], False
    for raw in [c for _, c in ann["series"]] + [c for _, _, c in ann["marks"]]:
        d = describe_colour(raw)
        if d is None:
            unknown = True
            continue
        key = str(raw).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        # Two raw colours with the same name are indistinguishable to a reader: merge them.
        same = [o for o in out if o["name"] == d["name"]]
        if same:
            same[0]["raws"].add(key)
            continue
        d["raws"] = {key}
        out.append(d)
    return out, unknown


def label_colours(ann):
    """(label, raw colour) for every named series and mark."""
    return [(l, c) for l, c in ann["series"] if l] + [(m, c) for _, m, c in ann["marks"]]


def colour_name(raw):
    d = describe_colour(raw)
    return d["name"] if d else "unknown"


def linearise(header, body, source, ann):
    lines = [f"Source: {source.upper() if source in ('owid', 'oecd') else source.capitalize()}"]
    series = header[1:]
    if ann:
        ct = CHART_TYPE_TEXT.get(ann["chart_type"], ann["chart_type"])
        n = len(ann["series"]) or ann["n_models"]
        lines.append(f"Chart type: {ct}" + (f", {n} series" if ct != "pie" and n > 1 else ""))
        if ann["title"]:
            lines.append(f"Title: {ann['title']}")
    lines.append(f"Category axis: {header[0] or 'Category'}")
    if ann and ann["series"]:
        parts = [f"{lab or '(unnamed)'}" + (f" (colour: {colour_name(c)})" if describe_colour(c) else "")
                 for lab, c in ann["series"]]
        if len(parts) == 1:
            mark = "bars" if "bar" in (ann["chart_type"] or "") else "points"
            parts[0] = (parts[0][:-1] + f"; single series, all {mark} share this colour)"
                        if parts[0].endswith(")") and "(colour:" in parts[0] else parts[0] + " (single series)")
        lines.append("Series: " + " | ".join(parts))
        listed = {clean(l).lower() for l, _ in ann["series"] if l}
        missing = [s for s in series if clean(s).lower() not in listed]
        if missing and len(series) > 1 and len(ann["series"]) != len(series):
            lines.append("Table columns: " + " | ".join(series))
    elif len(series) == 1:
        lines.append(f"Series: {series[0]}")
    else:
        lines.append("Series: " + " | ".join(series))
    if ann and ann["marks"]:
        word = "Slice" if ann["chart_type"] == "pie" else ("Bar" if "bar" in (ann["chart_type"] or "") else "Point")
        by_series = collections.OrderedDict()
        for s, m, c in ann["marks"]:
            if describe_colour(c) is None:
                continue
            by_series.setdefault(s, []).append(f"{m} = {colour_name(c)}")
        for s, items in by_series.items():
            prefix = f"{word} colours" + (f" ({s})" if s and len(by_series) > 1 else "")
            lines.append(f"{prefix}: " + "; ".join(items))
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


def domain_tags(header, body, question, ann):
    extra = " ".join(x for x in ((ann or {}).get("title"), (ann or {}).get("y_label")) if x)
    blob = " ".join(header) + " " + " ".join(r[0] for r in body) + " " + question + " " + extra
    return sorted(k for k, rx in DOMAIN_RE.items() if rx.search(blob))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_dir(path):
    h = hashlib.sha256()
    for name in sorted(os.listdir(path)):
        h.update(name.encode())
        h.update(bytes.fromhex(sha256(os.path.join(path, name))))
    return h.hexdigest()


# --------------------------------------------------------------------------- filtering
def classify(item, header, body, ann, text):
    """Return (exclusion_reason | None, flags dict). `text` is the full serialisation."""
    q = item["query"]
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
    flags["n_series"] = len(header) - 1
    ans, atype = item["answer"], item["answer_type"]
    colour_answer = False

    if APPEARANCE_OTHER.search(q):
        return "asks_about_chart_appearance", flags
    if TITLE_ASK.search(q) and not (ann and ann["title"]):
        return "asks_about_title_but_chart_has_none", flags
    if flags["mentions_colour"] or LEGEND.search(q):
        if ann is None:
            return "colour_reference_no_annotation", flags
        colours, unknown = chart_colours(ann)
        if not colours or unknown:
            # Mostly Pew charts whose annotator wrote "UNK" for some or all colours.
            return "colour_reference_annotation_lacks_colour", flags
        phrases = colour_phrases(q)
        resolved = []
        for p in phrases:
            hit = resolve_phrase(p, colours)
            if not hit:
                return "colour_reference_not_in_chart", flags
            if len(hit) > 1:
                return "colour_reference_ambiguous", flags
            resolved.append(colours[hit[0]]["name"])
        flags["colour_resolved_to"] = resolved
        if COLOUR_ASK.search(q) and atype == "text":
            # The answer itself is a colour: it must resolve, by its own colour family, to
            # exactly one colour in this chart.
            ap = colour_phrases(ans)
            hit = resolve_phrase(ap[0], colours) if ap else []
            if len(hit) != 1 or ap[0][1] not in {colours[hit[0]]["family"]} | colours[hit[0]]["extra"]:
                return "colour_answer_not_in_annotation", flags
            # If the question names exactly one series/mark, the gold colour must be that one's.
            named = {colour_name(c) for l, c in label_colours(ann) if len(norm_text(l)) >= 3
                     and re.search(r"(?<!\w)" + re.escape(norm_text(l)) + r"(?!\w)", norm_text(q))}
            if len(named) == 1 and colours[hit[0]]["name"] not in named:
                return "colour_answer_conflicts_with_annotation", flags
            colour_answer = True
            flags["colour_answer_input_name"] = colours[hit[0]]["name"]
        elif resolved and atype == "text":
            # "Which country does the red line represent?": when the gold answer is a series or
            # mark label, that label must carry the colour the question's phrase resolved to.
            label_colour = {norm_text(l): colour_name(c) for l, c in label_colours(ann)}
            if ans in label_colour and label_colour[ans] not in resolved:
                return "colour_reference_conflicts_with_gold", flags
        elif LEGEND.search(q) and not phrases:
            return "asks_about_chart_appearance", flags

    t_norm = norm_text(text)
    if atype == "text" and not colour_answer:
        # List answers such as "[United States, Spain]" must have every element in the serialisation.
        parts = [p for p in re.split(r"\s*,\s*", ans.strip("[]")) if p] if ans.startswith("[") else [ans]
        flat = re.sub(r"[^\w ]", "", t_norm)
        for a in parts:
            a = norm_text(a)
            if a not in t_norm and re.sub(r"[^\w ]", "", a) not in flat:
                return "text_answer_not_in_input", flags
    if atype == "number" and qtype != "arithmetic_comparison" and not COLOUR_ASK.search(q):
        # Plain look-up of a number: the gold value must be in the table (within rounding),
        # be a category label (e.g. a year) or appear in the title.
        vals = numeric_cells(body) + [v for v in (to_number(r[0]) for r in body) if v is not None] \
            + [v for v in (to_number(h) for h in header) if v is not None]
        if ann and ann["title"]:
            vals += [float(x) for x in re.findall(r"\d+(?:\.\d+)?", ann["title"].replace(",", ""))]
        g = float(ans)
        if not any(abs(v - g) <= max(0.011, abs(g) * 0.005) for v in vals):
            return "numeric_lookup_not_in_table", flags
    return None, flags


# --------------------------------------------------------------------------- sampling
def stratified_sample(items, n, rng, keys=STRATA):
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


def make_unanswerable(pool_items, texts, colours, n, rng, used_ids):
    """Pair a question with a different chart whose serialisation (title, series, colours,
    table) shares no content words or numbers with it."""
    stop = set("the a an of in on to for and or is are was were what which how many much value "
               "between by from with does do did this that it its at as be".split())
    charts = sorted(texts)
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
            ttext = texts[other].lower()
            twords = set(re.findall(r"[a-z]{3,}", ttext)) - {"source", "chart", "type", "category", "axis",
                                                            "series", "table", "colour", "colours", "single"}
            tnums = set(re.findall(r"\d+(?:\.\d+)?", ttext))
            if qwords & twords or qnums & tnums:
                continue
            if any(resolve_phrase(p, colours[other]) for p in colour_phrases(it["query"])):
                continue
            out.append((it, other))
            break
    return out


def build_record(split, it, header, body, text, chart_type, unanswerable=False, swapped_chart=None):
    src = it["source"] if not unanswerable else source_of(swapped_chart)
    user = f"Chart data:\n{text}\n\nQuestion: {it['query']}\n\nRespond with JSON: {SCHEMA_HINT}"
    if unanswerable:
        target = {"answer": None, "answer_type": "unanswerable", "unit": None, "evidence": [],
                  "rationale": "The chart data does not contain the information the question asks about."}
    else:
        answer = it["answer"] if it["answer_type"] == "number" else str(it["label"]).strip().rstrip(".")
        if it["answer_type"] == "yes_no":
            answer = answer.capitalize()
        cname = it["flags"].get("colour_answer_input_name")
        if cname and norm_text(answer).replace("gray", "grey") != cname:
            # The human wrote e.g. "gray" or "navy blue" for a colour the input calls "dark blue".
            # Score against the input's vocabulary; the human label stays in gold_answer_raw.
            answer = cname
            it = dict(it, answer=cname, flags=dict(it["flags"], colour_answer_rewritten=True))
        target = {"answer": answer, "answer_type": it["answer_type"],
                  "unit": infer_unit(header) if it["answer_type"] == "number" else None,
                  "evidence": [], "rationale": None}
    rid = f"{split}-{it['chart']}-{it['qidx']}" + ("-unans" if unanswerable else "")
    return {
        "id": rid,
        "version": VERSION,
        "split": split,
        "chart_id": swapped_chart if unanswerable else it["chart"],
        "question_source_chart": it["chart"],
        "source": src,
        "chart_type": chart_type,
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
    ap.add_argument("--chartqa", required=True, help='path to the "ChartQA Dataset" folder (with annotations/)')
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--domain-filter", choices=["off", "prefer", "strict"], default="prefer",
                    help="strict: keep only domain-matching items; prefer: fill from domain items first")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)

    stats = {"version": VERSION, "seed": args.seed, "domain_filter": args.domain_filter,
             "strata": list(STRATA), "splits": {}}
    manifest = {}
    raw = {}
    table_hash = {}
    for split in SPLITS:
        qa_path = os.path.join(args.chartqa, split, f"{split}_human.json")
        manifest[f"{split}/{split}_human.json"] = sha256(qa_path)
        manifest[f"{split}/tables/ (all files)"] = sha256_dir(os.path.join(args.chartqa, split, "tables"))
        manifest[f"{split}/annotations/ (all files)"] = sha256_dir(os.path.join(args.chartqa, split, "annotations"))
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
        pool, charts = [], {}   # charts: name -> (header, body, ann, text) or None
        colour_qs = collections.Counter()
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
            if name not in charts:
                t = read_table(p)
                if t is None or len(numeric_cells(t[1])) < 2:
                    charts[name] = None
                else:
                    ap_ = os.path.join(args.chartqa, split, "annotations", name + ".json")
                    ann = load_annotation(ap_, *t) if os.path.exists(ap_) else None
                    src = source_of(name)
                    charts[name] = (t[0], t[1], ann, linearise(t[0], t[1], src, ann))
            c = charts[name]
            if c is None:
                reasons["table_unparseable_or_lt2_cells"] += 1
                continue
            h = table_hash[(split, name)]
            if (split == "train" and h in eval_hashes) or (split == "val" and h in test_hashes):
                reasons["identical_table_in_eval_split"] += 1
                continue
            header, body, ann, text = c
            ans, atype = normalise_answer(x["label"])
            it = {"id": f"{split}-{name}-{qidx}", "chart": name, "qidx": qidx, "query": q,
                  "label": x["label"], "answer": ans, "answer_type": atype, "source": source_of(name),
                  "chart_type": (ann or {}).get("chart_type") or "unknown"}
            is_colour = bool(COLOUR.search(q))
            colour_qs["seen"] += is_colour
            reason, flags = classify(it, header, body, ann, text)
            if reason:
                reasons[reason] += 1
                continue
            colour_qs["kept"] += is_colour
            it["flags"] = flags
            it["question_type"] = flags["question_type"]
            it["domain_tags"] = domain_tags(header, body, q, ann)
            it["domain_match"] = bool(it["domain_tags"])
            pool.append(it)
        charts = {k: v for k, v in charts.items() if v is not None}
        pools[split] = (pool, charts)
        stats["splits"][split] = {
            "raw_human_pairs": len(raw[split]),
            "excluded": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
            "excluded_total": sum(reasons.values()),
            "answerable_pool": len(pool),
            "answerable_pool_domain_match": sum(it["domain_match"] for it in pool),
            "colour_questions_seen": colour_qs["seen"],
            "colour_questions_kept_in_pool": colour_qs["kept"],
        }

    for split, n_total in SPLITS.items():
        pool, charts = pools[split]
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
        texts = {k: v[3] for k, v in charts.items()}
        colours = {k: (chart_colours(v[2])[0] if v[2] else []) for k, v in charts.items()}
        unans = make_unanswerable(pool, texts, colours, n_unans, rng, used)

        def ctype(name):
            ann = charts[name][2]
            return (ann or {}).get("chart_type") or "unknown"

        records = [build_record(split, it, charts[it["chart"]][0], charts[it["chart"]][1],
                                charts[it["chart"]][3], it["chart_type"]) for it in chosen]
        records += [build_record(split, it, charts[o][0], charts[o][1], charts[o][3], ctype(o),
                                 unanswerable=True, swapped_chart=o) for it, o in unans]
        rng.shuffle(records)
        with open(os.path.join(args.out, f"{split}.jsonl"), "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        s = stats["splits"][split]
        s["final_total"] = len(records)
        s["final_answerable"] = len(chosen)
        s["final_unanswerable"] = len(unans)
        s["final_charts"] = len({r["chart_id"] for r in records})
        for key in ("source", "chart_type", "answer_type", "question_type"):
            s[f"final_by_{key}"] = dict(collections.Counter(r[key] for r in records).most_common())
        s["final_colour_questions"] = sum(1 for r in records if r["flags"].get("mentions_colour"))
        s["final_with_title"] = sum(1 for r in records if "\nTitle: " in r["input_text"])
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
