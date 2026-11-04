# ChartAudit working dataset v0.9 (superseded by v1.0, see ../README.md)

Built 4 Nov 2026 by `build_dataset_v0.9.py` (seed 42, standard-library Python, deterministic) from ChartQA commit `044eabf` of `github.com/vis-nlp/ChartQA`. Follows steps 1–8 of `../Week1_TrackA_ChartQA_Deliverable.md`.

| File | Contents |
|---|---|
| `v0.9/train.jsonl` | 1,450 items (1,378 answerable + 72 unanswerable), 1,068 charts |
| `v0.9/val.jsonl` | 170 items (162 + 8), 130 charts |
| `v0.9/test.jsonl` | 180 items (171 + 9), 143 charts. **Held out; never train or tune on it** |
| `v0.9/stats.json` | Every count below, plus breakdowns |
| `v0.9/MANIFEST.txt` | SHA-256 of the source QA files and of each output split |

Chart overlap between splits: **0** (checked on chart ids, plus identical-table leakage removed: 18 train and 2 val items whose CSV was byte-identical to an eval chart).

## Rebuild

```bash
git clone --depth 1 https://github.com/vis-nlp/ChartQA.git
python3 build_dataset.py --chartqa "ChartQA/ChartQA Dataset" --out v0.9
```
Only the `*_human.json` files and `tables/` folders are needed. Options: `--seed`, `--domain-filter off|prefer|strict` (default `prefer`).

## What one record looks like

Each line has `messages` (system / user / assistant) ready for an SFT chat template, plus metadata: `id`, `chart_id`, `source` (statista / pew / owid / oecd), `question`, `gold_answer_raw`, `answer`, `answer_norm`, `answer_type` (number / text / yes_no / unanswerable), `question_type` (retrieval / arithmetic_comparison / visual_reference / unanswerable), `domain_tags`, `flags`, `target_status`, `input_text`.

The user turn is the linearised chart plus the question:

```
Chart data:
Source: Statista
Category axis: Characteristic
Series: Premiums written in billion U.S. dollars (single series; all bars/points share one colour)
Table:
| Characteristic | Premiums written in billion U.S. dollars |
|---|---|
| State Farm Mutual Automobile Insurance | 65.61 |
...
Question: What is the value for Allstate Corp?
```

The assistant turn follows the deliverable's JSON schema:

```json
{"answer": 35.03, "answer_type": "number", "unit": "billion U.S. dollars", "evidence": [], "rationale": null}
```

**`evidence` and `rationale` are empty on purpose** (`target_status: answer_only_pending_teacher_rationale`). Step 7 (teacher rationales, kept only if they reproduce the gold answer) fills them in. The answer-only version is also exactly the Week 2 "no rationales" ablation set, so keep a copy.

`unit` is a heuristic read from the value-column header ("in billion U.S. dollars", "share", "%"). It is set on 340 / 43 / 62 numeric items (train / val / test) and null otherwise. Spot-check it during the audit.

## Exclusions (questions the text can't answer)

| Reason | Train | Val | Test |
|---|---:|---:|---:|
| Raw human-written pairs | 7,398 | 960 | 1,250 |
| Colour reference that can't be resolved without the annotation file (multi-series, pie, or non-Statista) | 1,221 | 136 | 190 |
| Asks about appearance (what colour, legend, title, gridlines, dotted line) | 373 | 44 | 52 |
| Number look-up whose gold value isn't in the table (mostly Pew labels on a different scale, e.g. 0.37 vs 37) | 331 | 36 | 31 |
| Text answer not in the table (e.g. comes from the chart title) | 61 | 5 | 6 |
| Table unparseable or < 2 numeric cells | 24 | 8 | 8 |
| Identical table in an eval split (leakage) | 18 | 2 | 0 |
| Exact duplicate | 1 | 0 | 0 |
| **Excluded total** | **2,029** (27%) | **231** (24%) | **287** (23%) |
| Answerable pool left | 5,369 | 729 | 963 |
| ...of which matches the finance/economy/health/business lexicon | 2,046 | 299 | 391 |

## Sampling

1. Domain filter (step 4): the lexicon is applied to column headers, row labels and the question (chart titles are not available, see below). Every split's domain pool was larger than needed, so **all answerable items are domain-matched**.
2. Stratified sampling within each official split, proportional over source × answer type × question type.
3. ~5% unanswerable items: a question from the same split paired with a different chart's table that shares no content words or numbers with it. Target is `answer: null`.

Final mix (train / val / test):

| | Train | Val | Test |
|---|---:|---:|---:|
| Statista | 1,013 | 105 | 116 |
| OWID | 217 | 40 | 44 |
| Pew | 163 | 13 | 11 |
| OECD | 57 | 12 | 9 |
| Number / text / yes-no | 1,017 / 278 / 83 | 116 / 38 / 8 | 123 / 35 / 13 |
| Arithmetic-comparison / retrieval / visual-reference | 795 / 317 / 266 | 91 / 36 / 35 | 99 / 42 / 30 |

Question types come from keyword rules (`ARITH`, `COLOUR`, `STRUCTURE` regexes in the script). They are good enough for stratification and breakdowns, not exact.

## Known gaps versus the deliverable

- **No annotation JSON, so no chart titles, chart types or series colours.** The annotations are only in the Hugging Face mirror (`ahmed-masry/ChartQA`), which this build environment could not reach. Effects: colour questions are kept only for single-series Statista bar/line charts (all marks share one colour), which is why "colour unresolvable" is the largest exclusion; chart type can't be a stratification key; the domain filter runs on headers instead of titles. With the annotations, a v1.0 rebuild would recover most of the ~1,500 colour exclusions.
- Pew tables contain OCR noise from the original extraction (e.g. "Ra rely", "understa nd"). Focus the 150-item audit (step 9) there.
- The domain filter skews the mix toward Statista (70% of train) because business vocabulary sits in Statista headers.
- Steps 7 (teacher rationales), 9 (manual audit) and 10 (freeze tag + data card) are still to do.
