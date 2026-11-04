# ChartAudit working dataset v1.0

Built 4 Nov 2026 by `build_dataset.py` (seed 42, standard-library Python, deterministic: two runs give byte-identical files) from **`ChartQA Dataset.zip` on Hugging Face** (`ahmed-masry/ChartQA`, revision `af8b6f5`, zip SHA-256 `1bf310e5…c3c160`). Follows steps 1–8 of `../Week1_TrackA_ChartQA_Deliverable.md`.

v1.0 replaces v0.9. The difference is that v1.0 uses the ChartQA **annotation JSON** (chart type, title, series and mark colours), which v0.9 could not download. The `*_human.json` question files are byte-identical to the ones v0.9 used (same SHA-256), so the question pool is the same and only what the model sees and what gets excluded has changed.

| File | Contents |
|---|---|
| `v1.0/train.jsonl` | 1,450 items (1,378 answerable + 72 unanswerable), 1,143 charts |
| `v1.0/val.jsonl` | 170 items (162 + 8), 144 charts |
| `v1.0/test.jsonl` | 180 items (171 + 9), 157 charts. **Held out; never train or tune on it** |
| `v1.0/stats.json` | Every count below, plus breakdowns |
| `v1.0/MANIFEST.txt` | SHA-256 of the source QA files, the tables and annotations folders, and each output split |
| `v0.9/` | Previous build, kept for reference (`README_v0.9.md`, built by `build_dataset_v0.9.py`) |

Human-written (ChartQA-H) questions only. Chart overlap between splits: **0** (checked on chart ids; identical-table leakage removed as in v0.9: 18 train and 2 val items).

## Rebuild

```bash
curl -L -o cq.zip "https://huggingface.co/datasets/ahmed-masry/ChartQA/resolve/af8b6f5c08c95085271561c2a3f9d15f2b5a9031/ChartQA%20Dataset.zip"
unzip -q cq.zip -x '*.png'      # the PNGs are not needed
python3 build_dataset.py --chartqa "ChartQA Dataset" --out v1.0
```
Each split folder needs `*_human.json`, `tables/` and `annotations/`. Options: `--seed`, `--domain-filter off|prefer|strict` (default `prefer`).

## What one record looks like

Each line has `messages` (system / user / assistant) ready for an SFT chat template, plus metadata: `id`, `version`, `chart_id`, `source` (statista / pew / owid / oecd), `chart_type` (v_bar / h_bar / line / pie), `question`, `gold_answer_raw`, `answer`, `answer_norm`, `answer_type` (number / text / yes_no / unanswerable), `question_type` (retrieval / arithmetic_comparison / visual_reference / unanswerable), `domain_tags`, `flags`, `target_status`, `input_text`.

The user turn is the linearised chart plus the question. New in v1.0: `Chart type`, `Title` (Pew, OWID and OECD charts; Statista annotations have no title) and a colour for every series, or for every bar or slice when they differ:

```
Chart data:
Source: Statista
Chart type: vertical bar, 2 series
Category axis: Characteristic
Series: Female (colour: blue) | Male (colour: dark blue)
Table:
| Characteristic | Female | Male |
...
Question: Does women have the higher value in all categories?
```

```
Chart data:
Source: OWID
Chart type: horizontal bar
Title: Revenue from income taxes in Europe, 1967
Category axis: Country
Series: Revenue from income taxes in Europe, 1967
Bar colours: Switzerland = dark teal; Norway = red; Denmark = purple; Austria = dark blue
Table:
...
```

Colours are named, not given as hex. Statista, OWID and OECD annotations store hex codes, which the script maps to a plain name (hue family plus "light"/"dark"); Pew annotations already store names such as "light brown", which are kept after fixing typos. Pew charts whose annotator wrote `UNK` show no colour.

The assistant turn follows the deliverable's JSON schema:

```json
{"answer": "No", "answer_type": "yes_no", "unit": null, "evidence": [], "rationale": null}
```

**`evidence` and `rationale` are empty on purpose** (`target_status: answer_only_pending_teacher_rationale`). Step 7 (teacher rationales, kept only if they reproduce the gold answer) fills them in. The answer-only version is also the Week 2 "no rationales" ablation set, so keep a copy.

`unit` is a heuristic read from the value-column header. It is set on 248 / 31 / 45 numeric items (train / val / test) and null otherwise. Spot-check it during the audit.

## How colour questions are handled

A colour question is kept when every colour it names (e.g. "the navy blue bar", "light blue", "red line") points to **exactly one** colour in that chart. Matching tries the colour's own family first ("blue" → blue), then close neighbours a reader might use (a red-orange OWID line called "red"), and uses "light"/"dark" to choose between two shades of the same hue. It is excluded when the colour isn't in the chart, when two colours fit equally well, or when the annotation has no colour for it.

Two consistency checks against the human gold answer:
- "Which country does the red line represent?": if the answer is a series or bar label, that label must have the colour the question named. 20 / 2 / 6 questions failed this and were dropped.
- "What colour is X?": the gold colour must be one of the chart's colours, and it must be X's colour when the question names X. When the human's word differs from the input's name for the same colour (e.g. gold "gray" or "navy blue" for a colour the input calls "dark blue"), the target **`answer` uses the input's name** and the human's word stays in `gold_answer_raw` (`flags.colour_answer_rewritten`). This affects 11 / 2 / 1 items.

"What colour", title and "how many colours" questions were all excluded as "appearance" in v0.9; they are kept now when the input contains the answer. Gridlines, dotted/dashed lines, fonts, shading and legend-only questions are still excluded.

## Exclusions (questions the text can't answer)

| Reason | Train | Val | Test |
|---|---:|---:|---:|
| Raw human-written pairs | 7,398 | 960 | 1,250 |
| Number look-up whose gold value isn't in the table (mostly Pew labels on a different scale, e.g. 0.37 vs 37) | 371 | 44 | 32 |
| Colour question on a chart whose annotation has no colour (Pew `UNK`) | 204 | 16 | 29 |
| Text answer not in the input (table, title, series names) | 54 | 4 | 5 |
| Colour named in the question isn't in the chart | 41 | 8 | 3 |
| Colour named matches two colours equally | 27 | 1 | 7 |
| "What colour" answer isn't a colour in the chart | 24 | 5 | 6 |
| Colour points to a different label than the gold answer | 20 | 2 | 6 |
| Gold colour isn't the named item's colour | 2 | 0 | 2 |
| Asks about appearance (gridlines, dotted line, font, shading, legend) | 16 | 2 | 0 |
| Asks about the title of a chart that has none | 1 | 0 | 0 |
| Table unparseable or < 2 numeric cells | 24 | 8 | 8 |
| Identical table in an eval split (leakage) | 18 | 2 | 0 |
| Exact duplicate | 1 | 0 | 0 |
| **Excluded total** | **803** (11%) | **92** (10%) | **98** (8%) |
| Answerable pool left | 6,595 | 868 | 1,152 |
| ...of which matches the finance/economy/health/business lexicon | 2,987 | 410 | 532 |

In v0.9 the excluded total was 2,029 / 231 / 287, with 1,221 / 136 / 190 for unresolvable colour references and 373 / 44 / 52 for appearance questions. Of the 1,636 / 185 / 248 questions that mention a colour, **1,259 / 142 / 193 are now in the answerable pool**.

## Sampling

1. Domain filter (step 4): the lexicon is applied to column headers, row labels, the question, and now the **chart title and y-axis label**. Every split's domain pool was larger than needed, so all answerable items are domain-matched.
2. Stratified sampling within each official split, proportional over **source × chart type × answer type × question type** (chart type is new in v1.0).
3. ~5% unanswerable items: a question from the same split paired with a different chart that shares no content words or numbers with it (checked against the full input, including title and series names) and has no colour the question names. Target is `answer: null`.

Final mix (train / val / test):

| | Train | Val | Test |
|---|---:|---:|---:|
| Statista | 902 | 93 | 110 |
| Pew | 275 | 17 | 19 |
| OWID | 208 | 45 | 42 |
| OECD | 65 | 15 | 9 |
| Vertical bar / horizontal bar / line / pie | 518 / 450 / 300 / 182 | 51 / 61 / 37 / 21 | 60 / 58 / 41 / 21 |
| Number / text / yes-no / unanswerable | 973 / 309 / 96 / 72 | 111 / 44 / 7 / 8 | 118 / 42 / 11 / 9 |
| Arithmetic-comparison / visual-reference / retrieval | 754 / 370 / 254 | 87 / 46 / 29 | 95 / 40 / 36 |
| Mentions a colour | 227 | 20 | 22 |
| Input has a chart title | 547 | 77 | 69 |

Statista is 62% of train (70% in v0.9) because titles let more Pew and OWID charts pass the domain filter.

Question types come from keyword rules (`ARITH`, `COLOUR`, `STRUCTURE` regexes in the script). They are good enough for stratification and breakdowns, not exact.

## Changes from v0.9 that affect other work

- **The test set changed.** Only 54 of the 180 v0.9 test items are also in v1.0 (651 / 65 shared in train / val). Any baseline already run on v0.9 test must be rerun on v1.0.
- The input text is longer (title and colour lines) and has two new header lines; prompts that parse the input should not assume the v0.9 layout.
- Tables now come from the Hugging Face zip, not the GitHub repo. The question files are identical; the tables were not compared file by file.

## Known gaps

- Colour names are a rule-based mapping from hex codes, so a few will disagree with what a person would say (Statista's `#2876dd` is "blue", though people sometimes call it "light blue" next to the dark blue series). Check colour items in the audit.
- Pew tables contain OCR noise from the original extraction (e.g. "Ra rely", "understa nd"), and some Pew series names in the annotation differ slightly from the table header. Focus the 150-item audit (step 9) on Pew and on colour questions.
- Steps 7 (teacher rationales), 9 (manual audit) and 10 (freeze tag + data card) are still to do.
