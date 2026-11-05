# ChartAudit dataset v1.0 (fine-tuning ready)

`v1.0/train.jsonl` and `v1.0/val.jsonl` are ready for supervised fine-tuning. Every answerable item has a target with **evidence** (table cells, colours or title) and a short **rationale**, written by Claude Opus as teacher and checked by a script. `v1.0/test.jsonl` is the frozen held-out set; it is scored on the answer only.

| File | Items | What it is |
|---|---:|---|
| `v1.0/train.jsonl` | 1,449 (1,377 teacher-verified + 72 unanswerable) | Main SFT training set |
| `v1.0/val.jsonl` | 170 (162 + 8) | Validation / model selection |
| `v1.0/test.jsonl` | 180 (171 + 9) | **Held out. Never train or tune on it** |
| `v1.0/answer_only/` | same items | Same splits with `evidence: []` and `rationale: null`: the Week 2 "no rationales" ablation |
| `v1.0/teacher_stats.json` | | Teacher acceptance, rejections and final mix |
| `v1.0/MANIFEST.txt` | | SHA-256 of every final file |
| `v1.0/selection/` | | Answer-only output of `build_dataset.py` (the sample before teacher labelling), its `stats.json` and source checksums |
| `teacher/` | | Teacher prompt, checker, every batch sent and every output returned, `rejected.jsonl` |
| `audit/audit_sheet.csv` | 150 | Manual audit sample for step 9 (to be filled in by two reviewers) |
| `DATA_CARD.md` | | Sources, licence, filters, counts, known biases |

Chart overlap between splits is 0. Train is 1,449 rather than 1,450 because one Pew line-chart stratum ran out of items the teacher could verify.

## One training record

`messages` is ready for a chat template (system / user / assistant). The system prompt is general:

```
You are a chart analyst. Answer questions about the chart data provided.
Return ONLY valid JSON matching the schema. If the data cannot answer the question,
set "answer" to null and explain in "rationale".
```

User turn (chart serialised as text, then the question):

```
Chart data:
Source: Statista
Chart type: vertical bar, 2 series
Category axis: Characteristic
Series: Minimum price (colour: blue) | Maximum price (colour: dark blue)
Table:
| Characteristic | Minimum price | Maximum price |
|---|---|---|
| Cannabis (resin) | 5.35 | 13 |
| Cocaine | 40.0 | 133 |
...
Question: What is the value of the highest dark blue bar?
```

Assistant turn:

```json
{"answer": 133, "answer_type": "number", "unit": null,
 "evidence": ["Colour: Maximum price = dark blue", "Col=Maximum price (all rows)"],
 "rationale": "Dark blue is Maximum price; its highest value is 133 (Cocaine and Heroin)."}
```

Unanswerable items (a question paired with the wrong chart) have `answer: null`, `evidence: []` and a fixed refusal rationale. Metadata fields: `id`, `version`, `split`, `chart_id`, `source`, `chart_type`, `question`, `gold_answer_raw`, `answer`, `answer_norm`, `answer_type`, `question_type`, `domain_tags`, `flags`, `target_status` (`teacher_verified` / `fixed_refusal`), `input_text`, and `replaces` on items that took a rejected item's place.

Longest record is about 2,400 characters (roughly 700 tokens), so `max_seq_length = 1024` is enough.

## How it was built

1. **Selection** (`build_dataset.py`): ChartQA human-written questions only, official splits, linearised with the annotation JSON (chart type, title, series and bar/slice colours), questions the text can't answer removed, domain filter, stratified over source × chart type × answer type × question type, ~5% unanswerable. Details of filters and exclusion counts are in `v1.0/selection/stats.json` and in `DATA_CARD.md`.
2. **Teacher labelling** (step 7): Claude Opus received each item's input, question and gold answer (`teacher/INSTRUCTIONS.md`) and returned evidence, a rationale of at most 40 words and whether it agrees with the gold. The gold answer is never changed.
3. **Verification** (`teacher/check_teacher.py`): schema valid, answer equals gold (relaxed match), every evidence string points at a real cell, colour mapping or title in the input, rationale length. Every kept item passes all checks.
4. **Rejection and replacement** (`add_teacher_labels.py`): when the teacher judged the gold answer wrong or unsupported, the item was dropped and replaced by an unused item from the same split and stratum, which went through the same teacher and checks.

| | Train | Val |
|---|---:|---:|
| Items labelled by the teacher (originals + replacement candidates) | 1,672 | 200 |
| Originals rejected | 143 (10.4%) | 13 (8.0%) |
| ...teacher disputed the gold | 134 | 12 |
| ...colour mapping found unreliable (below) | 9 | 1 |
| Replaced | 142 | 13 |

Across everything labelled, the teacher disputed 12% of gold answers: Pew 29%, OECD 13%, OWID 11%, Statista 5.5%; arithmetic questions 17%, visual-reference 7%, retrieval 4%. Typical causes are wrong arithmetic in the human label (a mean given as the median, a ratio inverted), percentage points written as fractions (0.58 for 58), and Pew tables with garbled or missing cells (`nan`, `0` as a placeholder). Every rejection and the teacher's reason is in `teacher/rejected.jsonl`.

**Colour fix found during labelling.** In about half of the OECD bar charts the annotation lists bar labels in a different order from bar values and colours, and some Statista pie annotations repeat a slice label. For those charts the per-bar/per-slice colour line was wrong. The build now flags them (`mark_colours_unreliable`); the final files drop that line from the input (`flags.mark_colours_removed`: 52 train, 6 val items), and colour questions on those charts were rejected and replaced. The test split got the same input repair (9 items); none of its colour questions were affected.

## Rebuild

```bash
curl -L -o cq.zip "https://huggingface.co/datasets/ahmed-masry/ChartQA/resolve/af8b6f5c08c95085271561c2a3f9d15f2b5a9031/ChartQA%20Dataset.zip"
unzip -q cq.zip -x '*.png'
python3 build_dataset.py --chartqa "ChartQA Dataset" --out v1.0/selection --pool-out pool.json
python3 add_teacher_labels.py merge --data v1.0/selection --pool pool.json --teacher teacher
```

The merge reuses the saved teacher outputs in `teacher/outputs/`, so the rebuild is byte-identical without calling the teacher again (checked). `select` and `topup` are the commands that chose the replacement candidates.

## Still to do

- **Manual audit (step 9):** two team members fill in `audit/audit_sheet.csv` (gold correct, rationale faithful, schema OK, Y/N each), then `python3 make_audit_sheet.py --kappa audit/audit_sheet.csv` reports agreement and Cohen's kappa. Fix or drop flagged items and log the change.
- **Freeze (step 10):** after the audit, tag the repo `v1.0-frozen`.

v0.9 (tables only, no annotations) is kept in `v0.9/` with its own README.
