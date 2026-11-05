# Data card: ChartAudit v1.0

## Summary

Instruction–response pairs for teaching a text-only ≤8B model to answer questions about charts given as text, with grounded evidence and a short rationale, in a strict JSON schema. Built for the Track A (QLoRA) capstone, Week 1.

| Split | Items | Answerable | Unanswerable | Charts |
|---|---:|---:|---:|---:|
| Train | 1,449 | 1,377 | 72 | 1,137 |
| Validation | 170 | 162 | 8 | 142 |
| Test (frozen, held out) | 180 | 171 | 9 | 157 |

## Source and licence

- **ChartQA** (Masry et al., *Findings of ACL 2022*), human-written split (ChartQA-H) only. Downloaded as `ChartQA Dataset.zip` from `huggingface.co/datasets/ahmed-masry/ChartQA`, revision `af8b6f5`. Question files are byte-identical to `github.com/vis-nlp/ChartQA` commit `044eabf`.
- Licence: ChartQA is released under **GPL-3.0**; this derived dataset carries the same licence.
- Charts come from Statista, Pew Research Center, Our World in Data and OECD.
- Evidence and rationales were written by **Claude Opus** (teacher model, synthetic generation as allowed by the capstone template). Gold answers are ChartQA's human labels and were never changed by the teacher. The only target edit is 15 "what colour" items (12 / 2 / 1) whose human wording ("gray", "navy blue") was mapped to the colour name used in the input (`flags.colour_answer_rewritten`).

## Input representation

Text only: source, chart type, title (Pew/OWID/OECD; Statista charts have none), series with colour names, per-bar or per-slice colours where they differ, and the data table in Markdown. Colours are named by rule from the annotation's hex codes. No images.

## Filters (in order)

1. Human-written questions only; official chart-disjoint splits kept; byte-identical tables shared between splits removed (18 train, 2 val).
2. Questions the text can't answer removed: number look-ups not in the table, text answers not in the input, colour references that don't resolve to exactly one chart colour or that contradict the gold, appearance questions (gridlines, fonts, shading, dotted lines), unparseable tables. Counts per reason: `v1.0/selection/stats.json`.
3. Domain lexicon (finance, economy, health, business) over title, axis label, headers, row labels and question. All answerable items match it.
4. Stratified sampling over source × chart type × answer type × question type; ~5% built unanswerable items.
5. Teacher verification; rejected items replaced within the same stratum (`teacher_stats.json`, `teacher/rejected.jsonl`).

## Composition (train / val / test)

| | Train | Val | Test |
|---|---:|---:|---:|
| Statista / Pew / OWID / OECD | 902 / 274 / 208 / 65 | 93 / 17 / 45 / 15 | 110 / 19 / 42 / 9 |
| Vertical bar / horizontal bar / line / pie | 517 / 454 / 297 / 181 | 51 / 62 / 36 / 21 | 60 / 58 / 41 / 21 |
| Number / text / yes-no / unanswerable | 972 / 309 / 96 / 72 | 111 / 44 / 7 / 8 | 118 / 42 / 11 / 9 |
| Arithmetic / visual reference / retrieval | 756 / 368 / 253 | 87 / 46 / 29 | 95 / 40 / 36 |

## Known biases and limitations

- **Statista-heavy** (62% of train) because the domain lexicon matches Statista headers most often.
- **Pew tables are noisy** (OCR errors, `nan`, `0` used for missing values). The teacher rejected 29% of Pew golds; kept Pew items can still rely on reading past a missing cell. Pew rows appear to be listed bottom-to-top relative to the chart, which matters for "top bar" questions.
- **Test labels are not teacher-checked.** Train/val gold error rates (10.4% / 8.0% rejected) suggest the test set also has wrong golds at a similar rate; report this alongside accuracy, or have the audit cover a test sample too.
- Colour names come from a rule-based hex mapping; humans sometimes use another name ("light blue" for the input's "blue"). Some kept items rely on the teacher reading such a near-match.
- Per-bar colour lines were removed for charts whose annotation was found unreliable (`flags.mark_colours_removed`).
- Question-type labels are keyword rules, good for stratification, not exact.
- Not a regulated domain; the "financial/economic reporting" framing is a domain filter, not a guarantee every chart is financial.

## Intended use

Training and evaluating small text-only models on grounded chart QA with JSON outputs. Not for drawing factual conclusions about the underlying statistics, which may be outdated or garbled.

## Pending

Manual audit of 150 training items by two reviewers (`audit/audit_sheet.csv`, step 9), then the `v1.0-frozen` tag (step 10).
