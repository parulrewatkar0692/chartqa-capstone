# Teacher task: write evidence + rationale for ChartQA items

You are the teacher model for a fine-tuning dataset. Each input line in your batch file is one item:
`{"id", "question", "gold_answer", "answer_type", "unit", "input_text"}`. `input_text` is the chart as the student model will see it
(source, chart type, title, series with colours, Markdown table). The gold answer is human-written and is NEVER changed by you.

For every item, write ONE output line (JSON, one object per line, same order) to your output file:

{"id": "<same id>", "answer": <gold_answer exactly as given>, "evidence": [...], "rationale": "...", "teacher_agrees": true|false, "note": "<only if teacher_agrees is false>"}

## Rules
1. Work it out yourself from `input_text` only. Do not use outside knowledge.
2. `teacher_agrees`: true if the input supports the gold answer (numbers within ~5%, or rounding of the same value). false if the input
   clearly gives a different answer or cannot support it; then put a one-sentence reason in `note` (e.g. "table gives 47.2, gold 52").
   Never bend the rationale to justify a gold answer you think is wrong. Disagreeing is useful, not a failure.
3. `answer`: copy `gold_answer` exactly (same type: number stays number, string stays string).
4. `evidence`: the minimal list of facts from the input that the answer depends on. Each string MUST be in one of these exact forms:
   - `Row=<first-column label>, Col=<column header> (<cell value exactly as in the table>)`   e.g. `Row=Air Force, Col=Male (49,970)`
   - `Col=<column header> (all rows)`  when the answer uses a whole column (max, count, average over all rows)
   - `Row=<first-column label> (all columns)`  when it uses a whole row
   - `Colour: <series or bar/slice label> = <colour name as written in the input>`  when the question names a colour
   - `Title: <exact substring of the Title line>`  when the answer comes from the title
   Labels and headers must be copied exactly as they appear in the table (the checker matches them case-insensitively).
   For colour questions include the Colour line(s) AND the cells used.
5. `rationale`: at most 40 words, plain English, shows the reasoning/arithmetic that gets to the gold answer,
   e.g. "Uruguay is 31.2 and Guatemala 20.72; 31.2 + 20.72 = 51.92." Do not mention "gold" or "the label".
6. Output only JSONL lines in the file, nothing else.

## Verify before you finish
Run: `python3 <TEACHER_DIR>/check_teacher.py <BATCH_FILE> <OUT_FILE>`
It checks the schema, answer == gold, every evidence string against the input, and rationale length.
Fix every failure it reports EXCEPT `teacher_disputes_gold` lines (those are your honest disagreements; leave them).
Rerun until the only failures left are disputes. Then reply with the final SUMMARY line and the number of disputes.
