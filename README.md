# ChartQA Capstone (Track A)

IIITK Exec PG GenAI capstone project: QLoRA fine-tuning of a ≤8B instruction model on a text-linearised subset of ChartQA (human-written questions only).

## Layout

- `docs/Week1_TrackA_ChartQA_Deliverable.md`: Week 1 exploration and plan
- `dataset/README.md`: how the dataset was built and how to rebuild it; `dataset/DATA_CARD.md`: sources, licence, filters, counts, known biases
- `dataset/v1.0/`: fine-tuning-ready splits (`train.jsonl` 1,449, `val.jsonl` 170, `test.jsonl` 180 held out) with teacher-verified evidence and rationales; `answer_only/` for the no-rationale ablation
- `dataset/build_dataset.py`: selects and linearises the items; `dataset/add_teacher_labels.py`: merges the Claude Opus teacher labels (`dataset/teacher/`) and replaces rejected items
- `dataset/audit/audit_sheet.csv`: 150-item manual audit sample (step 9, pending)
- `dataset/v0.9/`: earlier splits without colours, kept for reference (`build_dataset_v0.9.py` rebuilds them)

The raw ChartQA archive (Hugging Face `ahmed-masry/ChartQA`, about 875 MB) is not committed; see `dataset/README.md` for the download command.

ChartQA is released under GPL-3.0; derived data here follows that licence.
