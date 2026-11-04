# ChartQA Capstone (Track A)

IIITK Exec PG GenAI capstone project: QLoRA fine-tuning of a ≤8B instruction model on a text-linearised subset of ChartQA (human-written questions only).

## Layout

- `docs/Week1_TrackA_ChartQA_Deliverable.md`: Week 1 exploration and plan
- `dataset/build_dataset.py`: builds the v1.0 instruction–response splits
- `dataset/README.md`: dataset card and build notes
- `dataset/v1.0/`: current splits with chart colours (`train.jsonl`, `val.jsonl`, `test.jsonl`, 1,450/170/180), `stats.json`, `MANIFEST.txt`
- `dataset/v0.9/`: earlier splits without colours, kept for reference (`build_dataset_v0.9.py` rebuilds them)

The raw ChartQA archive (Hugging Face `ahmed-masry/ChartQA`, about 875 MB) is not committed; see `dataset/README.md` for the download command.

ChartQA is released under GPL-3.0; derived data here follows that licence.
