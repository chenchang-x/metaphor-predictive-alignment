# Screening and Measurement Provenance

Formal experiment: `Qwen/Qwen3.5-9B`, `results/formal/20260908T224807Z`. These tables use this project's materials and saved results.

## Screening records

The project archive did not contain separate `data/processed/exclusions.csv` or `preprocessing_summary.json` files. The screening records were reconstructed with `build_records` from `runpod/experiments/build_analysis_items.py`, using the two fixed source CSVs in `vendor/metaphor-understanding-challenge/correct_answers/`.

`validate_source_checkout` verifies source hashes before reconstruction. Every field, row and row position in the reconstructed 880 retained records matches the frozen `data/processed/analysis_items.csv`.
Frozen-table SHA-256: `e9282952a95f6a0b33a229c7bb4f9ab71496a2c7fafbf60af502917ef3a72f93`.

`screening_stages.csv` records the ordered sequence 1492 → 1072 → 1072 → 1046 → 880 → 880. `screening_exclusions.csv` retains all 612 exclusions and their reasons: 420 label failures, 26 non-target context mismatches and 166 cases with fewer than three shared post-target words. Target-markup and deduplication stages each excluded 0 records.

These are reproducible reconstructions rather than independently saved historical screening files. Frozen materials and formal results are unchanged.

## Measurement checks

`measurement_checks.csv` contains 31 checks. CSV counts, score finiteness, exact input matching and context-increment arithmetic are recomputed. Continuation completion counts and tokenizer-boundary checks are taken from their formal-run records, as identified in `detail`. Model generation and tokenisation are not rerun.

All 12 unambiguous contextual controls have positive scores; the prespecified threshold is at least 60%. Word-only controls have no correctness threshold. Passing verifies the scoring interface against its rules, rather than establishing the validity of every formal item.

The contextual task input comprises the complete left context, marked target and three shared post-target orthographic words. It is not the complete source sentence. Removing target markers from all 1760 contextual order records exactly recovers the RQ1 `m_prefix_q3`; all 1760 word-only order records contain only the marked target word. For 842 of the 880 materials, the original sentence extends beyond the actual input prefix.

## Reproduction

Run `python runpod/experiments/build_screening_audit.py` from the repository root. The script writes these four audit files to `figures/source_data/`.

Source-file SHA-256 values:

```json
{
  "correct_answers/for_generation.csv": "7816b422516608d015eb69926571eb7647899e9eee4bc65dbbf46aeb2f64ca7f",
  "correct_answers/for_judgement.csv": "6d3a69c9efeaf570e25ad981fa3e967563495c42429d6111877c1e2eb577d99a"
}
```
