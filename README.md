# Qwen3.5-9B Metaphor Predictive Alignment

Code, experimental materials and results for comparing explicit metaphor judgements with downstream continuation alignment in `Qwen/Qwen3.5-9B`.

The experiment uses the official post-trained checkpoint at revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. RQ2 uses the official chat template with `enable_thinking=False` and scores the fixed responses `A` and `B`. The formal dataset contains 880 triples, 595 target items and 553 source-sentence clusters.

- [Experimental protocol](docs/protocol_en.md)
- [Formal results](docs/formal_20260908T224807Z_summary.md)
- [Figures, tables and source data](figures/README.md)
- [Figure gallery](figures/index.html)

## Repository structure

```text
src/qwen_alignment/                  Core implementation
runpod/experiments/                  Experimental, analysis and plotting entry points
data/processed/                      Retained formal materials
data/controls/                       Functional controls
data/reference/                      Shared standardisation statistics
data/smoke/                          Small-scale test inputs
vendor/                              Original third-party materials and source manifests
results/formal/20260908T224807Z/       Complete formal machine outputs
figures/                             Main figures, supplementary figures, tables and source data
docs/figure_validation/               Archived figure validation reports
docs/                                Protocol and results documentation
environment/                         Recorded execution environment
runpod/                              GPU deployment and execution scripts
tests/                               Automated tests
```

`runpod/experiments/` contains executable analysis code; `results/` contains its outputs. All 17 core modules remain in `src/qwen_alignment/`. `docs/protocol.md` is an identical English copy of `docs/protocol_en.md`, retained for compatibility with the packaging scripts.

Run local commands from this repository's root. Python entry points resolve the root from their own location. RunPod scripts use the fixed deployment root `/workspace/qwen3p5-9b-metaphor-predictive-alignment`. Each deployment keeps its model snapshot, tokenizer artifacts, reference statistics, checkpoints and results inside that root.

## Tests and dependencies

Python 3.10–3.12 is required. Local dependencies are listed in `requirements-local.txt`; `tests/test_analysis_core.py` additionally requires PyTorch. With these dependencies installed, run:

```bash
python -m pytest
```

RunPod uses a PyTorch container with a CUDA-compatible torch build. `requirements-runpod.txt` supplies the remaining pinned dependencies; the bootstrap script records and validates the installed torch build.

## Build the RunPod source bundle

The following inputs are included:

- `data/processed/analysis_items.csv`
- `data/smoke/munch_smoke_subset.csv`
- `data/controls/qwen_instruct_direct_controls.json`
- `vendor/metaphor-understanding-challenge/correct_answers/for_judgement.csv`
- `vendor/metaphor-understanding-challenge/correct_answers/for_generation.csv`
- `vendor/metaphor-understanding-challenge/source_manifest.json`
- `vendor/ias-naturalstories/texts_400words.csv`
- `vendor/ias-naturalstories/source_manifest.json`

```bash
python runpod/experiments/build_transfer_bundle.py
```

The command creates `transfer/qwen3p5-9b-metaphor-predictive-alignment.tar.gz` and prints its SHA-256. The bundle contains only allowlisted code, inputs and protocols. It excludes saved results, model caches, generated reference statistics and environment reports.

## Upload and initialise

Use a persistent volume mounted at `/workspace`. The recorded setup used region `EU-RO-1`, a 50 GB network volume named `qwen35-9b-metaphor-data` and a 30 GB container disk. Preserve the volume when deleting a Pod.

Create the destination, upload the source bundle into its `transfer/` directory, and verify the printed SHA-256 before extraction:

```bash
mkdir -p /workspace/qwen3p5-9b-metaphor-predictive-alignment/transfer
sha256sum /workspace/qwen3p5-9b-metaphor-predictive-alignment/transfer/qwen3p5-9b-metaphor-predictive-alignment.tar.gz
tar -xzf /workspace/qwen3p5-9b-metaphor-predictive-alignment/transfer/qwen3p5-9b-metaphor-predictive-alignment.tar.gz -C /workspace
cd /workspace/qwen3p5-9b-metaphor-predictive-alignment
bash runpod/bootstrap.sh
```

Bootstrap creates the local environment and caches, runs unit tests, downloads the fixed model revision with an explicit `--allow-download`, and checks the BF16 single-GPU runtime. It also audits tokenizer boundaries for all 880 inputs. Subsequent stages load the local snapshot offline.

## Smoke test and runtime estimate

```bash
bash runpod/smoke.sh
```

The initial configuration uses one RTX 5090 with 32 GB VRAM. BF16 weights occupy approximately 16.7 GiB before runtime allocations. An RTX 4090 with 24 GB is an optional probe, subject to the same smoke and memory checks. If the 32 GB configuration fails, use a 48 GB GPU such as an L40S, RTX A6000 or A40 without changing scientific parameters or batch settings.

The smoke pipeline:

1. Builds and validates the model-specific IAS reference statistics, reusing existing valid files.
2. Generates RQ1 continuations for three triples and computes their distances.
3. Runs the three-item surprisal check and 12-control RQ2 check. The positive-score threshold applies to contextual controls; word-only scores are checked for completeness and finite values.
4. Benchmarks all three GPU stages on representative formal inputs. Continuation uses 12 prefix-length positions and a separate longest-prefix probe with 32 returns. Surprisal and RQ2 use 12 items covering their actual token-length distributions, including the longest item. Each RQ2 item covers both inputs, both candidate orders and both responses; benchmark scores are discarded.

The script prints `SMOKE_RUN_DIR`, `BENCHMARK_SUMMARY` and `END_TO_END_PLANNING_SUMMARY`. `BENCHMARK_SUMMARY` covers continuation only. The end-to-end report includes continuation, distance, surprisal, RQ2, one model load per stage and a CPU finishing allowance.

Formal execution requires `gpu_memory_gate.status=pass`, measured on the same GPU model. The gate uses peak reserved memory from the longest-prefix, 32-return probe. Its limit is `min(total measured VRAM - 2 GiB, floor(total measured VRAM × 0.92))`. Missing memory measurements or an out-of-memory error stop execution. A passing report is saved as `environment/latest_end_to_end_planning.json`.

Surprisal and RQ2 scoring time is extrapolated from 12 to 880 items. Distance computation is extrapolated from three to 880 triples after subtracting one measured model load. Planning upper estimates multiply computation time by 1.20 and add one model load per formal stage. The 12 RQ2 controls are a fixed gate cost. Pod startup, installation, downloading and file transfer are excluded. GPU prices and availability are not fixed in the repository; compare current hourly cost with the report's planning upper duration.

## Formal execution and resumption

After a passing smoke run and review of the runtime estimate:

```bash
bash runpod/formal.sh new
```

Save the printed `FORMAL_RUN_DIR`. To resume on a Pod mounting the same persistent volume:

```bash
bash runpod/formal.sh resume /workspace/qwen3p5-9b-metaphor-predictive-alignment/results/formal/<run-tag>
```

The stages produce continuations, reference-standardised distances, RQ1 inference, local surprisal, matched contextual/word-only judgements and RQ2 regression results. Each completed item-condition-seed group is committed to SQLite and skipped on resumption.

Before terminating the Pod, archive the complete run directory and compare its SHA-256 after downloading:

```bash
PROJECT_ROOT=/workspace/qwen3p5-9b-metaphor-predictive-alignment
FORMAL_RUN_DIR=$PROJECT_ROOT/results/formal/<run-tag>
RESULT_BUNDLE=$PROJECT_ROOT/transfer/<run-tag>-results.tar.gz
tar -czf "$RESULT_BUNDLE" -C "$FORMAL_RUN_DIR" .
sha256sum "$RESULT_BUNDLE"
```

The transfer directory is excluded from subsequent source bundles.

## Included formal run

`results/formal/20260908T224807Z/` contains the complete saved run, including continuation records, its SQLite checkpoint, distances, direct scores, RQ1 and RQ2 results, surprisal outputs, stage timings and the run log. See the [results summary](docs/formal_20260908T224807Z_summary.md) for interpretation and the [figure documentation](figures/README.md) for CPU-only reproduction.

Saved machine manifests and historical validation reports retain their original execution paths. The executable entry points and documentation use the reorganised paths.
