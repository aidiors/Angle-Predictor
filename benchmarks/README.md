# Measurements and report figures

These tools measure inference, compare validation predictions and regenerate
the figures used in the portfolio. The primary architecture is defined in
`configs/final/refinement.yaml`. Checkpoint identities and hashes are listed in
`configs/final/selection.json`.

| Tool | Purpose | Requirements |
| --- | --- | --- |
| `architecture_speed.py` | Count convolution and linear operations, measure batch-one and batch-32 latency | Config only for operation counts, CUDA for timing |
| `architecture_validation.py` | Compare two finished checkpoints on identical validation indices | MLflow artifacts and local dataset, CPU or CUDA |
| `backbone_error_forensics.py` | Inspect weak-line groups, angular seams and largest regressions | Paired validation CSV and local dataset |
| `cartesian_comparison.py` | Reproduce stock ConvNeXt and Cartesian refinement controls | Local dataset, CUDA and MLflow |
| `model_report.py` | Regenerate current accuracy charts, architecture diagram and prediction examples | Saved report data, Matplotlib and optional local dataset |

## Operation counts and latency

Operation counting uses tensor shapes without loading weights or running CUDA:

```sh
uv run python -m benchmarks.architecture_speed --operations-only --output artifacts/operations.json
```

The default comparison is the standalone polar coarse estimator against the
primary four-view model. To time those architectures on synthetic input:

```sh
uv run python -m benchmarks.architecture_speed --output artifacts/speed.json
```

Timing uses randomly initialized weights and inputs already on GPU. It includes
preprocessing and excludes image loading and host-to-device transfer. It does
not measure prediction accuracy. Training-step timing is available only through
the explicit `--training-step` option.

## Paired validation and error analysis

Use actual finished MLflow run identifiers. The tool checks the data partition,
preprocessing and exact validation indices before accepting a pair. Downloaded
checkpoints use a temporary directory that is removed after the comparison.
An explicit `--cache-dir` retains them for repeated use.

```sh
uv run python -m benchmarks.architecture_validation --baseline-run BASELINE_ID --candidate-run CANDIDATE_ID --output-dir artifacts/validation --export-predictions
uv run python -m benchmarks.backbone_error_forensics --result-dir artifacts/validation
```

Pass `--device cpu` for FP32 CPU evaluation. CUDA follows the checkpoint's
recorded inference precision. Neither mode evaluates the test split.

## Current report figures

The plot generator reads the committed CSV and JSON evidence. It does not
load model weights, train or run inference:

```sh
uv run --with matplotlib python -m benchmarks.model_report
```

To also recreate the three illustrated validation images:

```sh
uv run --with matplotlib python -m benchmarks.model_report --dataset data/datasets/synthetic_lines_150k
```

PNG and SVG files are written to `docs/reports/four-view-refinement/figures`.
Example angles come from the saved predictions of the primary seed 42 model.
The dataset supplies the image pixels. Historical report figures remain
unchanged.

## RGB controls

`cartesian_comparison.py` retains the matched control training and artifact
archival procedure from the original study. Its commands and configuration
layout are described in
[the control configurations](../configs/experiments/cartesian_comparison/README.md).
It is a reproduction tool rather than an autonomous experiment scheduler.
