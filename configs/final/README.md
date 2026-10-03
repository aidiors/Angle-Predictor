# Final model training configurations

These configurations reproduce the selected training recipe using local
checkpoints. Each stage completes 5 200 steps and retains its checkpoints.
Model registration is disabled. Seeds default to 42.

| Configuration | Initialization | Trainable weights | Total task steps |
| --- | --- | --- | ---: |
| [coarse.yaml](coarse.yaml) | ImageNet ConvNeXt Tiny | Coarse backbone and head | 5 200 |
| [refinement.yaml](refinement.yaml) | The same seed's coarse checkpoint | Fine CNN and correction MLP | 10 400 |
| [continuation.yaml](continuation.yaml) | The same seed's complete refinement checkpoint | Fine CNN and correction MLP | 15 600 |

Run from the project root with the dataset at
`data/datasets/synthetic_lines_150k` and MLflow available at the configured URI.

```sh
uv run angle-train run --config configs/final/coarse.yaml
```

The run prints its MLflow identifier. Copy its `best.pt` from
`outputs/experiments/coarse_seed42/<run-id>/` to `checkpoints/coarse.pt`.
Use that identifier when starting fresh fine training:

```sh
uv run angle-train run --config configs/final/refinement.yaml --set params.model.coarse_run_id=<coarse-run-id>
```

Copy the resulting best checkpoint to `checkpoints/refinement.pt`.
Keep the coarse identifier and supply the refinement run identifier for
continuation:

```sh
uv run angle-train run --config configs/final/continuation.yaml --set params.model.coarse_run_id=<coarse-run-id> --set params.training.initialization_run_id=<refinement-run-id>
```

Replace the angle-bracket placeholders with actual identifiers.
The zero-filled identifiers in YAML are placeholders and must be replaced.
The optimizer resets at each stage, while the previous model weights are
preserved. This is two fine-training cycles with matching cosine schedules.

For seed 43, change both `seed` and `params.data.split_seed` to 43.
In the fine configurations, also change `params.model.coarse_split_seed`.
Use new run names and checkpoints trained entirely on that seed's partition.
Changing the dataset root requires the same location throughout the chain.
