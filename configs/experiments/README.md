# Experiment configurations

The [final training recipe](../final/README.md) contains the three stages
used by the selected architecture. Its configurations preserve local
checkpoints and disable model registration.

The standalone experiment configurations retain the controls used to select
the loss and optimizer:

| Configuration | Purpose |
| --- | --- |
| [Tuned Tiny baseline](convnext_tiny_musgd_vector_charbonnier_tuned.yaml) | MuSGD settings selected on two seeds |
| [Reference Tiny baseline](convnext_tiny_musgd_vector_charbonnier.yaml) | Fixed settings for the loss comparison |
| [Angular Huber](convnext_tiny_musgd_angular_huber.yaml) | Alternative angular loss |
| [PolarLine](polar_line_musgd_vector_charbonnier.yaml) | Compact model retaining angular resolution |

The dataset must exist at `params.data.root` and match the configured version.
Training and validation indices use `params.data.split_seed`. Checkpoints
from a different partition cannot initialize a matching refinement run.

Each run completes its declared budget and selects `best.pt` by validation
angular MAE. Set `params.training.keep_local_checkpoints=true` to retain local
checkpoints after MLflow logging. `last.pt` stores the final training state.

A setting can be overridden without editing the configuration:

```sh
uv run angle-train run --config configs/final/coarse.yaml --set params.data.num_workers=4
```

The selected recipe uses batch size 32, BF16, Vector Charbonnier with ε = 0.1,
MuSGD, 200 warmup steps and cosine decay. Exact optimizer values are retained
in YAML. The [architecture report](../../docs/reports/architecture-search/README.md)
explains the comparisons, and the
[augmentation benchmark](../../docs/augmentation-speed.md) describes the
data-loading measurements.

Historical standalone controls can register a model when
`tracking.registered_model_name` is set. A successful registration updates
its candidate alias. Use `null` for experiments that should only log results,
as in the final-model configurations.
