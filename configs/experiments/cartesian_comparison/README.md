# Cartesian controls

These controls use the same RGB pixels, splits, augmentations, double-angle
labels, Vector Charbonnier loss and tuned MuSGD settings as the polar study.
The input remains 256 × 256. Normalization is unchanged, and polar resampling
is disabled.

The plain model keeps ConvNeXt Tiny's standard backbone, global pooling and
classifier structure. Its classification layer outputs two regression values,
which are normalized to represent `[sin(2θ), cos(2θ)]`. It trains from ImageNet
initialization for 15 600 steps with one cosine schedule.

The enhanced model retains the complete Tiny backbone and replaces global
pooling with a 4 × 4 spatial representation and a 256-unit regression head.
After 5 200 steps, coarse is frozen. A new refinement encoder trains for
5 200 steps, then continues for another 5 200 with the optimizer reset.
Each stage uses 200 warmup steps and its own cosine schedule.

Refinement samples a rotated rectangle through the input center. Its length
is 255 pixels and width is 32 pixels, sampled onto a 384 × 65 grid. The grid
is affine with parallel, equally spaced rows and columns. No column represents
an angle, and no polar image is constructed. The encoder receives RGB and two
rectangle coordinates. It uses longitudinal filtering, three CNN blocks with
16 → 32 → 128 channels, longitudinal mean/max pooling and a continuous ±4°
correction. Rotation aligns the rectangle with the coarse prediction.

Both final Cartesian models have 15 600 total task updates. Seeds 42 and 43
use their own 15 000-image validation splits. The existing best polar model
is replayed on those same indices. The independent test split is untouched.
This compares complete recipes. It does not isolate the transform alone,
because the heads, local geometry and training schedules differ.

The checkpoint paths and run IDs in the fine configurations are placeholders.
The comparison runner fills them from completed own-seed stages and checks
full training histories, validation milestones and frozen coarse weights.

```sh
uv run python benchmarks/cartesian_comparison.py preflight
uv run python benchmarks/cartesian_comparison.py run --reference-runs <seed42-run> <seed43-run>
```

The runner executes one GPU task at a time. The ignored
`outputs/cartesian-comparison` folder is temporary staging. Checkpoints,
validation predictions, metrics, configuration and logs are retained in MLflow.
The runner verifies downloaded artifact hashes before removing staging after
a successful comparison. Compact measurements belong in
`docs/reports/cartesian-comparison`.

Use `--keep-staging` when preparing additional plots from local predictions,
then run the `finalize` command to archive and clean them. Failed comparisons
retain staging for diagnosis and resumption. Preflight weights are discarded.
Each started training run completes its declared budget before evaluation
and profiling.
