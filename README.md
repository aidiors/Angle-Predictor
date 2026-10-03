# Angle-Predictor

[English](README.md) · [Русский](README.ru.md)

Estimating the angle of a thin line in cluttered images with signed polar
coordinates and a coarse-to-fine neural network.

The final model reaches **0.001880° and 0.001995° validation MAE** on two
seeds. At the same 15 600-step training budget, it reduces error by
**45.29% and 41.66%** compared with the earlier sequential model.

## Results

Each model below received **15 600 total task-training updates**, including
the training of any reused checkpoints. Each cell reports seed 42 / seed 43.
All errors are in degrees.

| Model | MAE ↓ | P99 ↓ | Maximum ↓ |
| --- | ---: | ---: | ---: |
| Earlier coarse → fine → joint model | 0.003437 / 0.003419 | 0.024526 / 0.024363 | 0.227046 / 0.676586 |
| Coarse and fine trained jointly from ImageNet initialization | 0.006239 / 0.007074 | 0.043590 / 0.045649 | 3.353607 / 3.762028 |
| Improved refinement with 64 final CNN channels, frozen coarse | 0.001972 / 0.002058 | 0.018069 / 0.018716 | 0.459905 / 1.026583 |
| **Final refinement with 128 final CNN channels, frozen coarse** | **0.001880 / 0.001995** | **0.016715 / 0.018106** | 0.423626 / 1.050872 |

The final model was selected for MAE and P99. Joint fine-tuning of a different
64-channel refinement model produced smaller maximum errors of 0.191248° and
0.357315°, but worse mean accuracy. The search exposed this tradeoff rather
than finding one model that won on every metric.

![Precision and rare errors at equal training cost](docs/reports/architecture-search/figures/precision-and-tail.png)

## What changed during the research

The starting point was an ImageNet-initialized ConvNeXt Tiny operating on a
polar image. Its tuned standalone MAE was 0.010441° and 0.010615° after
5 200 steps. The work covered data generation, angle representation, loss and
optimizer selection, then 113 completed architecture trials.

Smaller backbones and pruned Tiny variants reduced computation but did not
preserve accuracy under the tested recipe. Error analysis showed two different
problems: choosing the correct line among distractors and measuring its angle
precisely. A local correction branch improved the second task while retaining
Tiny for the first.

The refinement experiments then changed one part of the representation at a
time and compared against the corresponding parent at equal training cost.

| Change | Why it was tested | MAE reduction, seed 42 / 43 |
| --- | --- | ---: |
| Preserve 384 radial crop samples instead of 192 | Retain detail from faint and interrupted segments | 12.03% / 7.16% |
| Increase the last fine layer from 32 to 64 channels | Give the local encoder more capacity | 7.41% / 9.36% |
| Filter before radial downsampling | Reduce sensitivity to the sampled feature rows | 5.74% / 6.20% |
| Use 65 angular samples with a proportionally wider kernel | Preserve small angular shifts while retaining context | 20.79% / 16.01% |
| Increase the last fine layer from 64 to 128 channels after matched continuation | Improve the final local representation and correction projection | 4.63% / 3.09% |

The first four rows use 10 400 total steps. The last uses 15 600.
These are separate comparisons, so their percentages should not be added.
The final 23-sample angular kernel covers the full refinement window.
Larger radial kernels, extra pooling bins, a denser final radial feature map
and alternative correction heads did not provide consistent further gains.

![Refinement accuracy across matching training budgets](docs/reports/architecture-search/figures/equal-budget-evolution.png)

Earlier experiments established the training recipe. Vector Charbonnier
reduced mean error by **25.4% relative to MSE across ten seeds**. MuSGD had
the lowest median error among three optimizer families under the tested
hyperparameter search. Its subsequent learning-rate and weight-decay tuning
produced the standalone Tiny baseline used in the architecture comparisons.
These studies used different comparisons and are reported separately from
the architectural gains.

## Why polar coordinates help

In the original image, changing a line's angle changes its slope across both
pixel axes. A small rotation moves pixels near the ends more than those near
the center. Measuring a very small angular difference therefore requires
combining evidence from distant parts of the segment.

The polar transform reorganizes that evidence around the known image center.
Each column samples one orientation, and each row samples a distance along
that orientation:

```text
P(r, θ) = I(cx + r cos θ, cy + r sin θ)
```

Here `I` is the RGB image and `(cx, cy)` is its center. The radius `r` runs
from negative to positive values, so one column follows an entire diameter
through the center. Angles cover 0° to 180°. This signed radius represents
both halves of an undirected line without needing a separate 360° image.
The sampler uses the largest circle inside the input square, leaving the
corners outside the representation.

A line through the center becomes a narrow band along the radial axis,
located at its orientation. For example, a 30° line appears near the 30°
column. Rotating it to 31° moves its response along the angular axis instead
of giving the network a new diagonal pattern. In the 360-column input, that
rotation corresponds to two columns. Finite thickness and interpolation
spread the response across neighboring columns.

This arrangement gives the architecture three useful properties:

- **Evidence along the line can be combined directly.** Radial pooling
  collects support from different parts of the segment while retaining
  angular position. This is useful when some parts are faint or occluded
- **The fine branch can inspect a narrow angular neighborhood.** After the
  coarse prediction, it samples ±4° while retaining the full radial extent.
  It can compare nearby orientations using distant visible parts of the line
- **The representation matches the CNN's sampling strategy.** The fine
  encoder downsamples along radius but preserves angular positions. Its
  convolutions compare neighboring angle responses before a continuous
  correction head estimates the remaining offset

The global image has a 0.5° angular pitch. The final crop uses 65 interpolated
samples across 8°, giving a 0.125° pitch. These are sampling intervals,
not limits on the regression output. Bilinear sampling and the continuous
correction allow sub-column estimates, but interpolation adds no new source
detail.

The geometry relies on the target passing through the center. An off-center
line becomes a curved trace, and clutter can still produce competing
responses. Polar coordinates organize the evidence, while the learned coarse
branch decides which line to follow. The architecture comparisons above keep
this transform fixed, so their measured gains belong to the refinement and
training changes.

## Final architecture

![Coarse estimation and local angle correction](docs/reports/architecture-search/figures/final-architecture.png)

An undirected line has the same orientation at θ and θ + 180°.
The model predicts **[sin(2θ), cos(2θ)]**, avoiding a discontinuity at that
boundary. A 256 × 256 RGB image is resampled to a **384 × 360 signed polar
image** around its center. A center-crossing line then aligns along the
radial dimension.

ConvNeXt Tiny and a custom angular regression head estimate the global
direction. The fine branch samples the original polar pixels in a
**384 × 65 crop spanning ±4°** around that prediction. It reflects the radial
axis when crossing the 0°/180° seam, following
`P(r, θ + π) = P(−r, θ)`.

| Component | Design |
| --- | --- |
| Coarse estimator | Full ConvNeXt Tiny with an attention-based angular head |
| Fine input | RGB plus signed and absolute radial coordinates |
| Fine encoder | Three CNN blocks, 16 → 32 → 128 channels |
| Downsampling | Fixed radial filter followed by 5 × 23 convolutions with stride 2 × 1 |
| Aggregation | Radial mean and maximum, retaining all angular positions |
| Correction head | MLP 16 642 → 128 → 1, conditioned on the coarse vector |
| Output | Continuous correction δ = 4° × tanh(z), applied to the coarse vector |
| Parameters | 34 033 768, an 8.51% increase over standalone Tiny |

The fine branch measures from the detailed polar image instead of the
backbone's compressed feature map. Dense angular sampling is interpolation,
not extra source resolution. The output correction remains continuous.

## Training and evaluation

The selected model uses two training regimes across three 5 200-step runs:

1. Train the coarse backbone and head from ImageNet initialization
2. Freeze coarse and train a newly initialized refinement branch
3. Continue fine-only training from the complete preceding checkpoint

Each run resets MuSGD and uses 200 warmup steps followed by cosine decay.
The total is **15 600 updates**. Coarse stays in evaluation mode during both
fine runs, and all 193 coarse parameter tensors were verified unchanged on
both seeds. CNN computation uses BF16, with FP32 sampling, regression heads
and final angle geometry.

The `synthetic_lines_150k` dataset contains 150 000 images with finite target
lines, distracting geometry, background layers, occlusion and noise.
Each seed defines a 120 000 / 15 000 / 15 000 train, validation and test split.
Comparisons match validation indices within each seed and preserve the data,
labels, loss, optimizer and preprocessing. The independent test split was
not used during model selection.

On an RTX 5070, recorded single-image latency was **13.26 ms and 13.03 ms**,
compared with 10.90 ms and 10.75 ms for standalone coarse in the same paired
measurements. These PyTorch BF16 timings include preprocessing with inputs
already on the GPU, excluding image loading and host-to-device transfer.

## Reports and measurements

| Study | Evidence |
| --- | --- |
| [Architecture search and final selection](docs/reports/architecture-search/README.md) | Equal-budget comparisons, architecture details, difficult cases and latency |
| [Backbone comparison](docs/reports/backbone-comparison/README.md) | Compact backbones, local refinement and global line-selection failures |
| [Joint fine-tuning and shared features](docs/reports/end-to-end-comparison/README.md) | Sequential pretraining, joint continuation and multiscale fusion |
| [PolarLine](docs/reports/polar-line-architecture/README.md) | A much smaller model that traded precision for speed |
| [Loss comparison](docs/reports/loss-function-comparison/README.md) | Five losses, ten matched seeds and paired differences |
| [Optimizer comparison](docs/reports/optimizer-comparison-seed42-43/README.md) | Three optimizer families and 60 completed runs |
| [MuSGD tuning](docs/reports/musgd-tuning-seed42-43/README.md) | Sixteen configurations repeated on two seeds |
| [Augmentation throughput](docs/augmentation-speed.md) | GPU preprocessing and data-loading measurements |

The [architecture results table](docs/reports/architecture-search/data/results.csv)
and [selected model measurements](docs/reports/architecture-search/data/selected-model.json)
provide the numerical results behind the main comparison.

## Running the project

Python 3.14 and uv are required. The training dependency group installs
PyTorch with CUDA support on Windows and Linux.

```sh
uv sync --locked
docker compose -f infra/mlflow/compose.yaml up -d
uv run angle-train run --config configs/final/coarse.yaml
```

Training requires a local dataset containing `images.dat`, `labels.dat` and
`meta.json` at the configured location. Dataset files, checkpoints and the
visual assets used by the generator are excluded from Git. The generator
code is included, but reproducing the exact image distribution requires
the original local assets.

The three [final-model configurations](configs/final/README.md) describe the
coarse, fresh fine and continuation stages. Fine training requires the
preceding checkpoint and its actual run identifier. Checkpoint loading checks
the data split, dataset fingerprint, preprocessing and architecture.

For inference with a complete checkpoint:

```sh
uv run angle-predict --checkpoint checkpoints/model.pt --input image.png --device cuda
```

CPU inference is also available. An inference image must be 256 × 256 RGB
with the target line passing through the center.

```sh
uv run ruff check .
uv run ruff format --check .
uv run ty check src
uv run python -m unittest discover -s tests
```

## Limits

The reported accuracy is synthetic validation accuracy. Repeated adaptive
use of validation and two architecture seeds leave uncertainty about unseen
images. An independent test evaluation and real-image evaluation are needed
for a broader generalization claim.

Refinement is bounded to ±4° and cannot reliably rescue a coarse prediction
that selects a different line. Very faint lines and orientations near the
polar seam remain difficult. The final model improves typical precision,
while some jointly fine-tuned alternatives have smaller extreme errors.

## License

The project code is released under [Apache License 2.0](LICENSE).
