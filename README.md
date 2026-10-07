# Angle-Predictor

[English](README.md) · [Русский](README.ru.md)

Angle-Predictor estimates the orientation of a thin line passing through the
center of a cluttered image. It uses signed polar coordinates, ConvNeXt Tiny
for an initial estimate and a local CNN to refine the angle.

The primary model, V2, uses four shared views of a local polar crop. It reaches
**0.001615°, 0.001506° and 0.001698° validation MAE** on seeds 42, 43 and 44.
At the same 15 600-update training budget, MAE is **95.06% and 95.36% lower**
than the standard RGB ConvNeXt Tiny control on seeds 42 and 43.

## Results

Each model received **15 600 total task-training updates**, including any reused
checkpoints. Each cell reports seed 42 / seed 43 on 15 000 matching validation
images per seed. All errors are in degrees. MAE is the mean absolute angular
error. P99 is the error below which 99% of predictions fall.

| Model | MAE ↓ | P99 ↓ | Maximum ↓ |
| --- | ---: | ---: | ---: |
| Plain ConvNeXt Tiny on RGB | 0.032679 / 0.032459 | 0.278369 / 0.255570 | 2.702152 / 4.141962 |
| Spatial head and local RGB refinement | 0.009159 / 0.005435 | 0.041323 / 0.042328 | 60.444824 / 3.282609 |
| Previous single-view polar model | 0.001880 / 0.001997 | 0.016748 / 0.018200 | 0.426159 / 1.050881 |
| **Primary four-view polar model** | **0.001615 / 0.001506** | **0.014910 / 0.013887** | **0.338185 / 0.141791** |

The RGB control keeps the stock ConvNeXt architecture apart from its two angle
outputs. The RGB refinement model adds a spatial head and local Cartesian refinement
without polar resampling. The primary pipeline reduces P99 error by
**94.64% and 94.57%** relative to the stock RGB control. The four-view results
and RGB controls were evaluated on the same images within each seed, in separate
BF16 sessions. This comparison measures the complete pipelines at equal update
counts. Their computational costs differ, and polar geometry is one of several
changes.

To measure the last architectural change, radial reflection was added to a fine
head that already enforced angular antisymmetry. Both models were evaluated in
the same session on 15 000 images per seed.

| Seed | Angular MAE → four-view MAE (°) | MAE reduction | Angular P99 → four-view P99 (°) | P99 reduction |
| --- | ---: | ---: | ---: | ---: |
| 42 | 0.001753 → 0.001615 | 7.83% | 0.016220 → 0.014910 | 8.08% |
| 43 | 0.001681 → 0.001506 | 10.43% | 0.015343 → 0.013887 | 9.49% |
| 44 | 0.001941 → 0.001698 | 12.53% | 0.016531 → 0.015720 | 4.91% |

The four-view maximum on seed 44 is 0.512762°, compared with 0.414949° for
previous V1. Lower MAE and P99 do not imply improvement on every rare case.
The [three-seed report](docs/reports/four-view-refinement/README.md)
includes maximum errors, difficult subsets and independent BF16/FP32 checks.

![Accuracy compared with plain ConvNeXt](docs/reports/four-view-refinement/figures/quality.png)

The [RGB and polar comparison report](docs/reports/cartesian-comparison/README.md)
covers training curves, difficult subsets, large errors and computational cost.

## Prediction examples

These validation examples show the input, its polar representation and the
predictions from plain ConvNeXt and the four-view model. They use the same three
images as the earlier RGB comparison. Green dashed lines mark the reference
orientation, and red lines mark the prediction. The labels show errors that are
too small to distinguish visually.

![Clear line and predicted angle](docs/reports/four-view-refinement/figures/clear-line.png)

![Faint line and predicted angle](docs/reports/four-view-refinement/figures/faint-line.png)

![Difficult line near the angular seam](docs/reports/four-view-refinement/figures/near-seam.png)

## What changed during the research

The starting point was an ImageNet-initialized ConvNeXt Tiny operating on a
polar image. Its tuned standalone MAE was 0.010441° and 0.010615° after
5 200 steps. The research included a synthetic data generator, comparisons of
angle representations, losses and optimizers, and an initial architecture search
with 121 completed training runs. Later experiments focused on local geometry
and symmetry.

Smaller backbones and pruned Tiny variants reduced computation but did not
preserve accuracy under the tested recipe. Error analysis showed two different
problems: choosing the correct line among distractors and measuring its angle
precisely. A local correction branch improved the second task while retaining
Tiny for the first.

Each refinement variant was compared with its parent at the same update budget.

| Change | Why it was tested | MAE reduction, seed 42 / 43 |
| --- | --- | ---: |
| Preserve 384 radial crop samples instead of 192 | Retain detail from faint and interrupted segments | 12.03% / 7.16% |
| Increase the last fine layer from 32 to 64 channels | Give the local encoder more capacity | 7.41% / 9.36% |
| Filter before radial downsampling | Reduce sensitivity to the sampled feature rows | 5.74% / 6.20% |
| Use 65 angular samples with a proportionally wider kernel | Preserve small angular shifts while retaining context | 20.79% / 16.01% |
| Increase the last fine layer from 64 to 128 channels after matched continuation | Improve the final local representation and correction projection | 4.63% / 3.09% |
| Enforce angular antisymmetry | Reverse the correction when angular columns are reversed | 6.77% / 15.84% |
| Add radial reflection to angular antisymmetry | Preserve the correction when the signed radial direction is reversed | 7.83% / 10.43% |

The first four rows use 10 400 total steps. The remaining rows use 15 600.
These are separate comparisons, so their percentages should not be added.
Three angular convolutions with 23-sample kernels give a 67-sample receptive
field, covering the 65-column refinement window.
Larger radial kernels, extra pooling bins, a denser final radial feature map
and alternative correction heads did not provide consistent further gains.

![Refinement accuracy across matching training budgets](docs/reports/four-view-refinement/figures/symmetry.png)

Earlier experiments established the training recipe. Vector Charbonnier
reduced mean error by **25.4% relative to MSE across ten seeds**. MuSGD had
the lowest median error among three optimizer families under the tested
hyperparameter search. Its subsequent learning-rate and weight-decay tuning
produced the standalone Tiny baseline used in the architecture comparisons.

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

Radial pooling combines evidence from different parts of the segment while
retaining angular position. Visible sections can still support an estimate when
other sections are faint or occluded.

After the initial prediction, the fine branch samples a narrow ±4° neighborhood
across the full radial extent. Its CNN reduces resolution along the radius while
preserving angular columns, so nearby orientations remain distinguishable.
The correction head then estimates a continuous offset.

The global image has a 0.5° angular pitch. The final crop uses 65 interpolated
samples across 8°, giving a 0.125° pitch. These are sampling intervals,
not limits on the regression output. Bilinear sampling and the continuous
correction allow sub-column estimates, but interpolation adds no new source
detail.

The geometry relies on the target passing through the center. An off-center
line becomes a curved trace, and clutter can still produce competing
responses. Polar coordinates organize the evidence, while the learned coarse
branch decides which line to follow.

## Final architecture

![Coarse estimation and local angle correction](docs/reports/four-view-refinement/figures/architecture.png)

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
| Fine encoder | Three CNN blocks with GroupNorm and SiLU, 16 → 32 → 128 channels |
| Shared views | Original crop, angular reversal, radial reversal and both reversals |
| Downsampling | Fixed radial filter followed by 5 × 23 convolutions with stride 2 × 1 |
| Aggregation | Radial mean and maximum, retaining all angular positions |
| Correction head | MLP 16 642 → 128 → 1, conditioned on the coarse vector |
| Per-view readout | b(C) = 4° × tanh(z(C)), through the same CNN and MLP |
| Output | δ = [b(C) − b(AC) + b(RC) − b(ARC)] / 4, applied to the coarse vector |
| Parameters | 34 033 768, 22.33% more than the stock RGB ConvNeXt Tiny |

Here A reverses angular columns and R reverses radial rows. The combined
correction changes sign under A and stays the same under R, with the coarse
estimate held fixed. Coarse runs once, and all four fine views share weights.
The four-view readout therefore adds no parameters to the single-view head.
These local constraints do not imply global image-rotation equivariance.

The fine branch measures from the detailed polar image instead of the
backbone's compressed feature map.

## Training and evaluation

Training has three stages of 5 200 updates each:

1. Train the coarse backbone and head from ImageNet initialization
2. Freeze coarse and train a newly initialized refinement branch
3. Continue fine-only training from the complete preceding checkpoint

Each run resets MuSGD and uses 200 warmup steps followed by cosine decay.
The total is **15 600 updates**. Coarse stays in evaluation mode during both
fine runs, and all 193 coarse parameter tensors were verified unchanged on
all three seeds. CNN computation uses BF16, with FP32 sampling, regression heads
and final angle geometry.

The `synthetic_lines_150k` dataset contains 150 000 images with finite target
lines, distracting geometry, background layers, occlusion and noise.
Each seed defines a 120 000 / 15 000 / 15 000 train, validation and test split.
Comparisons match validation indices within each seed and preserve the data,
labels, loss, optimizer, augmentation and normalization settings. The test split
was not used during model selection.

On an RTX 5070, single-image latency for the primary model was
**14.34 / 14.57 / 14.84 ms** on seeds 42 / 43 / 44, compared with
14.17 / 14.49 / 14.82 ms for the angular-only parent in matching sessions.
These PyTorch BF16 medians include preprocessing with inputs already on the GPU,
excluding image loading and host-to-device transfer. Small differences can
include measurement noise.

Convolutions and linear layers require about **39.90 GFLOPs per image**,
compared with 28.17 for the previous single-view polar model, an increase of
41.6%. This counts two operations per multiply-accumulate and excludes
activations, reductions and sampling. Four views increase computation even
when GPU batch-one latency changes little.

## Reports and measurements

| Study | Evidence |
| --- | --- |
| [Current selection across three seeds](docs/reports/four-view-refinement/README.md) | Four-view correction, matched symmetry comparisons, tail limitations and numerical checks |
| [RGB controls and polar refinement](docs/reports/cartesian-comparison/README.md) | Stock ConvNeXt, Cartesian refinement, matched budgets and prediction examples |
| [Initial architecture search](docs/reports/architecture-search/README.md) | Equal-budget comparisons, architecture details, difficult cases and latency |
| [Backbone comparison](docs/reports/backbone-comparison/README.md) | Compact backbones, local refinement and global line-selection failures |
| [Joint fine-tuning and shared features](docs/reports/end-to-end-comparison/README.md) | Sequential pretraining, joint continuation and multiscale fusion |
| [PolarLine](docs/reports/polar-line-architecture/README.md) | A much smaller model that traded precision for speed |
| [Loss comparison](docs/reports/loss-function-comparison/README.md) | Five losses, ten matched seeds and paired differences |
| [Optimizer comparison](docs/reports/optimizer-comparison-seed42-43/README.md) | Three optimizer families and 60 completed runs |
| [MuSGD tuning](docs/reports/musgd-tuning-seed42-43/README.md) | Sixteen configurations repeated on two seeds |
| [Augmentation throughput](docs/augmentation-speed.md) | GPU preprocessing and data-loading measurements |

The [architecture results table](docs/reports/architecture-search/data/results.csv)
and [historical V1 measurements](docs/reports/architecture-search/data/selected-model.json)
retain the initial search evidence. The current
[selection manifest](configs/final/selection.json) identifies all three finished
checkpoints, their hashes and validation results. Seed 42 is the default
checkpoint.

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

The [model release](https://github.com/aidiors/Angle-Predictor/releases/tag/v0.2.0)
includes both versions:

- [V2 weights](https://github.com/aidiors/Angle-Predictor/releases/download/v0.2.0/angle-predictor-v2.pt): the primary four-view model
- [V1 weights](https://github.com/aidiors/Angle-Predictor/releases/download/v0.2.0/angle-predictor-v1.pt): the preceding single-view model

Both are complete seed 42 checkpoints with preprocessing and architecture
metadata. They do not require the dataset or earlier checkpoints for inference.
The release includes SHA-256 checksums. Use V2 by default:

```sh
uv run angle-predict --checkpoint angle-predictor-v2.pt --input image.png --device cuda
```

The [measurement tools](benchmarks/README.md) count operations, compare
validation checkpoints and regenerate the current figures from saved evidence.

CPU inference is also available. An inference image must be 256 × 256 RGB
with the target line passing through the center.

```sh
uv sync --locked --all-groups
uv run ruff check .
uv run ruff format --check .
uv run ty check src benchmarks --extra-search-path .
uv run python -m unittest discover -s tests
```

## Limits

These results come from synthetic validation images. The architecture was
selected using validation results, so the three-seed comparison does not replace
an independent test. Performance on real images also needs separate evaluation.

Refinement is bounded to ±4° and cannot reliably rescue a coarse prediction
that selects a different line. Very faint lines and orientations near the
polar seam remain difficult. The final model improves typical precision,
while the previous V1 has a smaller maximum error on seed 44.

## License

The project code is released under [Apache License 2.0](LICENSE).
