# Four-view polar refinement

The primary model estimates line orientation with ConvNeXt Tiny and a local polar correction constrained by two symmetries. Across three training seeds, four-view refinement improves MAE and P99 over its angular-only parent at the same 15 600 task updates. The model keeps 34 033 768 parameters. It performs more fine-branch computation and retains a worst-case limitation on seed 44.

## Results

The primary comparison evaluates both symmetry models in the same BF16 session on the same 15 000 validation images within each seed. Coarse checkpoints and training allocation match. All errors are in degrees.

| Seed | Angular MAE | Four-view MAE | MAE reduction | Angular P99 | Four-view P99 | P99 reduction | Angular maximum | Four-view maximum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.00175261 | 0.00161541 | 7.83% | 0.01622031 | 0.01490980 | 8.08% | 0.514739 | 0.338185 |
| 43 | 0.00168102 | 0.00150575 | 10.43% | 0.01534281 | 0.01388664 | 9.49% | 0.511127 | 0.141791 |
| 44 | 0.00194083 | 0.00169758 | 12.53% | 0.01653096 | 0.01571952 | 4.91% | 1.033664 | 0.512762 |

![Paired symmetry results across three seeds](figures/symmetry.png)

Compared with the earlier single-view polar V1, MAE falls by 14.07%, 24.61% and 18.06%. P99 falls by 10.97%, 23.70% and 16.64%. These secondary comparisons use matching validation indices from different BF16 replay sessions. The three-seed same-session comparison above is stronger evidence for the final change.

| Seed | Previous V1 MAE | Four-view MAE | Previous V1 P99 | Four-view P99 | Previous V1 maximum | Four-view maximum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.00187992 | 0.00161541 | 0.01674777 | 0.01490980 | 0.426159 | 0.338185 |
| 43 | 0.00199735 | 0.00150575 | 0.01820029 | 0.01388664 | 1.050881 | 0.141791 |
| 44 | 0.00207186 | 0.00169758 | 0.01885795 | 0.01571952 | 0.414949 | 0.512762 |

The maximum on seed 44 is 23.57% worse than previous V1. The selection prioritizes repeatable improvement of MAE and P99, without claiming improvement on every rare case.

The earlier RGB controls also received 15 600 task updates. Relative to stock RGB ConvNeXt Tiny, the primary model reduces MAE by 95.06% and 95.36%, and P99 by 94.64% and 94.57%, on seeds 42 and 43. These compare complete pipelines and different replay sessions. They do not isolate the effect of the polar transform or match computational cost.

![RGB controls and current polar model](figures/quality.png)

## Geometry before additional capacity

The target is an undirected line passing through the image center. Signed polar sampling represents both halves of that line within one angular column. A 256 × 256 RGB image becomes a 384 × 360 polar image. ConvNeXt Tiny and an angular regression head estimate the global direction. Fine samples a 384 × 65 crop spanning ±4° around that estimate.

The remaining error is a local signed correction. Reversing the crop's angular columns should reverse its sign. Reversing signed radial rows should preserve it. An unconstrained MLP has to learn these relationships from examples. The final readout imposes them directly.

For crop C, angular reversal A, radial reversal R and shared bounded predictor b:

```text
b(C) = 4° × tanh(z(C))
δ(C) = [b(C) − b(AC) + b(RC) − b(ARC)] / 4
δ(AC) = −δ(C)
δ(RC) = δ(C)
```

All views use the same CNN, MLP, radial coordinate channels and coarse conditioning. Coarse runs once. The correction rotates the coarse double-angle vector, preserving continuity at 0°/180°. The identities apply to the sampled local crop at a fixed coarse estimate. They do not make the complete raster pipeline globally rotation equivariant.

This is a task-specific finite-group output projection. The mathematical basis follows the equivariance definition in [Cohen and Welling](https://proceedings.mlr.press/v48/cohenc16.pdf) and the equivariant Reynolds operator in [Sannai, Kawano and Kumagai](https://arxiv.org/html/2110.08092v1). The papers motivate the construction. Accuracy evidence comes from the controlled comparisons here, rather than from those papers.

![Primary model architecture](figures/architecture.png)

The shared fine CNN has channels 16/32/128, GroupNorm and SiLU. A fixed radial antialias filter precedes each 5 × 23 convolution. Strides 2 × 1 reduce radial resolution while retaining all 65 angular positions. Radial mean and maximum feed an MLP with dimensions 16 642 → 128 → 1. No trainable parameters are added by the four-view projection.

## How the architecture was selected

The [initial architecture search](../architecture-search/README.md) established the detailed local crop, anisotropic convolutions and frozen coarse-to-fine training. Follow-up comparisons tested specific explanations for the remaining errors.

| Direction | Observation | Decision |
| --- | --- | --- |
| EfficientViT B2 | Did not preserve the selected precision on both seeds | Retain ConvNeXt Tiny |
| Less early radial downsampling | Seed 42 MAE worsened by 6.85%, seed 43 improved by only 0.18% | Retain strides 2/2/2 |
| Direct sampling from original RGB | Mean error improved, but P99 and maxima did not improve consistently | Keep as a separate alternative |
| Radial attention | Did not improve paired accuracy, sampled weights remained nearly uniform | Retain mean/max pooling |
| Fixed, gated and routed backbone feedback | No paired MAE gain, some large coarse failures | Preserve the reliable coarse estimator |
| Shallow detail skip, wider early layers and altered pooling | No consistent joint improvement of MAE and P99 | Retain the compact fine encoder |
| Heatmap readout and learned line moments | Failed the paired accuracy criterion | Retain continuous MLP correction |
| Radial reflection alone | MAE improved by 3.87% / 3.15%, with mixed P99 and worse maxima | Test its interaction with angular sign structure |
| Joint continuation of reflection | MAE worsened by 7.20% / 2.66% against frozen continuation | Keep coarse frozen |
| Angular antisymmetry | MAE and P99 improved on both seeds | Use as the parent for the final comparison |
| Angular antisymmetry plus radial reflection | MAE and P99 improved on seeds 42, 43 and 44 | Select four-view refinement |

The [search summary](data/search-summary.csv) retains the exported final-stage and reference measurements from the follow-up studies. Repeated reference rows are separate replay sessions, not additional trained models. The percentages above compare each candidate with its own matched reference. They cannot be added together.

## Training

Every final model has the same allocation of unique task updates:

| Stage | Initialization | Updated weights | New updates | Cumulative updates |
| --- | --- | --- | ---: | ---: |
| Coarse | ImageNet ConvNeXt Tiny | Coarse backbone and angular head | 5 200 | 5 200 |
| Fresh fine | Own-seed coarse checkpoint | New fine CNN and MLP | 5 200 | 10 400 |
| Fine continuation | Complete fresh-fine checkpoint | Fine CNN and MLP | 5 200 | 15 600 |

Each stage resets MuSGD, uses 200 warmup steps and cosine decay. Physical batch size is 32. CNN computation uses BF16. Sampling, regression heads and angle geometry use FP32. Vector Charbonnier loss, data, renderer, labels, preprocessing and augmentation remain unchanged during these architecture controls.

The continuation is a second 5 200-update cosine cycle rather than one uninterrupted 10 400-update fine schedule. Each seed uses its own 120 000 / 15 000 / 15 000 partition. Coarse stays in evaluation mode during both fine stages. All 193 coarse tensors and three preprocessing buffers match their source bitwise.

All completed stages reached the full budget and validation milestones. Checkpoint identities, unique ancestry, portable inference and artifact checksums were verified. One interrupted seed 44 continuation was restarted from its completed intermediate checkpoint. Its partial 1 700 updates were discarded. They consumed compute but do not contribute to the selected model's 15 600-update ancestry.

## Precision and computation

| Model | Conv/linear MACs per image | GFLOPs, two operations per MAC | Parameters |
| --- | ---: | ---: | ---: |
| Previous single-view polar V1 | 14.09 billion | 28.17 | 34 033 768 |
| Angular-only refinement | 16.04 billion | 32.08 | 34 033 768 |
| Primary four-view refinement | 19.95 billion | 39.90 | 34 033 768 |

Counts exclude activations, reductions, sampling and data transfers. Four-view refinement has 41.6% more convolution and linear operations than previous V1. Fine executes four shared views in one batch, while coarse executes once.

Measured batch-one latency on an RTX 5070 was 14.34 / 14.57 / 14.84 ms for four views, versus 14.17 / 14.49 / 14.82 ms for the angular-only parent in matching sessions. Inputs were already on GPU. Preprocessing is included, while file loading and host-to-device transfer are excluded. Small latency differences may include measurement noise. Other devices or larger batches can show a different cost increase.

## Where the gains and failures occur

Matched ordinary images, the weakest contrast quintile, seam and near-vertical groups improve on all three seeds. On seed 44, ordinary MAE falls from 0.00151176° to 0.00135727° and weak-line MAE from 0.00418452° to 0.00346370°. The contrast group is a pixel proxy along the labelled direction, not an occlusion annotation.

Errors above 0.1° change from 6 to 5, from 4 to 3 and from 6 to 7 for seeds 42, 43 and 44. Seed 44 image 21 897 near 179.83° regresses from 0.450961° to 0.512762°. Image 78 351 near 2.77° regresses from 0.310324° to 0.381108°. Every target in the saved coarse replay lies inside the ±4° correction window. A missed window therefore does not explain these failures.

The following examples reuse the three indices from the earlier RGB report, now with actual predictions from the primary seed 42 checkpoint. They illustrate behavior rather than estimate performance.

![Clear line](figures/clear-line.png)

![Faint line](figures/faint-line.png)

![Line near the seam](figures/near-seam.png)

## Numerical checks and limits

Historical coarse BF16 exports differed from the recovered seed 44 process despite identical weights. An independent sequential replay captured the actual coarse and fine outputs of both final models. Coarse outputs matched exactly between the paired models within each precision, and reconstructed errors agreed within 0.00000257°.

| Diagnostic precision | Angular MAE | Four-view MAE | Angular P99 | Four-view P99 | Angular maximum | Four-view maximum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 0.00193950 | 0.00169768 | 0.01675047 | 0.01576793 | 1.031651 | 0.512745 |
| FP32 | 0.00192934 | 0.00169502 | 0.01654226 | 0.01582077 | 0.945150 | 0.514886 |

TF32 was disabled for this diagnostic. No weights changed and no training updates occurred. The advantage survives both checks. The official results remain the original BF16 exports, without replacing them with the diagnostic values.

Validation was used repeatedly to select architectures. Seed 44 was introduced after fixing both symmetry models, but had already appeared in an earlier RGB experiment. It is a replication rather than a pristine held-out test. The test split remains unused. Real-image evaluation is still needed before making broader generalization claims.

The [selection manifest](../../../configs/final/selection.json) records checkpoint identifiers, hashes, architecture and training ancestry. Seed 42 remains the default checkpoint rather than choosing the lowest-error seed. The published release still contains the preceding model.
