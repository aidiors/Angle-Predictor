# Augmentation and data loading speed

For independent per-image augmentations, the custom PyTorch preprocessor on
GPU was the fastest option. Four DataLoader workers gave the best loading
speed among the settings tested.

Measurements used `synthetic_lines_150k` (RGB `uint8`, 256×256, memmap).
The machine ran Windows with an NVIDIA GeForce RTX 5070 (12 GiB) and
16 logical CPU cores. Software versions: Python 3.14.7, PyTorch 2.14.0+cu130,
TorchVision 0.29.0+cu130, Kornia 0.8.3 and Albumentations 2.0.8.

## Methods compared

- `AngleBatchPreprocessor` on GPU. It processes the batch with tensor
  operations: converts `uint8` to `float`, flips images and adjusts their
  labels, changes colors, adds two types of noise, applies a signed-polar
  transform through `grid_sample` and normalizes the result. Random
  parameters are sampled independently for each image.
- TorchVision v2 on GPU. ColorJitter and GaussianNoise are called separately
  for each image to keep random parameters independent. Flips, speckle noise,
  the polar transform and normalization use the same tensor code as the
  custom preprocessor.
- Kornia on GPU. `ColorJitter` and `RandomGaussianNoise` use
  `same_on_batch=False`. The remaining operations are the same.
- PyTorch, TorchVision v2 and Albumentations on CPU inside Dataset/DataLoader,
  followed by batch transfer to GPU for the polar transform and normalization.

The libraries differ in the details and order of color transforms, as well
as noise implementations and ranges. CPU versions return `uint8`, adding
rounding before the batch is assembled. These results compare the speed of
similar operations without showing that they produce identical images.

A separate measurement covers `torchvision_gpu_shared`, which calls
TorchVision v2 once for the entire batch. Random choices are shared across
images in this mode, so augmentation diversity differs from the current
preprocessor.

## Individual call timings

Input tensors are already on the device being measured. After four warmup
calls, the reported time is the median of 12–20 repetitions. CUDA is
synchronized before reading the elapsed time. Full preprocessor timings
include the polar transform and normalization. Preparation with augmentations
and data transfer are also measured separately.

| Operation | Batch 32, ms | Batch 64, ms |
| --- | ---: | ---: |
| Custom PyTorch, GPU, full training pipeline | 2.4–3.1 | 6.3 |
| Custom PyTorch, GPU, no augmentations | 0.76–1.05 | 1.86 |
| Custom PyTorch, GPU, preparation and augmentations without polar transform or normalization | 2.28 | 4.74 |
| Copy from pinned CPU memory to GPU | 0.55 | 0.99 |
| TorchVision v2, GPU, separate call per image | 15.8–15.9 | 31.1 |
| TorchVision v2, GPU, one call per batch¹ | 2.1–2.2 | 5.08 |
| Kornia, GPU, independent parameters | 4.6–5.4 | 7.0 |
| Custom PyTorch, CPU, full pipeline² | 159–166 | 327 |

¹ Random choices are shared across the batch. This mode does not preserve
our current augmentation policy.

² Includes the polar transform on CPU, so this is not a timing of CPU
augmentations alone.

## DataLoader with memmap

Settings: `batch_size` 32 or 64, `shuffle=True`, `pin_memory=True`, transfers
with `non_blocking=True` and `prefetch_factor=2` when workers > 0. Three warmup
batches are followed by 64 measured batches of size 32 or 32 batches of
size 64, giving 2048 images per run.

Timings include memmap reads, batch assembly, augmentations, transfers,
the polar transform and normalization. Model forward and backward passes
and worker startup are excluded. Data may have been in the filesystem cache.

| Mode | Workers | Images/s, batch 32 | Images/s, batch 64 |
| --- | ---: | ---: | ---: |
| PyTorch, GPU augmentations | 0 | 4386 | 4653 |
| PyTorch, GPU augmentations | 2 | 6603 | 6909 |
| PyTorch, GPU augmentations | 4 | **8055** | **7901** |
| Kornia, GPU augmentations | 2 | 5106 | 6612 |
| Kornia, GPU augmentations | 4 | 5271 | 6566 |
| TorchVision v2, GPU, per image | 2 | 1899 | not measured |
| Albumentations, CPU augmentations | 2 | 1301 | 1527 |
| TorchVision v2, CPU augmentations | 2 | 308 | not measured |

## Project settings

Keep augmentations, the polar transform and normalization in
`AngleBatchPreprocessor` on GPU. DataLoader reads `uint8` without expensive
processing. For CUDA, use `pin_memory=True` and transfers with
`non_blocking=True`.
