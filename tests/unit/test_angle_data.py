import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
import torch

from angle_predictor.data.angle_dataset import AngleMemmapDataset, make_split_indices
from angle_predictor.data.angle_transforms import (
    AngleBatchPreprocessor,
    SignedPolarTransform,
    build_polar_grid,
    pad_signed_polar_angle,
)


class AngleDatasetTests(TestCase):
    def test_memmap_splits_are_disjoint_reproducible_and_read_correct_samples(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            count, size = 20, 5
            images = np.memmap(
                root / "images.dat", dtype=np.uint8, mode="w+", shape=(count, size, size, 3)
            )
            labels = np.memmap(root / "labels.dat", dtype=np.float32, mode="w+", shape=(count, 2))
            for index in range(count):
                images[index] = index
                labels[index] = (math.sin(2 * index), math.cos(2 * index))
            images.flush()
            labels.flush()
            del images, labels
            (root / "meta.json").write_text(
                json.dumps(
                    {
                        "num_samples": count,
                        "img_size": size,
                        "images_shape": [count, size, size, 3],
                        "labels_shape": [count, 2],
                        "images_dtype": "uint8",
                        "labels_dtype": "float32",
                        "images_channels": "RGB",
                        "labels_format": ["sin(2*line_angle_rad)", "cos(2*line_angle_rad)"],
                        "angle_convention": {
                            "angle_range": "[0, pi)",
                            "coordinate_system": "image coordinates: x right, y down",
                            "orientation_period": (
                                "pi; theta and theta + pi are the same undirected line"
                            ),
                        },
                    }
                ),
                encoding="utf-8",
            )

            splits = {
                name: AngleMemmapDataset(root, name, seed=7) for name in ("train", "val", "test")
            }
            all_indices = [int(i) for dataset in splits.values() for i in dataset.indices]
            self.assertEqual(sorted(all_indices), list(range(count)))
            self.assertTrue(
                np.array_equal(splits["train"].indices, make_split_indices(count, seed=7)["train"])
            )
            self.assertEqual([len(splits[name]) for name in ("train", "val", "test")], [16, 2, 2])

            dataset = splits["train"]
            image, label = dataset[0]
            source_index = int(dataset.indices[0])
            self.assertEqual(image.shape, (3, size, size))
            self.assertEqual(image.dtype, torch.uint8)
            self.assertEqual(label.dtype, torch.float32)
            self.assertTrue(torch.all(image == source_index))
            self.assertAlmostEqual(float(label[0]), math.sin(2 * source_index), places=6)
            self.assertIsNone(dataset.__getstate__()["_images"])
            dataset.close()

    def test_rejects_corrupt_file_size(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "meta.json").write_text(
                json.dumps(
                    {
                        "num_samples": 1,
                        "img_size": 4,
                        "images_shape": [1, 4, 4, 3],
                        "labels_shape": [1, 2],
                        "images_dtype": "uint8",
                        "labels_dtype": "float32",
                        "images_channels": "RGB",
                        "labels_format": ["sin(2*line_angle_rad)", "cos(2*line_angle_rad)"],
                        "angle_convention": {
                            "angle_range": "[0, pi)",
                            "coordinate_system": "image coordinates: x right, y down",
                            "orientation_period": (
                                "pi; theta and theta + pi are the same undirected line"
                            ),
                        },
                    }
                ),
                encoding="utf-8",
            )
            (root / "images.dat").write_bytes(b"short")
            (root / "labels.dat").write_bytes(bytes(8))
            with self.assertRaisesRegex(ValueError, "images.dat size"):
                AngleMemmapDataset(root, "train")


class AngleTransformTests(TestCase):
    def test_polar_grid_stays_inside_image_and_constant_image_has_no_padding_seam(self) -> None:
        grid = build_polar_grid((65, 81), (96, 180))
        self.assertEqual(grid.shape, (1, 96, 180, 2))
        self.assertGreaterEqual(float(grid.min()), -1)
        self.assertLessEqual(float(grid.max()), 1)
        transform = SignedPolarTransform((65, 81), (96, 180))
        output = transform(torch.ones(2, 3, 65, 81))
        self.assertEqual(output.shape, (2, 3, 96, 180))
        self.assertTrue(torch.allclose(output, torch.ones_like(output), atol=1e-6))

    def test_line_becomes_stripe_at_its_angle(self) -> None:
        size, angle = 65, math.pi / 6
        yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
        distance = (xx - 32) * math.sin(angle) - (yy - 32) * math.cos(angle)
        image = (distance.abs() < 0.55).float()[None]
        polar = SignedPolarTransform((size, size), (96, 180))(image)
        self.assertGreater(float(polar[0, :, 30].mean()), 0.4)
        self.assertGreater(float(polar[0, :, 30].mean()), float(polar[0, :, 90].mean()) * 4)

    def test_each_flip_updates_sine_and_preserves_cosine(self) -> None:
        image = torch.arange(3 * 9 * 9, dtype=torch.uint8).reshape(1, 3, 9, 9)
        target = torch.tensor([[math.sqrt(3) / 2, 0.5]])
        for horizontal, vertical in ((1, 0), (0, 1), (1, 1)):
            with self.subTest(horizontal=horizontal, vertical=vertical):
                transform = AngleBatchPreprocessor(
                    (9, 9),
                    (12, 18),
                    horizontal_flip_probability=horizontal,
                    vertical_flip_probability=vertical,
                    jitter_probability=0,
                    gaussian_probability=0,
                    speckle_probability=0,
                )
                result, actual_target = transform(image, target, augment=True)
                expected_image = image.flip(-1) if horizontal else image
                expected_image = expected_image.flip(-2) if vertical else expected_image
                expected, _ = transform(expected_image, target, augment=False)
                self.assertTrue(torch.allclose(result, expected))
                assert actual_target is not None
                expected_sign = -1 if horizontal ^ vertical else 1
                self.assertTrue(
                    torch.allclose(
                        actual_target,
                        torch.tensor([[expected_sign * math.sqrt(3) / 2, 0.5]]),
                    )
                )

    def test_signed_polar_angular_padding_flips_radius_at_seam(self) -> None:
        image = torch.tensor([[[[1, 2, 3], [4, 5, 6]]]])
        padded = pad_signed_polar_angle(image, 1)
        expected = torch.tensor([[[[6, 1, 2, 3, 4], [3, 4, 5, 6, 1]]]])
        self.assertTrue(torch.equal(padded, expected))
