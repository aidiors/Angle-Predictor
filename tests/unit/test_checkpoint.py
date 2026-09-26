from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch
from torch import nn, optim

from dl_template.engine.checkpoint import save_checkpoint


class CheckpointTests(TestCase):
    def test_checkpoint_contains_resume_state(self) -> None:
        model = nn.Linear(2, 1)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "checkpoint.pt"
            save_checkpoint(model, path, optimizer=optimizer, epoch=3, metadata={"name": "test"})
            payload = torch.load(path, weights_only=False)

        self.assertEqual(payload["epoch"], 3)
        self.assertEqual(payload["metadata"], {"name": "test"})
        self.assertIn("model_state_dict", payload)
        self.assertIn("optimizer_state_dict", payload)
