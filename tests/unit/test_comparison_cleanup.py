from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from benchmarks.cartesian_comparison import finalize_comparison


class ArtifactClient:
    def __init__(self, remote: Path, corrupt: bool = False) -> None:
        self.remote = remote
        self.corrupt = corrupt
        self.remote.mkdir()

    def search_experiments(self, **kwargs: object) -> list:
        return []

    def search_runs(self, *args: object, **kwargs: object) -> list:
        return []

    def get_run(self, run_id: str) -> SimpleNamespace:
        return SimpleNamespace(info=SimpleNamespace(status="FINISHED"))

    def log_artifact(self, run_id: str, path: str, artifact_path: str) -> None:
        destination = self.remote / artifact_path / Path(path).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)

    def log_artifacts(self, run_id: str, path: str, artifact_path: str) -> None:
        shutil.copytree(path, self.remote / artifact_path, dirs_exist_ok=True)

    def log_dict(self, run_id: str, value: dict, artifact_path: str) -> None:
        destination = self.remote / artifact_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(value), encoding="utf-8")

    def log_metric(self, *args: object, **kwargs: object) -> None:
        pass

    def download_artifacts(self, run_id: str, artifact_path: str, destination: str) -> str:
        target = Path(destination) / Path(artifact_path).name
        shutil.copyfile(self.remote / artifact_path, target)
        if self.corrupt and artifact_path == "checkpoints/best.pt":
            target.write_bytes(b"corrupted")
        return str(target)


class ComparisonCleanupTests(unittest.TestCase):
    def setUp(self) -> None:
        project = Path(__file__).resolve().parents[2]
        (project / "outputs").mkdir(exist_ok=True)
        self.local = tempfile.TemporaryDirectory(prefix="cleanup-test-", dir=project / "outputs")
        self.extra = tempfile.TemporaryDirectory(prefix="angle-artifact-test-")
        self.output = Path(self.local.name)
        self.report = Path(self.extra.name) / "report"
        self.remote = Path(self.extra.name) / "remote"
        self.evaluation = self.output / "plain_seed42"
        self.evaluation.mkdir()
        (self.evaluation / "metrics.json").write_text('{"mae_deg": 0.01}')
        (self.evaluation / "validation.npz").write_bytes(b"validation evidence")
        checkpoints = self.output / "checkpoints"
        checkpoints.mkdir()
        for name in ("best.pt", "last.pt"):
            (checkpoints / name).write_bytes(name.encode())
        self.state = {
            "status": "COMPLETE",
            "child_pid": None,
            "results": [
                {
                    "key": "plain_seed42",
                    "run_id": "a" * 32,
                    "stage": "plain",
                    "seed": 42,
                    "total_steps": 15600,
                    "checkpoint": str(checkpoints / "best.pt"),
                    "metrics": {"seed": 42, "samples": 15000, "mae_deg": 0.01},
                }
            ],
        }
        (self.output / "state.json").write_text(json.dumps(self.state))
        (self.output / "preflight.json").write_text("[]")

    def tearDown(self) -> None:
        self.local.cleanup()
        self.extra.cleanup()

    def client(self, corrupt: bool = False) -> ArtifactClient:
        client = ArtifactClient(self.remote, corrupt)
        for path in (self.output / "checkpoints").iterdir():
            client.log_artifact("a" * 32, str(path), "checkpoints")
        return client

    def test_verified_artifacts_survive_staging_cleanup(self) -> None:
        finalize_comparison(self.client(), self.output, self.report)
        self.assertFalse(self.output.exists())
        self.assertEqual((self.remote / "checkpoints/best.pt").read_bytes(), b"best.pt")
        self.assertEqual(
            (self.remote / "comparison/evaluation/validation.npz").read_bytes(),
            b"validation evidence",
        )
        self.assertIn("15600", (self.report / "results.csv").read_text())
        manifest = json.loads(
            (self.remote / "comparison/campaign/artifact-verification.json").read_text()
        )
        self.assertEqual(manifest["runs"][0]["run_id"], "a" * 32)
        self.assertIn("checkpoints/best.pt", manifest["runs"][0]["verified_artifacts"])

    def test_corrupt_download_retains_local_checkpoints(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "artifact differs"):
            finalize_comparison(self.client(corrupt=True), self.output, self.report)
        self.assertEqual((self.output / "checkpoints/best.pt").read_bytes(), b"best.pt")
        self.assertTrue((self.evaluation / "validation.npz").exists())

    def test_active_campaign_cannot_be_cleaned(self) -> None:
        self.state.update(status="TRAINING", child_pid=123)
        (self.output / "state.json").write_text(json.dumps(self.state))
        with self.assertRaisesRegex(RuntimeError, "GPU work"):
            finalize_comparison(self.client(), self.output, self.report)
        self.assertTrue(self.output.exists())

    def test_completed_report_is_not_overwritten(self) -> None:
        self.report.mkdir()
        (self.report / "results.csv").write_text("previous completed results\n")
        with self.assertRaisesRegex(RuntimeError, "Existing completed report"):
            finalize_comparison(self.client(), self.output, self.report)
        self.assertEqual((self.report / "results.csv").read_text(), "previous completed results\n")
        self.assertTrue(self.output.exists())
