from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from dl_template.config.loader import load_config
from dl_template.config.schema import ExperimentConfig

EXPERIMENT_CONFIG = """\
name: test-experiment
entrypoint: my_project.experiment:run
tracking:
  experiment_name: test-project
  system_metrics: false
dataset:
  name: test-dataset
  version: v1
  source: local
params:
  optimizer: adamw
"""


class ConfigTests(TestCase):
    def test_experiment_config_loads(self) -> None:
        config = self._load_config()

        self.assertEqual(config.name, "test-experiment")
        self.assertEqual(config.params["optimizer"], "adamw")
        self.assertFalse(config.tracking.system_metrics)

    def test_dotted_override_is_typed_by_yaml(self) -> None:
        config = self._load_config(["seed=7", "params.learning_rate=0.001"])

        self.assertEqual(config.seed, 7)
        self.assertAlmostEqual(config.params["learning_rate"], 0.001)

    def _load_config(self, overrides: list[str] | None = None) -> ExperimentConfig:
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "experiment.yaml"
            config_path.write_text(EXPERIMENT_CONFIG, encoding="utf-8")
            return load_config(config_path, overrides)
