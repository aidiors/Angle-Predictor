# Dataset manifests

There is no separate dataset manifest in this directory yet. Current experiment
YAML files identify the dataset with `dataset.name`, `dataset.version`, and
`dataset.source`; `params.data.root` points to the local files, and each run
records the dataset metadata digest in MLflow. If a standalone manifest is
added, keep it version-controlled and small; store large dataset files outside
Git.
