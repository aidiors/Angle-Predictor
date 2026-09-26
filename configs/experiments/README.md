# Experiment configs

Add one YAML file for each reproducible experiment configuration. The common
fields are validated by `ExperimentConfig`; the task entrypoint validates its
own `params` fields.

The template does not include a task-specific config or runnable example. Set
`entrypoint` to a function in the task package you add under `src/`.
