# Evaluation

Internal evaluation and fine-tuning suite.

The package is intentionally small and registry-based. A YAML config selects:

- a task from `evaluation.tasks`
- a backbone from `evaluation.backbones`
- a trainer mode from `evaluation.trainers`
- a prediction head from `evaluation.heads`

The CLI loads the config, applies optional dot-list overrides, builds those
components, and runs the trainer.

## Run
To run evaluation, use:
```bash
uv run python -m evaluation.main --config <path_to_your_config>
```

Here's an example:

```bash
uv run python -m evaluation.main --config src/evaluation/config/fomo_brain_age_gap_probe.yaml
```

Override config values from the command line with OmegaConf dot-list syntax (although I suggest you create a new config instead of overriding values in cli):

```bash
uv run python -m evaluation.main \
  --config src/evaluation/config/fomo_brain_age_gap_probe.yaml \
  name=probe_cls \
  task.name=fomo_brain_age_gap \
  model.checkpoint_path=/path/to/checkpoint.pt \
  optimization.epochs=10 \
  device=cpu
```

Outputs are written under:

```text
<output_dir>/<name>/
```

For the shipped configs this resolves to `data/runs/evaluation/<name>/`.
The probe trainer writes:

- `metrics.json`: best epoch, best validation score, validation history, final
  validation metrics, and test metrics
- `predictions.csv`: test-set prediction and target values
- `head-best.pt`: state dict for the best head checkpoint

The scikit probe trainer writes:

- `metrics.json`: selected estimator, selected alpha, validation history, final
  validation metrics, and test metrics
- `predictions.csv`: test-set prediction and target values
- `features.npz`: raw pooled train/validation/test features and targets
- `model.joblib`: serialized scikit-learn pipeline

## Config Shape

The current configs use this top-level structure:

```yaml
name: eval_probe
output_dir: data/runs/evaluation

task:
  name: fomo_brain_age_gap
  overwrite_data: false

model:
  name: smri_mae

transforms: null

mode:
  name: probe

representation: cls

head:
  name: linear
  pooling: first

optimization:
  epochs: 50
  batch_size: 8
  lr: 1e-3
  weight_decay: 0.0
  num_workers: 0

evaluation:
  selection_metric: mae
  selection_mode: min

device: cuda
seed: 7338
```

`representation` selects one token sequence returned by the backbone. The
current backbones use names such as `cls`, `reg`, and `patch`. If the selected
representation is missing or `None`, the trainer raises an error listing the
available representations. These representations are derived from MAE model and can be expanded.

`evaluation.selection_metric` must be a metric returned by the task. For the
current regression task, available metrics are `mae`, `rmse`, `bias`, and `r2`.
`selection_mode` is `min` for lower-is-better metrics or `max` for
higher-is-better metrics. It's used for selecting the "best" checkpoint.

## Supported Modes

### Trainer Modes

Registered trainer modes:

| Config value | Trainer | Status |
| --- | --- | --- |
| `mode.name: probe` | `ProbeTrainer` | Supported for regression targets |
| `mode.name: scikit_probe` | `ScikitProbeTrainer` | Supported for regression targets |

`probe` freezes the backbone, trains only the head with AdamW and MSE loss,
selects the best head by validation metric, reloads that head, then evaluates
on validation and test splits.

Classification targets are represented by `TargetSpec(kind="classification",
...)`, but `ProbeTrainer` currently raises `NotImplementedError` for them. It will be properly implemented when a classification tasks gets implemented.

`scikit_probe` freezes the backbone, extracts pooled features once for each
split, saves them to `features.npz`, and fits a scikit-learn estimator on those
features. The first supported estimator is `ridge`, implemented as
`StandardScaler` followed by `Ridge`. If `mode.alphas` is omitted, the default
Ridge grid is:

```yaml
mode:
  name: scikit_probe
  estimator: ridge
  alphas: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
```

Use a single-value list to fit one alpha:

```yaml
mode:
  name: scikit_probe
  estimator: ridge
  alphas: [1.0]
```

Additional scikit estimators should be added to the estimator registry in
`evaluation.trainers`.

### Heads

Registered heads:

| Config value | Class | Options | Input |
| --- | --- | --- | --- |
| `head.name: linear` | `LinearHead` | `pooling: first` or `pooling: mean` | token tensor shaped `[B, T, D]` |

`LinearHead` pools a token sequence into `[B, D]`, then applies one linear
layer with output width equal to `task.target_spec().dim`.

Heads will be expanded with additional implementations - MLP, attention pooling etc.

### Backbones

Registered backbones:

| Config value | Class | Purpose |
| --- | --- | --- |
| `model.name: smri_mae` | `SmriMaeBackbone` | Adapter around `smri_mae.model_mae.MaskedViT` |

`SmriMaeBackbone` builds `MaskedViT` with:

- `img_size`
- `patch_size`
- `in_chans`, defaulting to `1`
- `model_kwargs`, passed through to `MaskedViT`
- optional `checkpoint_path`, loaded with `strict=False`
- optional `use_input_mask: true`, which expects each batch to include `mask`
- optional `calculate_mask: mean`, which computes
  `mask = image > image.mean()` per volume

`use_input_mask` and `calculate_mask` cannot both be enabled.

### Tasks

Registered tasks:

| Config value | Class | Status |
| --- | --- | --- |
| `task.name: fomo_brain_age_gap` | `FomoBrainAgeGapTask` | FOMO26 task 3 regression from asparagus-preprocessed tensors |

A task owns data preparation, split construction, collation, target metadata,
and metric computation.

`fomo_brain_age_gap` reads the standard asparagus task directory
`data/asparagus/data/REGR002_FOMO26_BrainAge` by default. It uses
`split_80_10_10.json` for train/validation folds and `TEST_80_10_10.json` for
test samples. The dataset returns stored tensors as-is: no resizing, padding,
cropping, or intensity normalization is applied inside the task.

## Batch Contract

Every dataset sample or collated batch must provide:

- `image`
- `target`

Optional keys are allowed. The trainer currently forwards `mask` to the
backbone when present. Other optional keys such as `id`, `meta`, or
`covariates` can be included by tasks for future use.

The task must return a `DatasetBundle` with `train`, `val`, and `test`
datasets. Its `target_spec()` controls head output dimension and tells the
trainer whether the task is regression or classification.

## Transforms

Top-level `transforms` is optional. When omitted or set to `null`, task samples
are passed through unchanged. The first supported transform is:

```yaml
transforms:
  name: pad_center_crop
  key: image
  size: [208, 240, 208]
  pad_value: 0.0
```

Transforms receive the full sample dict and return the full sample dict. By
default `pad_center_crop` only modifies `sample["image"]`; all other keys such
as `target`, `id`, and `meta` are preserved.

If you need other transformations, feel free to add them.

## Add a Task

1. Create a task class implementing the `EvaluationTask` protocol in
   `evaluation.core`.
2. Implement `prepare(overwrite_data: bool = False)`, `target_spec()`,
   `datasets()`, `collate_fn()`, and `metrics(predictions, targets)`.
3. Make `datasets()` return a `DatasetBundle(train=..., val=..., test=...)`.
4. Ensure batches include at least `image` and `target`.
5. Register the task in `_TASK_REGISTRY` in `evaluation.tasks.__init__`.
6. Select it in YAML with:

```yaml
task:
  name: your_task_name
```

For a new regression task, return
`TargetSpec(kind="regression", dim=..., loss="mse")` and metrics compatible
with the configured `evaluation.selection_metric`.

## Add a Backbone

1. Add an `nn.Module` adapter in `evaluation.backbones`.
2. Expose an `embed_dim` attribute. The head builder uses it as `input_dim`.
3. Make `forward(...)` return a dictionary of named token sequences:

```python
{
    "cls": cls_tokens,    # [B, T, D] or None
    "reg": reg_tokens,    # [B, T, D] or None
    "patch": patch_tokens # [B, T, D] or None
}
```

4. Keep pooling out of the backbone. Heads own pooling.
5. Add a `_build_<name>_backbone(cfg)` function that reads config values and
   returns the adapter.
6. Register the builder in `_BACKBONE_BUILDERS`.
7. Add focused tests for builder registration, output representation names and
   shapes, checkpoint loading, and any mask behavior.
8. Select it in YAML with:

```yaml
model:
  name: your_backbone_name
```

If the backbone supports masks, accept `mask=None` in `forward`; the probe
trainer forwards `batch["mask"]` when it exists.

## Add a Head

1. Add an `nn.Module` in `evaluation.heads`.
2. Define the input shape it accepts. Existing trainers pass a selected
   representation tensor directly to the head.
3. Add a `_build_<name>_head(cfg, *, target_spec, input_dim)` function.
4. Use `target_spec.dim` for the output width unless the head has a specific
   reason not to.
5. Register the builder in `_HEAD_BUILDERS`.
6. Add tests for valid configs, invalid options, expected tensor shapes, and
   unknown-head errors.
7. Select it in YAML with:

```yaml
head:
  name: your_head_name
```

## Add a Trainer Mode

1. Add a trainer class in `evaluation.trainers` with a `run()` method.
2. Decide which parts are trainable. `ProbeTrainer` freezes the backbone and
   trains only the head; a finetuning trainer would usually train at least part
   of the backbone.
3. Reuse `validate_batch(batch)` before consuming task batches.
4. Define how the trainer handles `TargetSpec.kind`, loss functions,
   optimization, checkpointing, and metrics.
5. Add a `_build_<name>_trainer(mode_cfg, *, cfg, backbone, head, task)`
   function.
6. Register the builder in `_TRAINER_BUILDERS`.
7. Add tests for builder registration, unknown-mode errors, outputs, and the
   trainer's target-type behavior - optional, but coding agents are great at writing tests so this is very low effort.
8. Select it in YAML with:

```yaml
mode:
  name: your_mode_name
```

Keep trainer-specific options either under `mode` when they affect that mode
only, or under existing top-level blocks (`optimization`, `evaluation`,
`device`, `seed`) when they should be shared across modes.
