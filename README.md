# ABA_VLM

Do vision models prioritize emotionally salient images when two images compete
for representation? Each trial places two images from an affective imageset on
one canvas, passes the composite through a ViT, and measures how much each image
is represented in the output (embedding similarity) or attended to (attention).

## Setup

This project uses [facebook SLIP](https://github.com/facebookresearch/SLIP) as a Git submodule.

```bash
git clone --recurse-submodules https://github.com/psymichaelzhu/ABA_VLM.git
cd ABA_VLM

conda env create -f environment.yml
conda activate slip_aba
```

Pretrained SLIP checkpoints go in `checkpoints/fb_slip/`, named as released
(e.g. `https://dl.fbaipublicfiles.com/slip/clip_base_25ep.pt` →
`checkpoints/fb_slip/clip_base_25ep.pt`).

Stimuli are prepared with the preprocessing scripts (see `script/preprocessing/README.md`).

## Pipeline

```
stimuli ──► cache ──► experiment_results ──► analysis_results
```

| Directory | Contents | Regenerable |
|---|---|---|
| `stimuli/{imageset}/` | Raw and standardized images, ratings, subsets, visual features | No (raw data) |
| `cache/` | Design matrices and constituent embeddings | Yes |
| `experiment_results/` | Trial-level metrics (GPU-expensive) | Costly |
| `analysis_results/` | Statistics and figures | Yes |

## Configuration: `config/experiment.yml`

An experiment is defined by five factors: **stimuli** (imageset + subset),
**trial design**, **layout**, **model** and **metrics**.

- `layouts`, `trial_designs`, `metrics`: named presets.
- `model_sets`: named grids over model coordinates
  (`source, training, size, dataset, epochs, seed`).
  Each set expands to the Cartesian product of its lists.
- `sweeps`: one trial design and one metric preset, applied to lists of
  layouts, model sets and stimuli (full factorial).
- `run.sweep`: the sweep to run by default.

A name, once used to produce results, should keep its definition; use a new
name for a new definition.

### Models

Model IDs are generated from coordinates:

| Models | ID |
|---|---|
| `fb_slip`, trained on yfcc15m | `{training}_{size}_{epochs}ep` (e.g. `clip_base_25ep`) |
| `fb_slip`, other datasets | `{training}_{size}_{dataset}_{epochs}ep` (e.g. `clip_base_cc3m_40ep`) |
| `fb_slip`, untrained | `untrained_{size}_seed{seed}` |

Pretrained models load `checkpoints/{source}/{model_id}.pt`; the checkpoint's
constructor is checked against the coordinates. Their `seed` is `null`
(not recorded in the released checkpoints). Untrained models are
built with the SLIP ViT constructor and seeded random initialization.

Sets may overlap: models are deduplicated by `(source, model_id)` within a sweep.

### Runtime settings: `config/__init__.py`

`INFERENCE_BATCH_SIZES` (per model size), `CHUNK_SIZE` and `MAX_ATTEMPTS` depend on
the machine and do not affect results.

## Run

```bash
python -m script.experiment._runner                       # run.sweep in experiment.yml
python -m script.experiment._runner --sweep test_pair_1k  # a specific sweep
```

Runs are resumable: finished experiments are skipped and partial experiments
continue from their last saved chunk. Failed experiments are retried in a fresh
process.

## Outputs

```
cache/
  design_matrix/{trial_design}/{imageset}/{subset}.csv
  constituent_embeddings/{imageset}/{source}/{model_id}/{layout | isolate}/{function}.pt

experiment_results/{sweep}/
  sweep_snapshot.csv
  {layout}/{imageset}/{subset}/{source}/{model_id}/
    experiment_snapshot.yml
    chunks/
    trial_metrics.csv
```

- `sweep_snapshot.csv`: one row per experiment of the sweep with its keys,
  status (`pending` / `partial` / `completed`) and accumulated duration.
  Rewritten on every run.
- `experiment_snapshot.yml`: the full definition of the experiment, plus
  `run_info` (torch, timm, device, finish time, accumulated duration).
  On resume, the definition must match the current config.
- `trial_metrics.csv`: one row per trial × position × metric variant
  (`Trial_id, Position_index, Metric, Variant_id, Value`).

## Code

| Script | Role |
|---|---|
| `config/__init__.py` | Paths, constants, runtime settings, sweep resolution |
| `script/experiment/_runner.py` | Runs a sweep with retries and progress |
| `script/experiment/experiment.py` | Computation for one experiment |
| `script/experiment/record.py` | All experiment file I/O: snapshots, chunks, run info |
| `script/experiment/model.py` | Model loading, attention hooks, forward pass |
| `script/experiment/stimulus.py` | Composite construction, constituent embedding cache |
| `script/experiment/design_matrix.py` | Trial sampling and cache |
| `script/experiment/metrics.py` | Embedding similarity and attention metrics |
| `script/preprocessing/` | Standardizes images, ratings, subsets and visual features |
