"""
Read and write experiment records: snapshots, chunks, run info and sweep snapshots.
"""

# %% Imports
from pathlib import Path

import pandas as pd
import yaml

from config import CHUNK_SIZE, EXPERIMENT_RESULTS_DIR, PROJECT_ROOT


# %% Experiment snapshot
def _snapshot_path(experiment: dict) -> Path:
    return Path(experiment["result_dir"]) / "experiment_snapshot.yml"


def _read_snapshot(experiment: dict) -> dict:
    path = _snapshot_path(experiment)
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_snapshot(experiment: dict, snapshot: dict):
    path = _snapshot_path(experiment)
    temp_path = path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(snapshot, f, sort_keys=False, allow_unicode=True)
    temp_path.replace(path)


def prepare_experiment(experiment: dict):
    """Create the experiment directory, then write or verify its snapshot."""
    (Path(experiment["result_dir"]) / "chunks").mkdir(parents=True, exist_ok=True)

    definition = {k: v for k, v in experiment.items() if k != "result_dir"}
    snapshot = _read_snapshot(experiment)

    if not snapshot:
        _write_snapshot(experiment, definition)
        return

    snapshot.pop("run_info", None)
    if snapshot != definition:
        raise ValueError(f"Experiment definition changed: {_snapshot_path(experiment)}")


def _update_run_info(experiment: dict, run_info: dict):
    snapshot = _read_snapshot(experiment)
    snapshot["run_info"] = run_info
    _write_snapshot(experiment, snapshot)


# %% Chunks
def _chunk_path(experiment: dict, start: int, end: int) -> Path:
    fmt = lambda i: f"{i // 1000}k" if i >= 1000 and i % 1000 == 0 else str(i)
    return Path(experiment["result_dir"]) / "chunks" / f"chunk_{fmt(start)}_{fmt(end)}.csv"


def _chunk_ranges(experiment: dict) -> list[tuple[int, int]]:
    n_trials = experiment["trial_design"]["n_trials"]
    return [
        (start, min(start + CHUNK_SIZE, n_trials))
        for start in range(0, n_trials, CHUNK_SIZE)
    ]


def pending_chunks(experiment: dict) -> list[tuple[int, int]]:
    """Trial intervals whose chunks have not been saved yet."""
    return [
        (start, end) for start, end in _chunk_ranges(experiment)
        if not _chunk_path(experiment, start, end).exists()
    ]


def save_chunk(experiment: dict, start: int, end: int, df: pd.DataFrame, run_info: dict):
    """Save a chunk atomically and update the run info."""
    path = _chunk_path(experiment, start, end)
    temp_path = path.with_suffix(".tmp")
    df.to_csv(temp_path, index=False)
    temp_path.replace(path)
    _update_run_info(experiment, run_info)


def finalize_experiment(experiment: dict, run_info: dict):
    """Merge all chunks into trial_metrics.csv and write the final run info."""
    result_path = Path(experiment["result_dir"]) / "trial_metrics.csv"
    temp_path = result_path.with_suffix(".tmp")

    with open(temp_path, "w", encoding="utf-8") as f:
        for i, (start, end) in enumerate(_chunk_ranges(experiment)):
            pd.read_csv(_chunk_path(experiment, start, end)).to_csv(
                f, index=False, header=(i == 0)
            )
    temp_path.replace(result_path)
    _update_run_info(experiment, run_info)


# %% Progress
def read_progress(experiment: dict) -> dict:
    """Return status, completed trials and accumulated duration (minutes)."""
    completed_trials = sum(
        end - start for start, end in _chunk_ranges(experiment)
        if _chunk_path(experiment, start, end).exists()
    )

    if (Path(experiment["result_dir"]) / "trial_metrics.csv").exists():
        status = "completed"
    elif completed_trials:
        status = "partial"
    else:
        status = "pending"

    run_info = _read_snapshot(experiment).get("run_info", {})
    return {
        "status": status,
        "completed_trials": completed_trials,
        "duration_minutes": run_info.get("duration_minutes", 0),
    }


def write_sweep_snapshot(sweep_id: str, experiments: list[dict]):
    """Write one row per experiment with its keys, status and duration."""
    rows = []
    for experiment in experiments:
        progress = read_progress(experiment)
        rows.append({
            "sweep": experiment["sweep"],
            "layout": experiment["layout_id"],
            "imageset": experiment["imageset"],
            "subset": experiment["subset"],
            "source": experiment["model"]["source"],
            "model": experiment["model_id"],
            "status": progress["status"],
            "duration_minutes": progress["duration_minutes"],
            "result_dir": Path(experiment["result_dir"]).relative_to(PROJECT_ROOT).as_posix(),
        })

    path = EXPERIMENT_RESULTS_DIR / sweep_id / "sweep_snapshot.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
