"""
Run a sweep, retrying failed experiments in fresh processes.

Usage: python -m script.experiment._runner [--sweep SWEEP_ID]
"""

# %% Imports
import argparse
import multiprocessing as mp
import os

from tqdm import tqdm

from config import ACTIVE_SWEEP, MAX_ATTEMPTS, message, resolve_sweep
from script.experiment import record


# %% Experiment execution
def _label(experiment: dict) -> str:
    return "/".join([
        experiment["layout_id"], experiment["imageset"],
        experiment["subset"], experiment["model_id"],
    ])


def _ignore_warning():
    import warnings

    warnings.filterwarnings(
        "ignore",
        message=r"pkg_resources is deprecated as an API.*",
        category=UserWarning,
    )


def _run_experiment(experiment: dict, progress_lock):
    """Run an experiment in an isolated worker process."""
    tqdm.set_lock(progress_lock)
    _ignore_warning()

    message(f"Worker PID: {os.getpid()}", flush=True)

    from script.experiment.experiment import run_experiment
    run_experiment(experiment)


def _run_with_retries(experiment: dict, context, progress_lock):
    """Run an experiment, retrying in a fresh process on failure."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        process = context.Process(
            target=_run_experiment,
            args=(experiment, progress_lock),
        )
        process.start()
        process.join()

        if process.exitcode == 0:
            minutes = record.read_progress(experiment)["duration_minutes"]
            tqdm.write(f"Completed: {_label(experiment)} | {minutes:.1f} min")
            return

        if attempt == MAX_ATTEMPTS:
            raise RuntimeError(
                f"Experiment failed after {MAX_ATTEMPTS} attempts: "
                f"{_label(experiment)} (exit code: {process.exitcode})"
            )

        tqdm.write(
            f"Retrying {_label(experiment)} "
            f"({attempt + 1}/{MAX_ATTEMPTS}) "
            f"after worker exit code {process.exitcode}"
        )


# %% Main
def main():
    """Run all unfinished experiments of a sweep."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", default=ACTIVE_SWEEP)
    sweep_id = parser.parse_args().sweep

    experiments = resolve_sweep(sweep_id)
    record.write_sweep_snapshot(sweep_id, experiments)
    pending = [
        experiment for experiment in experiments
        if record.read_progress(experiment)["status"] != "completed"
    ]

    context = mp.get_context("spawn")
    progress_lock = context.RLock()
    tqdm.set_lock(progress_lock)

    with tqdm(
        total=len(experiments),
        initial=len(experiments) - len(pending),
        desc=sweep_id,
        unit="experiment",
        position=0,
    ) as bar:
        for experiment in pending:
            _run_with_retries(experiment, context, progress_lock)
            record.write_sweep_snapshot(sweep_id, experiments)
            bar.update()


if __name__ == "__main__":
    main()
