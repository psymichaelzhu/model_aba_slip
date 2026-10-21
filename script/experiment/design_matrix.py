"""
Generate and cache design matrices for experiments.
"""

# %% Imports
import random
from math import perm

import pandas as pd

from config import CACHE_DIR, PROJECT_ROOT, message
from script.experiment.stimulus import get_image_ids


# %% Generation
def _generate_design_matrix(
    image_ids: list[str],
    n_trials: int,
    n_items_per_trial: int,
    sampling_seed: int,
) -> pd.DataFrame:
    """Generate unique ordered samples in random order."""
    max_trials = perm(len(image_ids), n_items_per_trial)

    if n_trials > max_trials:
        raise ValueError(
            f"Requested {n_trials} trials, but only {max_trials} unique trials possible."
        )

    rng = random.Random(sampling_seed)
    trials = []
    seen = set()

    while len(trials) < n_trials:
        trial = tuple(rng.sample(image_ids, n_items_per_trial))
        if trial not in seen:
            seen.add(trial)
            trials.append(trial)

    columns = [f"Constituent_{i}" for i in range(n_items_per_trial)]
    df = pd.DataFrame(trials, columns=columns)
    df.insert(0, "Trial_id", range(len(df)))
    return df


# %% Cache
def ensure_design_matrix(
    experiment: dict,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Load or generate the design matrix for an experiment."""
    imageset = experiment["imageset"]
    subset = experiment["subset"]

    path = (
        CACHE_DIR / "design_matrix" / experiment["trial_design_id"]
        / imageset / f"{subset}.csv"
    )

    if path.exists() and not overwrite:
        message(f"Loading design matrix: {path}")
        return pd.read_csv(path, dtype=str).astype({"Trial_id": int})

    df = _generate_design_matrix(
        get_image_ids(imageset, subset), **experiment["trial_design"]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    message(f"Generated design matrix: {path}")
    return df


# %% Test
if __name__ == "__main__":
    # Test design matrix generation with synthetic image IDs.
    image_ids = ["Apple", "Avocado", "Peach", "Banana"]

    df = _generate_design_matrix(
        image_ids,
        n_trials=6,
        n_items_per_trial=2,
        sampling_seed=42,
    )

    output_dir = PROJECT_ROOT / "test" / "design_matrix"
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / "design_matrix.csv"
    df.to_csv(path, index=False)

    print(f"Shape: {df.shape}")
    print(df.to_string(index=False))
    print(f"Test design matrix saved to {path}")
