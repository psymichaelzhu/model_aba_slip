"""
Paths, constants, runtime settings and sweep resolution.
"""

# %% Imports
from copy import deepcopy
from itertools import product
from pathlib import Path

import yaml


# %% Paths
CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CONFIG_DIR.parent

STIMULI_DIR = PROJECT_ROOT / "stimuli"
CACHE_DIR = PROJECT_ROOT / "cache"
EXPERIMENT_RESULTS_DIR = PROJECT_ROOT / "experiment_results"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


# %% Image-related constants
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

# Image configurations according to SLIP repo
STANDARD_IMAGE_SIZE = (224, 224)
SLIP_NORM_MEAN = [0.485, 0.456, 0.406]
SLIP_NORM_STD = [0.229, 0.224, 0.225]


# %% Runtime settings (device-dependent; not part of the experiment definition)
INFERENCE_BATCH_SIZES = {"small": 32, "base": 16, "large": 8}
CHUNK_SIZE = 5000
MAX_ATTEMPTS = 3

DEBUG = False
def message(*args, **kwargs):
    """Print diagnostic messages only in debug mode."""
    if DEBUG:
        print(*args, **kwargs)


# %% Experiment configuration
with open(CONFIG_DIR / "experiment.yml", "r", encoding="utf-8") as f:
    EXPERIMENT_CONFIG = yaml.safe_load(f)

ACTIVE_SWEEP = EXPERIMENT_CONFIG["run"]["sweep"]

MODEL_COORDINATES = ("source", "training", "size", "dataset", "epochs", "seed")


# %% Model expansion
def _expand_model_set(set_id: str) -> list[tuple[str, dict]]:
    """Expand a model set into (model_id, model coordinates) pairs."""
    model_set = EXPERIMENT_CONFIG["model_sets"][set_id]
    keys = list(model_set)

    models = []
    for values in product(*(model_set[key] for key in keys)):
        model = {"dataset": "none", "epochs": 0, "seed": None, **dict(zip(keys, values))}
        model = {key: model[key] for key in MODEL_COORDINATES}

        # fb_slip naming: model IDs match the released checkpoint filenames.
        if model["training"] == "untrained":
            model_id = f"untrained_{model['size']}_seed{model['seed']}"
        elif model["dataset"] == "yfcc15m":
            model_id = f"{model['training']}_{model['size']}_{model['epochs']}ep"
        else:
            model_id = (
                f"{model['training']}_{model['size']}_"
                f"{model['dataset']}_{model['epochs']}ep"
            )

        models.append((model_id, model))

    return models


# %% Sweep resolution
def resolve_sweep(sweep_id: str) -> list[dict]:
    """Expand a sweep into its experiments (layout -> model -> imageset -> subset)."""
    sweep = EXPERIMENT_CONFIG["sweeps"][sweep_id]

    models = {}
    for set_id in sweep["models"]:
        for model_id, model in _expand_model_set(set_id):
            models[(model["source"], model_id)] = model

    experiments = []
    for layout_id in sweep["layouts"]:
        for (source, model_id), model in models.items():
            for imageset, subsets in sweep["stimuli"].items():
                for subset in subsets:
                    experiments.append({
                        "sweep": sweep_id,
                        "imageset": imageset,
                        "subset": subset,

                        "trial_design_id": sweep["trial_design"],
                        "trial_design": deepcopy(
                            EXPERIMENT_CONFIG["trial_designs"][sweep["trial_design"]]
                        ),

                        "layout_id": layout_id,
                        "layout": deepcopy(EXPERIMENT_CONFIG["layouts"][layout_id]),

                        "model_id": model_id,
                        "model": dict(model),

                        "metrics_id": sweep["metrics"],
                        "metrics": deepcopy(EXPERIMENT_CONFIG["metrics"][sweep["metrics"]]),

                        "result_dir": (
                            EXPERIMENT_RESULTS_DIR / sweep_id / layout_id
                            / imageset / subset / source / model_id
                        ),
                    })

    return experiments
