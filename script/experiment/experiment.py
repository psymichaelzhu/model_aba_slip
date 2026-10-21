"""
Run a single experiment and save its trial-level metrics in chunks.
"""

# %% Imports
from datetime import datetime
from time import perf_counter

import pandas as pd
import timm
import torch
from tqdm import tqdm

from config import INFERENCE_BATCH_SIZES, message
from script.experiment import record
from script.experiment.design_matrix import ensure_design_matrix
from script.experiment.metrics import compute_metrics
from script.experiment.model import (
    forward_composites,
    get_device,
    load_model,
    remove_hooks,
    setup_attention,
)
from script.experiment.stimulus import (
    ensure_constituent_embeddings,
    load_image_cache,
    make_composite,
)


# %% Batch processing
# Process trials in batches to limit memory usage.
def _prepare_batch_inputs(
    batch_design_matrix: pd.DataFrame,
    image_cache: dict,
    constituent_embedding_cache: dict,
    layout: dict,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
    """Prepare composite images and constituent embeddings for one batch."""
    constituent_columns = [
        col for col in batch_design_matrix
        if col.startswith("Constituent_")
    ]

    constituent_images = []
    embeddings_by_variant = {
        config_key: []
        for config_key in constituent_embedding_cache
    }

    for position_index, column in enumerate(constituent_columns):
        image_ids = batch_design_matrix[column]

        images = torch.stack([
            image_cache[image_id]
            for image_id in image_ids
        ]).to(device=device, dtype=torch.float32) / 255

        constituent_images.append(images)

        for config_key, cached_embeddings in constituent_embedding_cache.items():
            embeddings = torch.stack([
                cached_embeddings[image_id][position_index]
                for image_id in image_ids
            ])
            embeddings_by_variant[config_key].append(embeddings)

    composites = make_composite(constituent_images, layout)
    return composites, embeddings_by_variant


def _process_batch(
    batch_design_matrix: pd.DataFrame,
    experiment: dict,
    model,
    image_cache: dict,
    constituent_embedding_cache: dict,
    attention_cache: list,
    constituent_patch_indices: list[list[int]],
) -> pd.DataFrame:
    """Process one batch and return long-form metric results."""
    device = next(model.parameters()).device

    composites, constituent_embeddings_by_variant = _prepare_batch_inputs(
        batch_design_matrix,
        image_cache,
        constituent_embedding_cache,
        experiment["layout"],
        device,
    )

    composite_outputs = forward_composites(
        model, composites, experiment["metrics"], attention_cache
    )

    return compute_metrics(
        batch_design_matrix["Trial_id"],
        composite_outputs,
        experiment["metrics"],
        constituent_embeddings_by_variant,
        constituent_patch_indices,
    )


# %% Experiment execution
def run_experiment(experiment: dict):
    """Run one experiment, resuming from saved chunks if any."""
    record.prepare_experiment(experiment)
    progress = record.read_progress(experiment)
    if progress["status"] == "completed":
        message(f"Already completed: {experiment['result_dir']}")
        return

    session_start = perf_counter()
    device = get_device()
    run_info = {
        "torch": str(torch.__version__),
        "timm": str(timm.__version__),
        "device": (
            f"cuda: {torch.cuda.get_device_name(device)}"
            if device.type == "cuda" else device.type
        ),
    }

    def current_run_info(**extra) -> dict:
        minutes = progress["duration_minutes"] + (perf_counter() - session_start) / 60
        return {**run_info, **extra, "duration_minutes": round(minutes, 2)}

    design_matrix = ensure_design_matrix(experiment)
    model = load_model(experiment["model_id"], experiment["model"], device)
    batch_size = INFERENCE_BATCH_SIZES[experiment["model"]["size"]]
    constituent_embeddings = ensure_constituent_embeddings(experiment, model, batch_size)
    image_cache = load_image_cache(experiment["imageset"], experiment["subset"])
    attention_cache, handles, constituent_patch_indices = setup_attention(
        model, experiment["metrics"], experiment["layout"]
    )

    message(f"Experiment: {experiment['result_dir']}")
    message(f"Device: {device} | Batch size: {batch_size}")

    try:
        with tqdm(
            total=len(design_matrix),
            initial=progress["completed_trials"],
            desc=" / ".join([
                experiment["layout_id"], experiment["imageset"],
                experiment["subset"], experiment["model_id"],
            ]),
            unit="trial",
            position=1,
            leave=False,
        ) as bar:
            for chunk_start, chunk_end in record.pending_chunks(experiment):
                chunk_results = []
                for start in range(chunk_start, chunk_end, batch_size):
                    batch_design_matrix = design_matrix.iloc[
                        start:min(start + batch_size, chunk_end)
                    ]
                    chunk_results.append(_process_batch(
                        batch_design_matrix,
                        experiment,
                        model,
                        image_cache,
                        constituent_embeddings,
                        attention_cache,
                        constituent_patch_indices,
                    ))
                    bar.update(len(batch_design_matrix))

                record.save_chunk(
                    experiment, chunk_start, chunk_end,
                    pd.concat(chunk_results, ignore_index=True),
                    current_run_info(),
                )
    finally:
        remove_hooks(handles)

    record.finalize_experiment(
        experiment,
        current_run_info(finished_at=datetime.now().isoformat(timespec="seconds")),
    )
