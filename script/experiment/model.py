"""
Load models from their coordinates and collect forward outputs.
"""

# %% Imports
import sys

import torch

from config import CHECKPOINT_DIR, PROJECT_ROOT
from script.experiment.stimulus import normalize_images

sys.path.insert(0, str(PROJECT_ROOT / "external" / "SLIP"))
import models


# %% Model loading
def get_device() -> torch.device:
    """Select the available computation device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_fb_slip(model_id: str, model: dict):
    """Build a model from the official SLIP repo: checkpoint or seeded random init."""
    arch = {"small": "S", "base": "B", "large": "L"}[model["size"]]

    if model["training"] == "untrained":
        # SIMCLR wraps the same timm ViT as CLIP/SLIP, without the unused text tower.
        # fork_rng restores the caller's CPU RNG state afterwards.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(model["seed"])
            return getattr(models, f"SIMCLR_VIT{arch}16")(
                rand_embed=False, ssl_mlp_dim=4096, ssl_emb_dim=256,
            )

    checkpoint = torch.load(
        CHECKPOINT_DIR / model["source"] / f"{model_id}.pt",
        map_location="cpu",
    )
    args = checkpoint["args"]

    constructor = f"{model['training'].upper()}_VIT{arch}16"
    if args.model != constructor:
        raise ValueError(f"Checkpoint mismatch for {model_id}: model={args.model}")

    net = getattr(models, constructor)(
        rand_embed=False,
        ssl_mlp_dim=args.ssl_mlp_dim,
        ssl_emb_dim=args.ssl_emb_dim,
    )
    state_dict = {
        k.replace("module.", "", 1): v
        for k, v in checkpoint["state_dict"].items()
    }
    net.load_state_dict(state_dict, strict=True)
    return net


_LOADERS = {"fb_slip": _load_fb_slip}


def load_model(model_id: str, model: dict, device: torch.device):
    """Build a model according to its source and move it to the device."""
    return _LOADERS[model["source"]](model_id, model).to(device).eval()


# %% Attention setup
def _register_attention_hooks(model):
    """Capture attention matrices from ViT layers."""
    cache = []

    def hook(module, inputs, output):
        cache.append(output.detach())

    handles = [
        block.attn.attn_drop.register_forward_hook(hook)
        for block in model.visual.blocks
    ]
    return cache, handles


def _get_patch_indices(layout: dict, patch_size: int):
    """Convert constituent regions to patch indices."""
    grid_width = layout["canvas_size"][0] // patch_size
    item_w, item_h = layout["item_size"]

    return [
        [
            row * grid_width + col
            for row in range(y // patch_size, (y + item_h) // patch_size)
            for col in range(x // patch_size, (x + item_w) // patch_size)
        ]
        for x, y in layout["positions"]
    ]


def setup_attention(model, metrics: dict, layout: dict):
    """Register attention hooks and locate constituent patches."""
    if not any(name.startswith("attention_") for name in metrics):
        return [], [], []

    attention_cache, handles = _register_attention_hooks(model)
    patch_size = model.visual.patch_embed.patch_size[0]
    constituent_patch_indices = _get_patch_indices(layout, patch_size)
    return attention_cache, handles, constituent_patch_indices


def remove_hooks(handles: list):
    """Remove registered attention hooks."""
    for handle in handles:
        handle.remove()


# %% Model forward
def forward_composites(
    model,
    composites: torch.Tensor,
    metrics: dict,
    attention_cache: list,
):
    """Forward composites once and collect raw embeddings and attention."""
    embedding_functions = {
        config["embedding_function"]
        for config in metrics.get("embedding_similarity", {}).values()
    }

    with torch.no_grad():
        visual_embeddings = model.visual(normalize_images(composites))
        embeddings = {}

        if "visual" in embedding_functions:
            embeddings["visual"] = visual_embeddings

        if "encode_image" in embedding_functions:
            embeddings["encode_image"] = (
                visual_embeddings @ model.image_projection
                if hasattr(model, "image_projection")
                else visual_embeddings  # SimCLR has no image_projection
            )

    attentions = (
        torch.stack(attention_cache, dim=1)
        if attention_cache
        else None  # No attention metrics requested; hooks not registered
    )
    attention_cache.clear()

    return embeddings, attentions


# %% Test
if __name__ == "__main__":
    # Test untrained model forward and attention extraction with synthetic inputs.
    model = load_model(
        "untrained_small_seed0",
        {"source": "fb_slip", "training": "untrained", "size": "small",
         "dataset": "none", "epochs": 0, "seed": 0},
        torch.device("cpu"),
    )

    layout = {
        "canvas_size": [224, 224],
        "item_size": [96, 96],
        "positions": [[0, 64], [128, 64]],
    }
    metrics = {
        "embedding_similarity": {
            "semi_visual": {"embedding_function": "visual"},
        },
        "attention_layer": {"last_mean_mean": {}},
    }
    attention_cache, handles, patch_indices = setup_attention(model, metrics, layout)
    outputs = forward_composites(
        model, torch.rand(1, 3, 224, 224), metrics, attention_cache
    )
    remove_hooks(handles)

    print("Visual embedding:", outputs[0]["visual"].shape)
    print("Attention:", outputs[1].shape)
    print("Constituent patches:", [len(p) for p in patch_indices])
