"""
Construct stimuli and cache constituent embeddings.
"""

# %% Imports
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from config import CACHE_DIR, PROJECT_ROOT, SLIP_NORM_MEAN, SLIP_NORM_STD, STIMULI_DIR, message


# %% Imageset
def get_image_ids(imageset: str, subset: str) -> list[str]:
    """Get image IDs belonging to a subset."""
    subsets = pd.read_csv(STIMULI_DIR / imageset / "subsets.csv", dtype={"Image_id": str})

    if subset == "all":
        image_ids = subsets["Image_id"]
    else:
        image_ids = subsets.loc[subsets[subset] == 1, "Image_id"]

    return sorted(image_ids.tolist())


def load_image_cache(imageset: str, subset: str) -> dict[str, torch.Tensor]:
    """Load all images in an imageset subset."""
    image_dir = STIMULI_DIR / imageset / "images"

    image_cache = {}
    for image_id in get_image_ids(imageset, subset):
        with Image.open(image_dir / f"{image_id}.jpg") as image:
            image_cache[image_id] = pil_to_tensor(image.convert("RGB"))

    return image_cache


# %% Stimulus construction
def _place_item(
    canvas: torch.Tensor,
    images: torch.Tensor,
    layout: dict,
    position_index: int,
) -> torch.Tensor:
    """Place a batch of images on canvases."""
    w, h = layout["item_size"]
    x, y = layout["positions"][position_index]

    items = F.interpolate(
        images,
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    )

    canvas[:, :, y:y+h, x:x+w] = items
    return canvas


def _make_canvas(
    batch_size: int,
    layout: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create a batch of blank canvases."""
    w, h = layout["canvas_size"]
    background = torch.tensor(
        layout["background"],
        dtype=dtype,
        device=device,
    ) / 255

    return background.view(1, 3, 1, 1).expand(
        batch_size, 3, h, w
    ).clone()


def _make_constituent(
    images: torch.Tensor,
    layout: dict,
    present: str,
    position_index: int,
) -> torch.Tensor:
    """Construct constituent images in isolate or semi presentation."""
    if present == "isolate":
        return images

    canvas = _make_canvas(
        len(images), layout, images.device, images.dtype
    )
    return _place_item(canvas, images, layout, position_index)


def make_composite(
    images: list[torch.Tensor],
    layout: dict,
) -> torch.Tensor:
    """Construct a batch of composite images."""
    canvas = _make_canvas(
        len(images[0]), layout, images[0].device, images[0].dtype
    )

    for position_index, image in enumerate(images):
        _place_item(canvas, image, layout, position_index)

    return canvas


def normalize_images(images: torch.Tensor) -> torch.Tensor:
    """Normalize images using SLIP training statistics."""
    mean = images.new_tensor(SLIP_NORM_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(SLIP_NORM_STD).view(1, 3, 1, 1)
    return (images - mean) / std


# %% Extraction
def _extract_constituent_embeddings(
    imageset: str,
    layout: dict,
    embedding_config: dict,
    model,
    variant_id: str,
    model_id: str,
    layout_id: str,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """Extract embeddings for all images and positions."""
    device = next(model.parameters()).device

    image_ids = get_image_ids(imageset, "all")
    image_dir = STIMULI_DIR / imageset / "images"

    present = embedding_config["present"]
    encoder = getattr(model, embedding_config["embedding_function"])

    n_constituents = len(layout["positions"])
    position_indices = range(n_constituents) if present == "semi" else range(1)

    embeddings = {}

    with torch.no_grad(), tqdm(
        total=len(image_ids),
        desc=f"Embedding: {imageset} / {model_id} / {layout_id} / {variant_id}",
        unit="image",
        position=1,
        leave=False,
    ) as progress:
        for start in range(0, len(image_ids), batch_size):
            batch_ids = image_ids[start:start + batch_size]

            images = torch.stack([
                pil_to_tensor(
                    Image.open(image_dir / f"{image_id}.jpg").convert("RGB")
                ).float() / 255
                for image_id in batch_ids
            ]).to(device)

            for position_index in position_indices:
                batch = _make_constituent(images, layout, present, position_index)
                features = encoder(normalize_images(batch)).cpu()

                for image_id, embedding in zip(batch_ids, features):
                    if present == "isolate":
                        embeddings[image_id] = embedding.unsqueeze(0)
                    else:
                        if position_index == 0:
                            embeddings[image_id] = torch.empty(
                                n_constituents, embedding.numel(),
                                dtype=embedding.dtype,
                            )
                        embeddings[image_id][position_index] = embedding

            progress.update(len(batch_ids))

    return embeddings


# %% Cache
def ensure_constituent_embeddings(
    experiment: dict,
    model,
    batch_size: int,
    overwrite: bool = False,
) -> dict:
    """Load or generate constituent embeddings for all embedding variants."""
    imageset = experiment["imageset"]
    layout = experiment["layout"]
    device = next(model.parameters()).device

    model_dir = (
        CACHE_DIR / "constituent_embeddings" / imageset
        / experiment["model"]["source"] / experiment["model_id"]
    )

    embeddings_by_variant = {}
    for variant_id, config in experiment["metrics"].get("embedding_similarity", {}).items():
        layout_dir = "isolate" if config["present"] == "isolate" else experiment["layout_id"]
        path = model_dir / layout_dir / f"{config['embedding_function']}.pt"

        if path.exists() and not overwrite:
            message(f"Loading constituent embeddings: {path}")
            embeddings = torch.load(path, map_location="cpu")
        else:
            embeddings = _extract_constituent_embeddings(
                imageset, layout, config, model,
                variant_id=variant_id,
                model_id=experiment["model_id"],
                layout_id=layout_dir,
                batch_size=batch_size,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(embeddings, path)
            message(f"Generated constituent embeddings: {path}")

        if config["present"] == "isolate":
            n_constituents = len(layout["positions"])
            embeddings = {
                k: v.repeat(n_constituents, 1)
                for k, v in embeddings.items()
            }

        embeddings_by_variant[variant_id] = {
            k: v.to(device) for k, v in embeddings.items()
        }

    return embeddings_by_variant


# %% Test
if __name__ == "__main__":
    # Test constituent and composite construction using synthetic color images.
    output_dir = PROJECT_ROOT / "test" / "stimulus"
    output_dir.mkdir(parents=True, exist_ok=True)

    colors = torch.tensor([
        [1, 0, 0],  # Red
        [0, 1, 0],  # Green
        [0, 0, 1],  # Blue
        [1, 1, 0],  # Yellow
    ], dtype=torch.float32)

    images = colors.view(4, 3, 1, 1).expand(4, 3, 224, 224)

    layout = {
        "canvas_size": [224, 224],
        "item_size": [96, 96],
        "positions": [[0, 64], [128, 64]],
        "background": [128, 128, 128],
    }

    grid_layout = {
        **layout,
        "positions": [[0, 0], [128, 0], [0, 128], [128, 128]],
    }

    for present in ("semi", "isolate"):
        position_indices = range(len(layout["positions"])) if present == "semi" else range(1)

        for position_index in position_indices:
            batch = _make_constituent(images, layout, present, position_index)
            to_pil_image(batch[0]).save(
                output_dir / f"{present}_{position_index}.png"
            )

    composite = make_composite(
        [images[0:1], images[1:2]], layout
    )
    to_pil_image(composite[0]).save(output_dir / "composite_pair.png")

    grid = make_composite(
        [images[i:i+1] for i in range(4)], grid_layout
    )
    to_pil_image(grid[0]).save(output_dir / "composite_grid.png")

    print(f"Test images saved to {output_dir}")
