"""
Compute metrics from composite embeddings and attention matrices.
"""

# %% Imports
import pandas as pd
import torch
import torch.nn.functional as F


# %% Attention computation
# Raw attention dimensions: [Batch, Layer, Head, Token, Token]
def _aggregate_heads(attention: torch.Tensor, method: str):
    """Aggregate attention across heads."""
    if method == "mean":
        return attention.mean(dim=-3)
    if method == "max":
        return attention.amax(dim=-3)
    if method == "min":
        return attention.amin(dim=-3)


def _discard_attention(attention: torch.Tensor, ratio: float):
    """Discard the lowest attention values."""
    flat = attention.flatten(start_dim=2)
    n_discard = int(flat.shape[-1] * ratio)

    if n_discard:
        indices = flat.argsort(dim=-1)[..., :n_discard]
        flat.scatter_(-1, indices, 0)

    return flat.reshape_as(attention)


def _add_residual_and_normalize(attention: torch.Tensor):
    """Add residual connections and normalize attention rows."""
    n_tokens = attention.shape[-1]
    identity = torch.eye(
        n_tokens, device=attention.device, dtype=attention.dtype
    ).view(1, 1, n_tokens, n_tokens)

    attention = attention + identity
    return attention / attention.sum(dim=-1, keepdim=True)


def _attention_rollout(attentions: torch.Tensor, config: dict):
    """Compute attention rollout across transformer layers."""
    attention = _aggregate_heads(attentions, config["head_aggregation"])
    attention = _discard_attention(attention, config["discard_ratio"])
    attention = _add_residual_and_normalize(attention)

    rollout = attention[:, 0]
    for layer in range(1, attention.shape[1]):
        rollout = attention[:, layer] @ rollout

    return rollout


def _attention_layer(attentions: torch.Tensor, config: dict):
    """Extract attention from a single layer."""
    return _aggregate_heads(
        attentions[:, config["layer"]],
        config["head_aggregation"],
    )


def _summarize_attention(
    attention: torch.Tensor,
    constituent_patch_indices: list[list[int]],
    method: str,
) -> torch.Tensor:
    """Summarize CLS-to-patch attention for each constituent."""
    cls_to_patch_attention = attention[:, 0, 1:]
    constituent_scores = []

    for patch_indices in constituent_patch_indices:
        constituent_attention = cls_to_patch_attention[:, patch_indices]

        if method == "mean":
            score = constituent_attention.mean(dim=1)
        elif method == "max":
            score = constituent_attention.amax(dim=1)
        elif method == "min":
            score = constituent_attention.amin(dim=1)

        constituent_scores.append(score)

    return torch.stack(constituent_scores, dim=1)


# %% Embedding computation
def _embedding_similarity(
    composite_embeddings: torch.Tensor,
    constituent_embeddings_by_position: list[torch.Tensor],
) -> torch.Tensor:
    """Compute composite-constituent cosine similarities for each position index."""
    return torch.stack([
        F.cosine_similarity(composite_embeddings, constituent_embeddings, dim=-1)
        for constituent_embeddings in constituent_embeddings_by_position
    ], dim=1)


# %% Metric extraction
def _build_metric_df(trial_ids, scores, metric_name, variant_id):
    """Build long-form trial-level results."""
    n_constituents = scores.shape[1]

    return pd.DataFrame({
        "Trial_id": trial_ids.repeat(n_constituents).to_numpy(),
        "Position_index": list(range(n_constituents)) * len(trial_ids),
        "Metric": metric_name,
        "Variant_id": variant_id,
        "Value": scores.detach().cpu().numpy().reshape(-1),
    })


def compute_metrics(
    trial_ids: pd.Series,
    composite_outputs: tuple,
    metrics: dict,
    constituent_embeddings_by_variant: dict,
    constituent_patch_indices: list[list[int]],
) -> pd.DataFrame:
    """Compute all metric variants and return long-form results."""
    composite_embeddings, attentions = composite_outputs
    results = []

    for metric_name, variants in metrics.items():
        for variant_id, config in variants.items():

            if metric_name == "embedding_similarity":
                scores = _embedding_similarity(
                    composite_embeddings[config["embedding_function"]],
                    constituent_embeddings_by_variant[variant_id],
                )

            elif metric_name == "attention_rollout":
                attention = _attention_rollout(attentions, config)
                scores = _summarize_attention(
                    attention, constituent_patch_indices, config["region_summary"]
                )

            elif metric_name == "attention_layer":
                attention = _attention_layer(attentions, config)
                scores = _summarize_attention(
                    attention, constituent_patch_indices, config["region_summary"]
                )

            results.append(_build_metric_df(
                trial_ids, scores, metric_name, variant_id
            ))

    return pd.concat(results, ignore_index=True)


# %% Test
if __name__ == "__main__":
    # Test all three metrics using synthetic embeddings and attention matrices.
    trials = pd.Series([0, 1], name="Trial_id")
    composite_embeddings = {"visual": torch.tensor([[1., 0.], [0., 1.]])}
    constituent_embeddings = {
        "semi_visual": [
            torch.tensor([[1., 0.], [1., 0.]]),
            torch.tensor([[0., 1.], [0., 1.]]),
        ]
    }
    attentions = torch.rand(2, 2, 2, 5, 5)
    attentions = attentions / attentions.sum(dim=-1, keepdim=True)
    metrics = {
        "embedding_similarity": {
            "semi_visual": {"embedding_function": "visual"},
        },
        "attention_rollout": {
            "mean_000_mean": {
                "head_aggregation": "mean", "discard_ratio": 0.0,
                "region_summary": "mean",
            },
        },
        "attention_layer": {
            "last_mean_mean": {
                "layer": -1, "head_aggregation": "mean",
                "region_summary": "mean",
            },
        },
    }
    df = compute_metrics(
        trials, (composite_embeddings, attentions), metrics,
        constituent_embeddings, [[0, 1], [2, 3]],
    )
    print(df.to_string(index=False))
