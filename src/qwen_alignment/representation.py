"""Canonical h=5 continuation representation and standardisation."""

from __future__ import annotations

from typing import Any, Sequence

from .contracts import H_QWEN_TOKENS


HIDDEN_SIZE = 4_096
REPRESENTATION_LAYER = "hidden_states[-1]"
POOLING = "arithmetic mean over all 5 continuation-token positions"


class RepresentationError(RuntimeError):
    """Raised when a representation or standardisation invariant fails."""


def continuation_representations(
    model: Any,
    token_id_batch: Sequence[Sequence[int]] | Any,
    *,
    device: Any,
    torch: Any,
) -> Any:
    input_ids = torch.as_tensor(token_id_batch, dtype=torch.long, device=device)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.ndim != 2 or input_ids.shape[0] < 1:
        raise RepresentationError("token_id_batch must have shape (batch, 5)")
    if int(input_ids.shape[1]) != H_QWEN_TOKENS:
        raise RepresentationError(
            f"each alternative must contain exactly {H_QWEN_TOKENS} Qwen tokens"
        )
    vocabulary_size = int(model.config.vocab_size)
    if bool(((input_ids < 0) | (input_ids >= vocabulary_size)).any().item()):
        raise RepresentationError("token_id_batch contains an invalid token ID")

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    hidden_states = outputs.hidden_states
    expected_count = int(model.config.num_hidden_layers) + 1
    if hidden_states is None or len(hidden_states) != expected_count:
        raise RepresentationError(
            "model did not return the embedding state plus every transformer layer"
        )
    final_hidden = hidden_states[-1]
    expected_shape = (int(input_ids.shape[0]), H_QWEN_TOKENS, HIDDEN_SIZE)
    if tuple(final_hidden.shape) != expected_shape:
        raise RepresentationError(
            f"final hidden state has shape {tuple(final_hidden.shape)}, expected {expected_shape}"
        )
    # The checkpoint runs in BF16 to fit one 24 GB GPU.  Convert before pooling;
    # otherwise accumulation and all downstream geometry would remain BF16.
    representations = final_hidden.to(dtype=torch.float32).mean(dim=1)
    if tuple(representations.shape) != (int(input_ids.shape[0]), HIDDEN_SIZE):
        raise RepresentationError(
            f"mean pooling did not produce {HIDDEN_SIZE}-vectors"
        )
    if not bool(torch.isfinite(representations).all().item()):
        raise RepresentationError("representation contains a non-finite value")
    return representations


def standardise_representations(
    representations: Any,
    *,
    reference_mean: Any,
    reference_std: Any,
    torch: Any,
) -> Any:
    representations = representations.to(dtype=torch.float32)
    mean = torch.as_tensor(reference_mean, dtype=torch.float32, device=representations.device)
    std = torch.as_tensor(reference_std, dtype=torch.float32, device=representations.device)
    if tuple(mean.shape) != (HIDDEN_SIZE,) or tuple(std.shape) != (HIDDEN_SIZE,):
        raise RepresentationError(
            f"reference mean and std must each have shape ({HIDDEN_SIZE},)"
        )
    if not bool(torch.isfinite(mean).all().item()):
        raise RepresentationError("reference mean contains a non-finite value")
    if not bool(torch.isfinite(std).all().item()) or not bool((std > 0).all().item()):
        raise RepresentationError("reference std must be finite and strictly positive")
    standardised = (representations - mean) / std
    if not bool(torch.isfinite(standardised).all().item()):
        raise RepresentationError("standardised representation is non-finite")
    return standardised
