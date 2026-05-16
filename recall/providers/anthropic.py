from __future__ import annotations

from typing import Any


def normalize_request(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in kwargs.items()}


def normalize_response(response: Any) -> dict[str, Any]:
    try:
        return response.model_dump()
    except AttributeError:
        return {"raw": str(response)}


def reconstruct_message(payload: dict[str, Any]) -> Any:
    try:
        from anthropic.types import Message
        return Message.model_validate(payload)
    except ImportError as e:
        raise ImportError("anthropic package required for replay: pip install anthropic") from e


# Approximate pricing per million tokens as of mid-2025
_COST_TABLE: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (0.80, 4.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-7": (15.00, 75.00),
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-opus": (15.00, 75.00),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    for prefix, (in_rate, out_rate) in _COST_TABLE.items():
        if model.startswith(prefix):
            return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000
    return None
