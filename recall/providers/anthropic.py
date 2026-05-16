from __future__ import annotations

from typing import Any


def normalize_request(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in kwargs.items()}


def normalize_response(response: Any) -> dict[str, Any]:
    try:
        return response.model_dump()
    except AttributeError:
        return {"raw": str(response)}
