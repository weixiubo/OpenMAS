"""Safe JSON sequence coercion."""

from typing import Any


def sequence_value(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []
