from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class StructuredOutput:
    """Validates and parses raw LLM output into typed Pydantic models.

    Wraps the common pattern of asking an LLM for JSON and parsing it
    into a validated data structure.
    """

    def __init__(self, model_class: type[T]) -> None:
        self._model_class = model_class

    @property
    def model_class(self) -> type[T]:
        return self._model_class

    def parse(self, raw: str | dict[str, Any]) -> T:
        """Parse raw LLM output (JSON string or dict) into the target model.

        Raises:
            ValidationError: If the data does not match the model schema.
            json.JSONDecodeError: If raw is a string that is not valid JSON.
        """
        data = json.loads(raw) if isinstance(raw, str) else raw
        return self._model_class.model_validate(data)

    def parse_safe(self, raw: str | dict[str, Any]) -> T | None:
        """Parse without raising -- returns None on failure."""
        try:
            return self.parse(raw)
        except (ValidationError, json.JSONDecodeError, TypeError):
            return None

    def json_schema(self) -> dict[str, Any]:
        """Return the JSON schema for the target model.

        Useful for passing as response_format hints to the LLM.
        """
        return self._model_class.model_json_schema()
