from __future__ import annotations

import inspect
from typing import Any, Callable, Sequence


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        self._tools[name] = fn

    def get(self, name: str) -> Callable[..., Any]:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not registered")
        return self._tools[name]

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def generate_schemas(self, names: Sequence[str] | None = None) -> list[dict[str, Any]]:
        target = names or self._tools.keys()
        schemas = []
        for name in target:
            fn = self._tools[name]
            schemas.append(self._fn_to_schema(name, fn))
        return schemas

    @staticmethod
    def _fn_to_schema(name: str, fn: Callable[..., Any]) -> dict[str, Any]:
        sig = inspect.signature(fn)
        properties: dict[str, Any] = {}
        required: list[str] = []

        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

        for param_name, param in sig.parameters.items():
            annotation = param.annotation
            json_type = type_map.get(annotation, "string")
            properties[param_name] = {"type": json_type}
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "function",
            "function": {
                "name": name,
                "description": (fn.__doc__ or "").strip(),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
