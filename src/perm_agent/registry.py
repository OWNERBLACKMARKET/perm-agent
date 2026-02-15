from __future__ import annotations

import dataclasses
import inspect
import re
import types
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, Union, get_args, get_origin, get_type_hints

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


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

        # Resolve string annotations to actual types
        try:
            hints = get_type_hints(fn)
        except Exception:
            hints = {}

        param_descriptions = _parse_docstring_params(fn.__doc__ or "")

        for param_name, param in sig.parameters.items():
            annotation = hints.get(param_name, param.annotation)
            is_optional_type = _is_optional(annotation)

            prop = _resolve_type(annotation)
            if param_name in param_descriptions:
                prop["description"] = param_descriptions[param_name]

            properties[param_name] = prop

            if param.default is inspect.Parameter.empty and not is_optional_type:
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


def _is_optional(annotation: Any) -> bool:
    """Check if a type annotation is Optional (Union[X, None])."""
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        args = get_args(annotation)
        return type(None) in args
    return False


_BASIC_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _resolve_type(annotation: Any) -> dict[str, Any]:
    """Resolve a Python type annotation to JSON Schema."""
    # Handle missing annotation
    if annotation is inspect.Parameter.empty:
        return {"type": "string"}

    origin = get_origin(annotation)
    args = get_args(annotation)

    # Handle Optional[X] / Union[X, None] / X | None
    if origin is Union or origin is types.UnionType:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _resolve_type(non_none[0])
        # General Union — just use string fallback
        return {"type": "string"}

    # Handle list[X]
    if origin is list:
        if args:
            return {"type": "array", "items": _resolve_type(args[0])}
        return {"type": "array"}

    # Handle dict[K, V]
    if origin is dict:
        return {"type": "object"}

    # Handle Literal["a", "b"]
    if origin is Literal:
        return {"type": "string", "enum": list(args)}

    # Handle Enum subclasses
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return {"type": "string", "enum": [e.value for e in annotation]}

    # Handle Pydantic BaseModel
    try:
        from pydantic import BaseModel

        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return annotation.model_json_schema()
    except ImportError:
        pass

    # Handle dataclass
    if dataclasses.is_dataclass(annotation) and isinstance(annotation, type):
        return _dataclass_to_schema(annotation)

    # Basic types
    json_type = _BASIC_TYPE_MAP.get(annotation, "string")
    return {"type": json_type}


def _dataclass_to_schema(cls: type) -> dict[str, Any]:
    """Generate a basic JSON schema from a dataclass."""
    properties: dict[str, Any] = {}
    required: list[str] = []

    try:
        hints = get_type_hints(cls)
    except Exception:
        hints = {}

    for field in dataclasses.fields(cls):
        annotation = hints.get(field.name, field.type)
        properties[field.name] = _resolve_type(annotation)
        if field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING:
            required.append(field.name)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "title": cls.__name__,
    }
    if required:
        schema["required"] = required
    return schema


def _parse_docstring_params(docstring: str) -> dict[str, str]:
    """Parse Google-style and NumPy-style docstring for parameter descriptions."""
    if not docstring:
        return {}

    params: dict[str, str] = {}

    # Google style: "Args:" section with "param: description" or "param (type): description"
    google_pattern = (
        r"(?:Args|Arguments|Parameters)\s*:\s*\n"
        r"(.*?)(?:\n\s*\n|\n\S|\Z)"
    )
    google_match = re.search(google_pattern, docstring, re.DOTALL)
    if google_match:
        block = google_match.group(1)
        # Match lines like "    param: desc" or "    param (type): desc"
        param_pattern = (
            r"^\s+(\w+)(?:\s*\([^)]*\))?\s*:\s*(.+?)"
            r"(?=\n\s+\w+(?:\s*\([^)]*\))?\s*:|\Z)"
        )
        for match in re.finditer(param_pattern, block, re.MULTILINE | re.DOTALL):
            name = match.group(1)
            desc = re.sub(r"\s+", " ", match.group(2)).strip()
            params[name] = desc

    return params


def tool(fn: Callable[..., Any] | None = None, *, name: str | None = None) -> Any:
    """Decorator that marks a function as a tool and attaches metadata.

    Usage:
        @tool
        def search(query: str) -> str: ...

        @tool(name="custom_name")
        def search(query: str) -> str: ...
    """

    def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = name or f.__name__
        schema = ToolRegistry._fn_to_schema(tool_name, f)
        f._tool_metadata = {  # type: ignore[attr-defined]
            "name": tool_name,
            "schema": schema,
        }
        return f

    if fn is not None:
        return decorator(fn)
    return decorator
