from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..registry import ToolRegistry


class AsyncToolHandler:
    """Async tool handler that supports both sync and async tool functions."""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    async def execute(
        self,
        *,
        name: str,
        args: dict[str, Any] | list[Any] | Any | None = None,
    ) -> Any:
        fn = self._registry.get(name)
        resolved_args = args if args is not None else {}

        result = self._call(fn, resolved_args)
        if asyncio.iscoroutine(result):
            result = await result

        return result

    @staticmethod
    def _call(fn: Callable[..., Any], args: Any) -> Any:
        if isinstance(args, dict):
            return fn(**args)
        if isinstance(args, list):
            return fn(*args)
        return fn(args)
