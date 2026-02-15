from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import litellm

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..registry import ToolRegistry


class AsyncAgentLoopHandler:
    """Async version of AgentLoopHandler using litellm.acompletion."""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    async def execute(
        self,
        *,
        model: str,
        instructions: str,
        input: str,
        tools: list[str] | None = None,
        max_iterations: int = 10,
        memory_limit: int = 20,
    ) -> str:
        schemas = self._registry.generate_schemas()
        tool_schemas = [s for s in schemas if s["function"]["name"] in tools] if tools else schemas

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": str(input)},
        ]

        for _ in range(max_iterations):
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages[-memory_limit:],
            }
            if tool_schemas:
                kwargs["tools"] = tool_schemas

            response = await litellm.acompletion(**kwargs)
            choice = response.choices[0].message

            if not choice.tool_calls:
                return choice.content

            messages.append(choice.model_dump())

            for tool_call in choice.tool_calls:
                fn_name = tool_call.function.name

                try:
                    fn_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as e:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"Error parsing arguments for tool '{fn_name}': {e}",
                        }
                    )
                    continue

                fn = self._registry.get(fn_name)
                try:
                    result = self._call(fn, fn_args)
                    if asyncio.iscoroutine(result):
                        result = await result
                except Exception as e:
                    result = f"Error executing tool '{fn_name}': {type(e).__name__}: {e}"

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result) if not isinstance(result, str) else result,
                    }
                )

        return messages[-1].get("content", "")

    @staticmethod
    def _call(fn: Callable[..., Any], args: Any) -> Any:
        if isinstance(args, dict):
            return fn(**args)
        return fn(args)
