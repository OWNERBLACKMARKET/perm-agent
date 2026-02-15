from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

from .factory import build_agent_engine

if TYPE_CHECKING:
    from collections.abc import Callable


class Agent:
    """High-level Python API for building agents on top of the JSON spec engine."""

    def __init__(
        self,
        *,
        name: str,
        model: str,
        instructions: str,
        tools: list[Callable[..., Any]] | None = None,
        max_iterations: int = 10,
        memory_limit: int = 20,
        temperature: float | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.instructions = instructions
        self.tools = tools or []
        self.max_iterations = max_iterations
        self.memory_limit = memory_limit
        self.temperature = temperature

    def run(self, input: str, **context: Any) -> str:
        """Run the agent synchronously and return the final text output."""
        tool_map = {fn.__name__: fn for fn in self.tools}
        tool_names = list(tool_map.keys())

        spec = self._build_spec(tool_names, input)
        engine = build_agent_engine(tools=tool_map)
        result = engine.apply(spec, source=context, dest={})
        return result.get("result", "") if isinstance(result, dict) else result

    def to_spec(self) -> list[dict[str, Any]]:
        """Export the agent configuration as a JSON spec."""
        tool_names = [fn.__name__ for fn in self.tools]
        return self._build_spec(tool_names, "${/input}")

    @classmethod
    def from_spec(
        cls,
        spec: dict[str, Any],
        tools: dict[str, Callable[..., Any]] | None = None,
    ) -> Agent:
        """Create an Agent from a previously exported spec dict.

        Expects the metadata format:
        {name, model, instructions, tools, max_iterations, memory_limit, ...}
        """
        tool_map = tools or {}
        tool_callables = [tool_map[name] for name in spec.get("tools", []) if name in tool_map]

        kwargs: dict[str, Any] = {
            "name": spec["name"],
            "model": spec["model"],
            "instructions": spec["instructions"],
            "tools": tool_callables,
            "max_iterations": spec.get("max_iterations", 10),
            "memory_limit": spec.get("memory_limit", 20),
        }
        if "temperature" in spec:
            kwargs["temperature"] = spec["temperature"]

        return cls(**kwargs)

    def _build_spec(self, tool_names: list[str], input_expr: str) -> list[dict[str, Any]]:
        step: dict[str, Any] = {
            "op": "agent_loop",
            "model": self.model,
            "instructions": self.instructions,
            "input": input_expr,
            "tools": tool_names,
            "max_iterations": self.max_iterations,
            "memory_limit": self.memory_limit,
            "path": "/result",
        }
        if self.temperature is not None:
            step["temperature"] = self.temperature
        return [step]


def agent(
    *,
    model: str,
    tools: list[Callable[..., Any]] | None = None,
    max_iterations: int = 10,
    memory_limit: int = 20,
    temperature: float | None = None,
) -> Callable[[Callable[..., str]], Callable[..., str]]:
    """Decorator that turns a function into an agent.

    The decorated function's docstring becomes the agent instructions.
    The first positional argument becomes the agent input.
    """

    def decorator(fn: Callable[..., str]) -> Callable[..., str]:
        instructions = (fn.__doc__ or "").strip()
        name = fn.__name__

        wrapped_agent = Agent(
            name=name,
            model=model,
            instructions=instructions,
            tools=tools,
            max_iterations=max_iterations,
            memory_limit=memory_limit,
            temperature=temperature,
        )

        @functools.wraps(fn)
        def wrapper(input_text: str, **context: Any) -> str:
            return wrapped_agent.run(input_text, **context)

        wrapper._agent = wrapped_agent  # type: ignore[attr-defined]
        return wrapper

    return decorator


class Pipeline:
    """Multi-step pipeline that chains agents and spec steps together."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._steps: list[dict[str, Any]] = []
        self._tools: dict[str, Callable[..., Any]] = {}

    def add_step(
        self,
        agent_or_spec: Agent | list[dict[str, Any]],
        *,
        input_map: dict[str, str] | None = None,
        output_path: str | None = None,
    ) -> Pipeline:
        """Add a step to the pipeline.

        Args:
            agent_or_spec: Either an Agent instance or a raw JSON spec list.
            input_map: Optional mapping of {dest_key: source_template} to prepare input.
            output_path: Path where the step result is stored in dest.
        """
        if isinstance(agent_or_spec, Agent):
            for fn in agent_or_spec.tools:
                self._tools[fn.__name__] = fn

            tool_names = [fn.__name__ for fn in agent_or_spec.tools]

            # Use input_map to build the input expression, or default
            input_expr = input_map["input"] if input_map and "input" in input_map else "${/input}"

            step: dict[str, Any] = {
                "op": "agent_loop",
                "model": agent_or_spec.model,
                "instructions": agent_or_spec.instructions,
                "input": input_expr,
                "tools": tool_names,
                "max_iterations": agent_or_spec.max_iterations,
                "memory_limit": agent_or_spec.memory_limit,
            }
            if agent_or_spec.temperature is not None:
                step["temperature"] = agent_or_spec.temperature
            if output_path:
                step["path"] = output_path

            # If input_map has keys other than "input", add set ops before the agent step
            if input_map:
                for key, val in input_map.items():
                    if key != "input":
                        self._steps.append({"op": "set", "path": key, "value": val})

            self._steps.append(step)
        else:
            # Raw spec list
            if output_path:
                for s in agent_or_spec:
                    if "path" not in s:
                        s["path"] = output_path
            self._steps.extend(agent_or_spec)

        return self

    def run(self, input: dict[str, Any]) -> dict[str, Any]:
        """Execute the pipeline with the given input dict."""
        engine = build_agent_engine(tools=self._tools if self._tools else None)
        result = engine.apply(self._steps, source=input, dest={})
        return result if isinstance(result, dict) else {"result": result}

    def to_spec(self) -> list[dict[str, Any]]:
        """Export the pipeline as a JSON spec."""
        return list(self._steps)
