from __future__ import annotations

from typing import Any, Callable

from j_perm import (
    ActionNode,
    Engine,
    ExecutionContext,
    Middleware,
    OpMatcher,
    build_default_engine,
)

from .handlers.agent_loop import AgentLoopHandler
from .handlers.handoff import HandoffHandler
from .handlers.llm import LlmHandler
from .handlers.tool import ToolHandler
from .registry import ToolRegistry


class _AgentMetadataMiddleware(Middleware):
    name = "agent_metadata"
    priority = 100

    def __init__(
        self,
        registry: ToolRegistry,
        schemas: list[dict[str, Any]],
        agent_specs: dict[str, Any],
    ) -> None:
        self._registry = registry
        self._schemas = schemas
        self._agent_specs = agent_specs

    def process(self, step: Any, ctx: ExecutionContext) -> Any:
        ctx.metadata.setdefault("_tool_registry", self._registry)
        ctx.metadata.setdefault("_tool_schemas", self._schemas)
        ctx.metadata.setdefault("_agent_specs", self._agent_specs)
        return step


def build_agent_engine(
    *,
    tools: dict[str, Callable[..., Any]] | None = None,
    agent_specs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Engine:
    engine = build_default_engine(**kwargs)

    registry = ToolRegistry()
    for name, fn in (tools or {}).items():
        registry.register(name, fn)

    schemas = registry.generate_schemas()
    specs = agent_specs or {}

    engine.main_pipeline.register_middleware(
        _AgentMetadataMiddleware(registry, schemas, specs)
    )

    handlers = [
        ("tool", ToolHandler()),
        ("llm", LlmHandler()),
        ("agent_loop", AgentLoopHandler()),
        ("handoff", HandoffHandler()),
    ]

    for op_name, handler in handlers:
        engine.main_pipeline.registry.register(
            ActionNode(
                name=op_name,
                priority=10,
                matcher=OpMatcher(op_name),
                handler=handler,
            )
        )

    return engine
