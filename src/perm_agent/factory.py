from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
from .handlers.streaming import StreamingAgentLoopHandler, StreamingLlmHandler
from .handlers.tool import ToolHandler
from .registry import ToolRegistry

if TYPE_CHECKING:
    from collections.abc import Callable

    from .communication import CommunicationHub
    from .observability import Tracer


class _AgentMetadataMiddleware(Middleware):
    name = "agent_metadata"
    priority = 100

    def __init__(
        self,
        registry: ToolRegistry,
        schemas: list[dict[str, Any]],
        agent_specs: dict[str, Any],
        tracer: Tracer | None = None,
        communication: CommunicationHub | None = None,
    ) -> None:
        self._registry = registry
        self._schemas = schemas
        self._agent_specs = agent_specs
        self._tracer = tracer
        self._communication = communication

    def process(self, step: Any, ctx: ExecutionContext) -> Any:
        ctx.metadata.setdefault("_tool_registry", self._registry)
        ctx.metadata.setdefault("_tool_schemas", self._schemas)
        ctx.metadata.setdefault("_agent_specs", self._agent_specs)
        if self._tracer is not None:
            ctx.metadata.setdefault("_tracer", self._tracer)
        if self._communication is not None:
            ctx.metadata.setdefault("_communication", self._communication)
            if isinstance(step, dict):
                agent_name = step.get("agent_name")
                if agent_name:
                    self._communication.agent_context.current_agent = agent_name
        return step


def build_agent_engine(
    *,
    tools: dict[str, Callable[..., Any]] | None = None,
    agent_specs: dict[str, Any] | None = None,
    tracer: Tracer | None = None,
    communication: CommunicationHub | None = None,
    **kwargs: Any,
) -> Engine:
    engine = build_default_engine(**kwargs)

    registry = ToolRegistry()
    for name, fn in (tools or {}).items():
        registry.register(name, fn)

    if communication is not None:
        from .comm_tools import build_comm_tools

        for name, fn in build_comm_tools(communication).items():
            registry.register(name, fn)

    schemas = registry.generate_schemas()
    specs = agent_specs or {}

    engine.main_pipeline.register_middleware(
        _AgentMetadataMiddleware(
            registry, schemas, specs, tracer=tracer, communication=communication,
        )
    )

    handlers = [
        ("tool", ToolHandler()),
        ("llm", LlmHandler()),
        ("agent_loop", AgentLoopHandler()),
        ("handoff", HandoffHandler()),
        ("streaming_llm", StreamingLlmHandler()),
        ("streaming_agent_loop", StreamingAgentLoopHandler()),
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
