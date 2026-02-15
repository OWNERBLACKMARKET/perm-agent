from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """Base event for streaming."""

    event_type: str


@dataclass(frozen=True, slots=True)
class TokenEvent(StreamEvent):
    """Emitted for each token received from the LLM."""

    event_type: str = "token"
    token: str = ""


@dataclass(frozen=True, slots=True)
class ToolCallEvent(StreamEvent):
    """Emitted when the agent invokes a tool."""

    event_type: str = "tool_call"
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolResultEvent(StreamEvent):
    """Emitted after a tool returns its result."""

    event_type: str = "tool_result"
    tool_name: str = ""
    result: Any = None


@dataclass(frozen=True, slots=True)
class AgentCompleteEvent(StreamEvent):
    """Emitted when the agent loop finishes."""

    event_type: str = "agent_complete"
    result: Any = None


class EventHandler(Protocol):
    def on_event(self, event: StreamEvent) -> None: ...
