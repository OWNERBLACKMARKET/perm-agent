from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class SpanEvent:
    """A timestamped event within a span."""

    timestamp: float
    name: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    """Represents a single operation span."""

    span_id: str
    parent_id: str | None
    operation: str
    name: str
    start_time: float
    end_time: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)
    status: str = "ok"
    error: str | None = None


class TracerHook(Protocol):
    """Observer interface for span lifecycle events."""

    def on_span_start(self, span: Span) -> None: ...
    def on_span_end(self, span: Span) -> None: ...
    def on_event(self, span_id: str, event: SpanEvent) -> None: ...


class Tracer:
    """Collects spans and events during workflow execution."""

    def __init__(self, hooks: list[TracerHook] | None = None) -> None:
        self._spans: dict[str, Span] = {}
        self._completed: list[Span] = []
        self._hooks: list[TracerHook] = hooks or []
        self._active_span_id: str | None = None

    def start_span(
        self,
        operation: str,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        span_id = uuid.uuid4().hex[:16]
        span = Span(
            span_id=span_id,
            parent_id=self._active_span_id,
            operation=operation,
            name=name,
            start_time=time.monotonic(),
            metadata=metadata or {},
        )
        self._spans[span_id] = span
        self._active_span_id = span_id
        for hook in self._hooks:
            hook.on_span_start(span)
        return span_id

    def end_span(
        self,
        span_id: str,
        status: str = "ok",
        error: str | None = None,
    ) -> None:
        span = self._spans.pop(span_id, None)
        if span is None:
            return
        span.end_time = time.monotonic()
        span.status = status
        span.error = error
        self._completed.append(span)
        if self._active_span_id == span_id:
            self._active_span_id = span.parent_id
        for hook in self._hooks:
            hook.on_span_end(span)

    def add_event(
        self,
        span_id: str,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        event = SpanEvent(
            timestamp=time.monotonic(),
            name=name,
            attributes=attributes or {},
        )
        span = self._spans.get(span_id)
        if span is not None:
            span.events.append(event)
        for hook in self._hooks:
            hook.on_event(span_id, event)

    def get_span(self, span_id: str) -> Span | None:
        """Return an active (in-flight) span by its ID, or None."""
        return self._spans.get(span_id)

    @property
    def spans(self) -> list[Span]:
        return list(self._completed)

    def to_dict(self) -> list[dict[str, Any]]:
        """Export all completed spans as JSON-serializable dicts."""
        result = []
        for s in self._completed:
            result.append(
                {
                    "span_id": s.span_id,
                    "parent_id": s.parent_id,
                    "operation": s.operation,
                    "name": s.name,
                    "start_time": s.start_time,
                    "end_time": s.end_time,
                    "duration": (s.end_time - s.start_time) if s.end_time is not None else None,
                    "metadata": s.metadata,
                    "events": [
                        {"timestamp": e.timestamp, "name": e.name, "attributes": e.attributes}
                        for e in s.events
                    ],
                    "status": s.status,
                    "error": s.error,
                }
            )
        return result


class ConsoleTracerHook:
    """Prints spans to console for debugging."""

    def on_span_start(self, span: Span) -> None:
        print(f"[TRACE] start {span.operation}: {span.name}")

    def on_span_end(self, span: Span) -> None:
        duration = (span.end_time - span.start_time) if span.end_time is not None else 0
        print(f"[TRACE] end   {span.operation}: {span.name} ({duration:.4f}s) [{span.status}]")

    def on_event(self, span_id: str, event: SpanEvent) -> None:
        print(f"[TRACE] event {event.name}")


class CostTracker:
    """Tracks token usage and estimated costs from LLM calls."""

    def __init__(self) -> None:
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_cost: float = 0.0
        self.calls: list[dict[str, Any]] = []

    def on_span_start(self, span: Span) -> None:
        pass

    def on_span_end(self, span: Span) -> None:
        if span.operation != "llm":
            return
        usage = span.metadata.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.calls.append(
            {
                "span_id": span.span_id,
                "model": span.metadata.get("model"),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        )

    def on_event(self, span_id: str, event: SpanEvent) -> None:
        pass
