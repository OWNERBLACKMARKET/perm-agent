import json
from unittest.mock import MagicMock, patch

import pytest

from perm_agent.observability import (
    ConsoleTracerHook,
    CostTracker,
    Span,
    SpanEvent,
    Tracer,
)


class TestTracer:
    def test_creates_spans(self):
        tracer = Tracer()
        span_id = tracer.start_span("llm", "gpt-4o")
        tracer.end_span(span_id)

        assert len(tracer.spans) == 1
        assert tracer.spans[0].operation == "llm"
        assert tracer.spans[0].name == "gpt-4o"

    def test_span_lifecycle(self):
        tracer = Tracer()
        span_id = tracer.start_span("tool", "search")

        # Span is active, not yet in completed list
        assert len(tracer.spans) == 0

        tracer.end_span(span_id)

        assert len(tracer.spans) == 1
        span = tracer.spans[0]
        assert span.start_time > 0
        assert span.end_time is not None
        assert span.end_time >= span.start_time
        assert span.status == "ok"
        assert span.error is None

    def test_nested_spans(self):
        tracer = Tracer()
        parent_id = tracer.start_span("agent_loop", "gpt-4o")
        child_id = tracer.start_span("tool", "search")

        # Child should have parent_id set
        child_span = tracer._spans[child_id]
        assert child_span.parent_id == parent_id

        tracer.end_span(child_id)
        tracer.end_span(parent_id)

        assert len(tracer.spans) == 2
        child = [s for s in tracer.spans if s.operation == "tool"][0]
        assert child.parent_id == parent_id

    def test_events(self):
        tracer = Tracer()
        span_id = tracer.start_span("llm", "gpt-4o")
        tracer.add_event(span_id, "llm.request", {"model": "gpt-4o"})
        tracer.add_event(span_id, "llm.response", {"tokens": 100})
        tracer.end_span(span_id)

        span = tracer.spans[0]
        assert len(span.events) == 2
        assert span.events[0].name == "llm.request"
        assert span.events[0].attributes["model"] == "gpt-4o"
        assert span.events[1].name == "llm.response"

    def test_span_metadata(self):
        tracer = Tracer()
        span_id = tracer.start_span("llm", "gpt-4o", metadata={"model": "gpt-4o"})
        tracer.end_span(span_id)

        assert tracer.spans[0].metadata == {"model": "gpt-4o"}

    def test_captures_errors(self):
        tracer = Tracer()
        span_id = tracer.start_span("tool", "failing_tool")
        tracer.end_span(span_id, status="error", error="Something broke")

        span = tracer.spans[0]
        assert span.status == "error"
        assert span.error == "Something broke"

    def test_to_dict_serializable(self):
        tracer = Tracer()
        span_id = tracer.start_span("llm", "gpt-4o", metadata={"model": "gpt-4o"})
        tracer.add_event(span_id, "llm.request", {"count": 3})
        tracer.end_span(span_id)

        data = tracer.to_dict()
        # Should be JSON-serializable
        serialized = json.dumps(data)
        assert isinstance(serialized, str)

        assert len(data) == 1
        assert data[0]["operation"] == "llm"
        assert data[0]["name"] == "gpt-4o"
        assert data[0]["duration"] is not None
        assert data[0]["duration"] >= 0
        assert len(data[0]["events"]) == 1
        assert data[0]["metadata"]["model"] == "gpt-4o"

    def test_end_nonexistent_span_is_noop(self):
        tracer = Tracer()
        tracer.end_span("nonexistent")
        assert len(tracer.spans) == 0

    def test_multiple_spans(self):
        tracer = Tracer()
        for i in range(5):
            sid = tracer.start_span("tool", f"tool_{i}")
            tracer.end_span(sid)

        assert len(tracer.spans) == 5


class TestTracerHooks:
    def test_hook_called_on_start(self):
        hook = MagicMock()
        tracer = Tracer(hooks=[hook])
        tracer.start_span("llm", "gpt-4o")

        hook.on_span_start.assert_called_once()
        span = hook.on_span_start.call_args[0][0]
        assert span.operation == "llm"

    def test_hook_called_on_end(self):
        hook = MagicMock()
        tracer = Tracer(hooks=[hook])
        span_id = tracer.start_span("llm", "gpt-4o")
        tracer.end_span(span_id)

        hook.on_span_end.assert_called_once()
        span = hook.on_span_end.call_args[0][0]
        assert span.end_time is not None

    def test_hook_called_on_event(self):
        hook = MagicMock()
        tracer = Tracer(hooks=[hook])
        span_id = tracer.start_span("llm", "gpt-4o")
        tracer.add_event(span_id, "llm.request", {"model": "gpt-4o"})

        hook.on_event.assert_called_once()
        call_args = hook.on_event.call_args[0]
        assert call_args[0] == span_id
        assert call_args[1].name == "llm.request"

    def test_multiple_hooks(self):
        hook1 = MagicMock()
        hook2 = MagicMock()
        tracer = Tracer(hooks=[hook1, hook2])
        span_id = tracer.start_span("tool", "search")
        tracer.end_span(span_id)

        assert hook1.on_span_start.call_count == 1
        assert hook2.on_span_start.call_count == 1
        assert hook1.on_span_end.call_count == 1
        assert hook2.on_span_end.call_count == 1


class TestConsoleTracerHook:
    def test_prints_on_start(self, capsys):
        hook = ConsoleTracerHook()
        span = Span(
            span_id="abc",
            parent_id=None,
            operation="llm",
            name="gpt-4o",
            start_time=1.0,
        )
        hook.on_span_start(span)
        captured = capsys.readouterr()
        assert "llm" in captured.out
        assert "gpt-4o" in captured.out

    def test_prints_on_end(self, capsys):
        hook = ConsoleTracerHook()
        span = Span(
            span_id="abc",
            parent_id=None,
            operation="llm",
            name="gpt-4o",
            start_time=1.0,
            end_time=1.5,
        )
        hook.on_span_end(span)
        captured = capsys.readouterr()
        assert "llm" in captured.out
        assert "0.5" in captured.out

    def test_prints_on_event(self, capsys):
        hook = ConsoleTracerHook()
        event = SpanEvent(timestamp=1.0, name="llm.request")
        hook.on_event("abc", event)
        captured = capsys.readouterr()
        assert "llm.request" in captured.out


class TestCostTracker:
    def test_accumulates_tokens(self):
        tracker = CostTracker()
        span = Span(
            span_id="s1",
            parent_id=None,
            operation="llm",
            name="gpt-4o",
            start_time=1.0,
            end_time=2.0,
            metadata={"usage": {"input_tokens": 100, "output_tokens": 50}},
        )
        tracker.on_span_end(span)

        assert tracker.total_input_tokens == 100
        assert tracker.total_output_tokens == 50
        assert len(tracker.calls) == 1

    def test_ignores_non_llm_spans(self):
        tracker = CostTracker()
        span = Span(
            span_id="s1",
            parent_id=None,
            operation="tool",
            name="search",
            start_time=1.0,
            end_time=2.0,
        )
        tracker.on_span_end(span)

        assert tracker.total_input_tokens == 0
        assert len(tracker.calls) == 0

    def test_accumulates_across_calls(self):
        tracker = CostTracker()
        for i in range(3):
            span = Span(
                span_id=f"s{i}",
                parent_id=None,
                operation="llm",
                name="gpt-4o",
                start_time=1.0,
                end_time=2.0,
                metadata={"usage": {"input_tokens": 100, "output_tokens": 50}},
            )
            tracker.on_span_end(span)

        assert tracker.total_input_tokens == 300
        assert tracker.total_output_tokens == 150
        assert len(tracker.calls) == 3

    def test_handles_missing_usage(self):
        tracker = CostTracker()
        span = Span(
            span_id="s1",
            parent_id=None,
            operation="llm",
            name="gpt-4o",
            start_time=1.0,
            end_time=2.0,
            metadata={},
        )
        tracker.on_span_end(span)

        assert tracker.total_input_tokens == 0
        assert tracker.total_output_tokens == 0


def _make_response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.choices[0].message.tool_calls = None
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 5
    return response


def _make_tool_response(tool_calls: list[dict]) -> MagicMock:
    response = MagicMock()
    choice = MagicMock()
    choice.message.content = None
    calls = []
    for tc in tool_calls:
        call = MagicMock()
        call.id = tc["id"]
        call.function.name = tc["name"]
        call.function.arguments = json.dumps(tc["args"])
        calls.append(call)
    choice.message.tool_calls = calls
    choice.message.model_dump.return_value = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc["args"]),
                },
            }
            for tc in tool_calls
        ],
    }
    response.choices = [choice]
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 5
    return response


def search(query: str) -> str:
    """Search for information"""
    return f"Found: {query}"


class TestHandlerSpanIntegration:
    @patch("litellm.completion")
    def test_llm_handler_emits_spans(self, mock_completion):
        from perm_agent import build_agent_engine

        mock_completion.return_value = _make_response("Hello!")
        tracer = Tracer()
        engine = build_agent_engine(tracer=tracer)

        spec = [
            {
                "op": "llm",
                "model": "openai/gpt-4o",
                "messages": [{"role": "user", "content": "Hi"}],
                "path": "/answer",
            }
        ]
        result = engine.apply(spec, source={}, dest={})

        assert result == {"answer": "Hello!"}
        assert len(tracer.spans) == 1
        span = tracer.spans[0]
        assert span.operation == "llm"
        assert span.name == "openai/gpt-4o"
        assert span.status == "ok"
        # Should have llm.request and llm.response events
        event_names = [e.name for e in span.events]
        assert "llm.request" in event_names
        assert "llm.response" in event_names

    def test_tool_handler_emits_spans(self):
        from perm_agent import build_agent_engine

        tracer = Tracer()
        engine = build_agent_engine(tools={"search": search}, tracer=tracer)

        spec = [{"op": "tool", "name": "search", "args": {"query": "hello"}, "path": "/result"}]
        result = engine.apply(spec, source={}, dest={})

        assert result == {"result": "Found: hello"}
        assert len(tracer.spans) == 1
        span = tracer.spans[0]
        assert span.operation == "tool"
        assert span.name == "search"
        event_names = [e.name for e in span.events]
        assert "tool.execution" in event_names
        assert "tool.result" in event_names

    @patch("litellm.completion")
    def test_agent_loop_emits_iteration_events(self, mock_completion):
        from perm_agent import build_agent_engine

        mock_completion.side_effect = [
            _make_tool_response([{"id": "call_1", "name": "search", "args": {"query": "test"}}]),
            _make_response("Done"),
        ]
        tracer = Tracer()
        engine = build_agent_engine(tools={"search": search}, tracer=tracer)

        spec = [
            {
                "op": "agent_loop",
                "model": "openai/gpt-4o",
                "instructions": "Help",
                "input": "test",
                "tools": ["search"],
                "path": "/result",
            }
        ]
        engine.apply(spec, source={}, dest={})

        # 2 child llm spans + 1 parent agent_loop span
        assert len(tracer.spans) == 3
        llm_spans = [s for s in tracer.spans if s.operation == "llm"]
        loop_spans = [s for s in tracer.spans if s.operation == "agent_loop"]
        assert len(llm_spans) == 2
        assert len(loop_spans) == 1
        span = loop_spans[0]
        # Child spans have parent_id pointing to the loop span
        for ls in llm_spans:
            assert ls.parent_id == span.span_id
        iteration_events = [e for e in span.events if e.name == "agent_loop.iteration"]
        assert len(iteration_events) == 2
        assert iteration_events[0].attributes["iteration"] == 1
        assert iteration_events[1].attributes["iteration"] == 2

    def test_handoff_emits_spans(self):
        from perm_agent import build_agent_engine

        echo_spec = [{"/echo": "${/msg}"}]
        tracer = Tracer()
        engine = build_agent_engine(agent_specs={"echo": echo_spec}, tracer=tracer)

        spec = [{"op": "handoff", "to": "echo", "input": {"msg": "hello"}, "path": "/out"}]
        result = engine.apply(spec, source={}, dest={})

        assert result == {"out": {"echo": "hello"}}
        assert len(tracer.spans) == 1
        span = tracer.spans[0]
        assert span.operation == "handoff"
        assert span.name == "echo"
        event_names = [e.name for e in span.events]
        assert "handoff.delegate" in event_names

    @patch("litellm.completion")
    def test_tracer_captures_errors(self, mock_completion):
        from perm_agent import build_agent_engine

        mock_completion.side_effect = RuntimeError("API down")
        tracer = Tracer()
        engine = build_agent_engine(tracer=tracer)

        spec = [
            {
                "op": "llm",
                "model": "openai/gpt-4o",
                "messages": [{"role": "user", "content": "Hi"}],
                "path": "/answer",
            }
        ]
        with pytest.raises(RuntimeError, match="API down"):
            engine.apply(spec, source={}, dest={})

        assert len(tracer.spans) == 1
        span = tracer.spans[0]
        assert span.status == "error"
        assert "API down" in span.error

    def test_engine_with_tracer_param(self):
        """build_agent_engine accepts tracer and injects it."""
        from perm_agent import build_agent_engine

        tracer = Tracer()
        engine = build_agent_engine(
            tools={"search": search},
            tracer=tracer,
        )

        spec = [{"op": "tool", "name": "search", "args": {"query": "test"}, "path": "/r"}]
        engine.apply(spec, source={}, dest={})

        assert len(tracer.spans) == 1

    def test_engine_without_tracer_backward_compatible(self):
        """Engine works fine without tracer (backward compatibility)."""
        from perm_agent import build_agent_engine

        engine = build_agent_engine(tools={"search": search})
        spec = [{"op": "tool", "name": "search", "args": {"query": "test"}, "path": "/r"}]
        result = engine.apply(spec, source={}, dest={})
        assert result == {"r": "Found: test"}

    @patch("perm_agent.handlers.streaming.litellm.completion")
    def test_streaming_llm_emits_spans(self, mock_completion):
        """StreamingLlmHandler emits streaming_llm span with events."""
        from perm_agent import build_agent_engine

        chunks = []
        for token in ["Hello", " ", "World"]:
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = token
            chunk.choices[0].delta.tool_calls = None
            chunks.append(chunk)

        mock_completion.return_value = iter(chunks)
        tracer = Tracer()
        engine = build_agent_engine(tracer=tracer)

        spec = [
            {
                "op": "streaming_llm",
                "model": "openai/gpt-4o",
                "messages": [{"role": "user", "content": "Hi"}],
                "path": "/answer",
            }
        ]
        result = engine.apply(spec, source={}, dest={})

        assert result == {"answer": "Hello World"}
        assert len(tracer.spans) == 1
        span = tracer.spans[0]
        assert span.operation == "streaming_llm"
        assert span.status == "ok"
        event_names = [e.name for e in span.events]
        assert "streaming.start" in event_names
        assert "streaming.complete" in event_names

    @patch("perm_agent.handlers.streaming.litellm.completion")
    def test_streaming_agent_loop_emits_spans(self, mock_completion):
        """StreamingAgentLoopHandler emits span with iteration events."""
        from perm_agent import build_agent_engine

        chunks = []
        for token in ["Done"]:
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = token
            chunk.choices[0].delta.tool_calls = None
            chunks.append(chunk)

        mock_completion.return_value = iter(chunks)
        tracer = Tracer()
        engine = build_agent_engine(tools={"search": search}, tracer=tracer)

        spec = [
            {
                "op": "streaming_agent_loop",
                "model": "openai/gpt-4o",
                "instructions": "Help",
                "input": "test",
                "tools": ["search"],
                "path": "/result",
            }
        ]
        engine.apply(spec, source={}, dest={})

        loop_spans = [s for s in tracer.spans if s.operation == "streaming_agent_loop"]
        assert len(loop_spans) == 1
        span = loop_spans[0]
        assert span.status == "ok"
        iteration_events = [e for e in span.events if e.name == "streaming_agent_loop.iteration"]
        assert len(iteration_events) == 1
        assert iteration_events[0].attributes["iteration"] == 1

    @patch("litellm.completion")
    def test_agent_loop_child_spans_visible_to_cost_tracker(self, mock_completion):
        """Agent loop creates child llm spans that CostTracker can see."""
        from perm_agent import build_agent_engine

        mock_completion.return_value = _make_response("Answer")
        cost = CostTracker()
        tracer = Tracer(hooks=[cost])
        engine = build_agent_engine(tools={"search": search}, tracer=tracer)

        spec = [
            {
                "op": "agent_loop",
                "model": "openai/gpt-4o",
                "instructions": "Help",
                "input": "test",
                "tools": ["search"],
                "path": "/result",
            }
        ]
        engine.apply(spec, source={}, dest={})

        # CostTracker should see the child llm span
        assert len(cost.calls) == 1
        assert cost.total_input_tokens == 10
        assert cost.total_output_tokens == 5


class TestTracerGetSpan:
    def test_get_span_returns_active_span(self):
        tracer = Tracer()
        span_id = tracer.start_span("llm", "test")
        span = tracer.get_span(span_id)
        assert span is not None
        assert span.span_id == span_id
        tracer.end_span(span_id)

    def test_get_span_returns_none_after_end(self):
        tracer = Tracer()
        span_id = tracer.start_span("llm", "test")
        tracer.end_span(span_id)
        assert tracer.get_span(span_id) is None

    def test_get_span_returns_none_for_unknown_id(self):
        tracer = Tracer()
        assert tracer.get_span("nonexistent") is None
