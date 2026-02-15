import json
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, patch

import pytest

from perm_agent import build_agent_engine
from perm_agent.events import (
    AgentCompleteEvent,
    StreamEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)


def _make_stream_chunks(tokens: list[str]) -> list[MagicMock]:
    """Build a list of mock streaming chunks from token strings."""
    chunks = []
    for token in tokens:
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = token
        chunk.choices[0].delta.tool_calls = None
        chunks.append(chunk)
    return chunks


def _make_tool_call_chunks(
    tool_calls: list[dict],
) -> list[MagicMock]:
    """Build streaming chunks that represent tool calls."""
    chunks = []
    for i, tc in enumerate(tool_calls):
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = None

        tc_mock = MagicMock()
        tc_mock.index = i
        tc_mock.id = tc["id"]
        tc_mock.function.name = tc["name"]
        tc_mock.function.arguments = json.dumps(tc["args"])

        chunk.choices[0].delta.tool_calls = [tc_mock]
        chunks.append(chunk)
    return chunks


def search(query: str) -> str:
    """Search for information"""
    return f"Found: {query}"


class TestStreamingLlmHandler:
    @patch("perm_agent.handlers.streaming.litellm.completion")
    def test_streaming_llm_yields_tokens(self, mock_completion):
        tokens = ["Hello", " ", "world", "!"]
        mock_completion.return_value = iter(_make_stream_chunks(tokens))

        collected_tokens: list[str] = []

        def on_event(event: StreamEvent) -> None:
            if isinstance(event, TokenEvent):
                collected_tokens.append(event.token)

        engine = build_agent_engine()
        spec = [
            {
                "op": "streaming_llm",
                "model": "openai/gpt-4o",
                "messages": [{"role": "user", "content": "Hi"}],
                "path": "/answer",
                "on_event": on_event,
            }
        ]
        engine.apply(spec, source={}, dest={})
        assert collected_tokens == ["Hello", " ", "world", "!"]

    @patch("perm_agent.handlers.streaming.litellm.completion")
    def test_streaming_llm_collects_full_response(self, mock_completion):
        tokens = ["Hello", " ", "world"]
        mock_completion.return_value = iter(_make_stream_chunks(tokens))

        engine = build_agent_engine()
        spec = [
            {
                "op": "streaming_llm",
                "model": "openai/gpt-4o",
                "messages": [{"role": "user", "content": "Hi"}],
                "path": "/answer",
            }
        ]
        result = engine.apply(spec, source={}, dest={})
        assert result == {"answer": "Hello world"}

    @patch("perm_agent.handlers.streaming.litellm.completion")
    def test_streaming_stores_result_at_path(self, mock_completion):
        mock_completion.return_value = iter(_make_stream_chunks(["result"]))

        engine = build_agent_engine()
        spec = [
            {
                "op": "streaming_llm",
                "model": "openai/gpt-4o",
                "messages": [{"role": "user", "content": "test"}],
                "path": "/output",
            }
        ]
        result = engine.apply(spec, source={}, dest={})
        assert result == {"output": "result"}

    @patch("perm_agent.handlers.streaming.litellm.completion")
    def test_streaming_without_path(self, mock_completion):
        mock_completion.return_value = iter(_make_stream_chunks(["direct"]))

        engine = build_agent_engine()
        spec = [
            {
                "op": "streaming_llm",
                "model": "openai/gpt-4o",
                "messages": [{"role": "user", "content": "test"}],
            }
        ]
        result = engine.apply(spec, source={}, dest={})
        assert result == "direct"

    @patch("perm_agent.handlers.streaming.litellm.completion")
    def test_streaming_callback_invoked(self, mock_completion):
        mock_completion.return_value = iter(_make_stream_chunks(["a", "b"]))

        callback = MagicMock()
        engine = build_agent_engine()
        spec = [
            {
                "op": "streaming_llm",
                "model": "openai/gpt-4o",
                "messages": [{"role": "user", "content": "test"}],
                "path": "/r",
                "on_event": callback,
            }
        ]
        engine.apply(spec, source={}, dest={})

        assert callback.call_count == 2
        callback.assert_any_call(TokenEvent(token="a"))
        callback.assert_any_call(TokenEvent(token="b"))


class TestStreamingAgentLoopHandler:
    @patch("perm_agent.handlers.streaming.litellm.completion")
    def test_streaming_agent_loop_token_events(self, mock_completion):
        tokens = ["The", " answer"]
        mock_completion.return_value = iter(_make_stream_chunks(tokens))

        collected: list[StreamEvent] = []
        engine = build_agent_engine(tools={"search": search})
        spec = [
            {
                "op": "streaming_agent_loop",
                "model": "openai/gpt-4o",
                "instructions": "You are helpful",
                "input": "What is the answer?",
                "tools": ["search"],
                "path": "/result",
                "on_event": collected.append,
            }
        ]
        engine.apply(spec, source={}, dest={})

        token_events = [e for e in collected if isinstance(e, TokenEvent)]
        assert len(token_events) == 2
        assert token_events[0].token == "The"
        assert token_events[1].token == " answer"

    @patch("perm_agent.handlers.streaming.litellm.completion")
    def test_streaming_agent_loop_tool_call_events(self, mock_completion):
        tool_chunks = _make_tool_call_chunks(
            [{"id": "call_1", "name": "search", "args": {"query": "test"}}]
        )
        text_chunks = _make_stream_chunks(["Done"])

        mock_completion.side_effect = [
            iter(tool_chunks),
            iter(text_chunks),
        ]

        collected: list[StreamEvent] = []
        engine = build_agent_engine(tools={"search": search})
        spec = [
            {
                "op": "streaming_agent_loop",
                "model": "openai/gpt-4o",
                "instructions": "Use search",
                "input": "Find test",
                "tools": ["search"],
                "path": "/result",
                "on_event": collected.append,
            }
        ]
        engine.apply(spec, source={}, dest={})

        tool_call_events = [e for e in collected if isinstance(e, ToolCallEvent)]
        assert len(tool_call_events) == 1
        assert tool_call_events[0].tool_name == "search"
        assert tool_call_events[0].arguments == {"query": "test"}

    @patch("perm_agent.handlers.streaming.litellm.completion")
    def test_streaming_agent_loop_tool_result_events(self, mock_completion):
        tool_chunks = _make_tool_call_chunks(
            [{"id": "call_1", "name": "search", "args": {"query": "test"}}]
        )
        text_chunks = _make_stream_chunks(["Done"])

        mock_completion.side_effect = [
            iter(tool_chunks),
            iter(text_chunks),
        ]

        collected: list[StreamEvent] = []
        engine = build_agent_engine(tools={"search": search})
        spec = [
            {
                "op": "streaming_agent_loop",
                "model": "openai/gpt-4o",
                "instructions": "Use search",
                "input": "Find test",
                "tools": ["search"],
                "path": "/result",
                "on_event": collected.append,
            }
        ]
        engine.apply(spec, source={}, dest={})

        result_events = [e for e in collected if isinstance(e, ToolResultEvent)]
        assert len(result_events) == 1
        assert result_events[0].tool_name == "search"
        assert result_events[0].result == "Found: test"

    @patch("perm_agent.handlers.streaming.litellm.completion")
    def test_streaming_agent_complete_event(self, mock_completion):
        mock_completion.return_value = iter(_make_stream_chunks(["Final", " answer"]))

        collected: list[StreamEvent] = []
        engine = build_agent_engine(tools={"search": search})
        spec = [
            {
                "op": "streaming_agent_loop",
                "model": "openai/gpt-4o",
                "instructions": "Help",
                "input": "test",
                "tools": ["search"],
                "path": "/result",
                "on_event": collected.append,
            }
        ]
        engine.apply(spec, source={}, dest={})

        complete_events = [e for e in collected if isinstance(e, AgentCompleteEvent)]
        assert len(complete_events) == 1
        assert complete_events[0].result == "Final answer"


class TestEventDataclasses:
    def test_event_dataclass_immutability(self):
        event = TokenEvent(token="hello")
        with pytest.raises(FrozenInstanceError):
            event.token = "world"

        tool_event = ToolCallEvent(tool_name="search", arguments={"q": "test"})
        with pytest.raises(FrozenInstanceError):
            tool_event.tool_name = "other"

        result_event = ToolResultEvent(tool_name="search", result="data")
        with pytest.raises(FrozenInstanceError):
            result_event.result = "changed"

        complete_event = AgentCompleteEvent(result="done")
        with pytest.raises(FrozenInstanceError):
            complete_event.result = "changed"

    def test_event_default_values(self):
        token = TokenEvent()
        assert token.event_type == "token"
        assert token.token == ""

        tool_call = ToolCallEvent()
        assert tool_call.event_type == "tool_call"
        assert tool_call.tool_name == ""
        assert tool_call.arguments == {}

        tool_result = ToolResultEvent()
        assert tool_result.event_type == "tool_result"
        assert tool_result.result is None

        complete = AgentCompleteEvent()
        assert complete.event_type == "agent_complete"
        assert complete.result is None

    def test_event_slots(self):
        """Verify slots=True is effective (no __dict__)."""
        event = TokenEvent(token="x")
        assert not hasattr(event, "__dict__")
