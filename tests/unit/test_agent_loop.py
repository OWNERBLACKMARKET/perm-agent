import json
from unittest.mock import MagicMock, patch

from perm_agent import build_agent_engine


def _make_text_response(content: str) -> MagicMock:
    response = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    choice.message.tool_calls = None
    response.choices = [choice]
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
            {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])}}
            for tc in tool_calls
        ],
    }
    response.choices = [choice]
    return response


def search(query: str) -> str:
    """Search for information"""
    return f"Found: {query}"


class TestAgentLoopHandler:
    @patch("litellm.completion")
    def test_direct_response(self, mock_completion):
        mock_completion.return_value = _make_text_response("The answer is 42")
        engine = build_agent_engine(tools={"search": search})

        spec = [
            {
                "op": "agent_loop",
                "model": "openai/gpt-4o",
                "instructions": "You are helpful",
                "input": "What is the answer?",
                "tools": ["search"],
                "path": "/result",
            }
        ]
        result = engine.apply(spec, source={}, dest={})
        assert result == {"result": "The answer is 42"}

    @patch("litellm.completion")
    def test_tool_call_then_response(self, mock_completion):
        mock_completion.side_effect = [
            _make_tool_response([{"id": "call_1", "name": "search", "args": {"query": "test"}}]),
            _make_text_response("Based on search: Found: test"),
        ]
        engine = build_agent_engine(tools={"search": search})

        spec = [
            {
                "op": "agent_loop",
                "model": "openai/gpt-4o",
                "instructions": "Use search when needed",
                "input": "Find test",
                "tools": ["search"],
                "path": "/result",
            }
        ]
        result = engine.apply(spec, source={}, dest={})
        assert result == {"result": "Based on search: Found: test"}
        assert mock_completion.call_count == 2

    @patch("litellm.completion")
    def test_max_iterations(self, mock_completion):
        mock_completion.return_value = _make_tool_response(
            [{"id": "call_x", "name": "search", "args": {"query": "loop"}}]
        )
        engine = build_agent_engine(tools={"search": search})

        spec = [
            {
                "op": "agent_loop",
                "model": "openai/gpt-4o",
                "instructions": "Keep searching",
                "input": "Loop forever",
                "tools": ["search"],
                "max_iterations": 3,
                "path": "/result",
            }
        ]
        result = engine.apply(spec, source={}, dest={})
        assert mock_completion.call_count == 3

    @patch("litellm.completion")
    def test_template_input(self, mock_completion):
        mock_completion.return_value = _make_text_response("Done")
        engine = build_agent_engine(tools={"search": search})

        spec = [
            {
                "op": "agent_loop",
                "model": "openai/gpt-4o",
                "instructions": "Help",
                "input": "${/question}",
                "tools": ["search"],
                "path": "/result",
            }
        ]
        result = engine.apply(spec, source={"question": "What is Python?"}, dest={})
        assert result == {"result": "Done"}

        call_kwargs = mock_completion.call_args[1]
        user_msg = [m for m in call_kwargs["messages"] if m["role"] == "user"][0]
        assert user_msg["content"] == "What is Python?"

    @patch("litellm.completion")
    def test_without_path(self, mock_completion):
        mock_completion.return_value = _make_text_response("direct")
        engine = build_agent_engine(tools={"search": search})

        spec = [
            {
                "op": "agent_loop",
                "model": "openai/gpt-4o",
                "instructions": "Help",
                "input": "test",
                "tools": ["search"],
            }
        ]
        result = engine.apply(spec, source={}, dest={})
        assert result == "direct"
