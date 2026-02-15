from unittest.mock import MagicMock, patch

import pytest

from perm_agent import build_agent_engine


def _make_response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


class TestLlmHandler:
    @patch("perm_agent.handlers.llm.litellm")
    def test_basic_call(self, mock_litellm):
        mock_litellm.completion.return_value = _make_response("Hello!")
        engine = build_agent_engine()

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
        mock_litellm.completion.assert_called_once()
        call_kwargs = mock_litellm.completion.call_args[1]
        assert call_kwargs["model"] == "openai/gpt-4o"

    @patch("perm_agent.handlers.llm.litellm")
    def test_template_in_messages(self, mock_litellm):
        mock_litellm.completion.return_value = _make_response("Paris")
        engine = build_agent_engine()

        spec = [
            {
                "op": "llm",
                "model": "openai/gpt-4o",
                "messages": [{"role": "user", "content": "Capital of ${/country}?"}],
                "path": "/answer",
            }
        ]
        result = engine.apply(spec, source={"country": "France"}, dest={})

        assert result == {"answer": "Paris"}
        call_msgs = mock_litellm.completion.call_args[1]["messages"]
        assert call_msgs[0]["content"] == "Capital of France?"

    @patch("perm_agent.handlers.llm.litellm")
    def test_json_response_format(self, mock_litellm):
        mock_litellm.completion.return_value = _make_response('{"name": "Alice", "age": 30}')
        engine = build_agent_engine()

        spec = [
            {
                "op": "llm",
                "model": "openai/gpt-4o",
                "messages": [{"role": "user", "content": "Extract info"}],
                "response_format": {"name": "str", "age": "int"},
                "path": "/parsed",
            }
        ]
        result = engine.apply(spec, source={}, dest={})

        assert result == {"parsed": {"name": "Alice", "age": 30}}
        call_kwargs = mock_litellm.completion.call_args[1]
        assert call_kwargs["response_format"] == {"type": "json_object"}

    @patch("perm_agent.handlers.llm.litellm")
    def test_custom_temperature(self, mock_litellm):
        mock_litellm.completion.return_value = _make_response("ok")
        engine = build_agent_engine()

        spec = [
            {
                "op": "llm",
                "model": "openai/gpt-4o",
                "messages": [{"role": "user", "content": "test"}],
                "temperature": 0.2,
                "path": "/r",
            }
        ]
        engine.apply(spec, source={}, dest={})

        call_kwargs = mock_litellm.completion.call_args[1]
        assert call_kwargs["temperature"] == 0.2

    @patch("perm_agent.handlers.llm.litellm")
    def test_without_path(self, mock_litellm):
        mock_litellm.completion.return_value = _make_response("direct")
        engine = build_agent_engine()

        spec = [
            {
                "op": "llm",
                "model": "openai/gpt-4o",
                "messages": [{"role": "user", "content": "test"}],
            }
        ]
        result = engine.apply(spec, source={}, dest={})
        assert result == "direct"
