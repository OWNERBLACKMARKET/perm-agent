from unittest.mock import MagicMock, patch

from perm_agent import build_agent_engine


def _make_text_response(content: str) -> MagicMock:
    response = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    choice.message.tool_calls = None
    response.choices = [choice]
    return response


def search(query: str) -> str:
    """Search the web"""
    return f"Result: {query}"


def calculate(expression: str) -> str:
    """Evaluate math"""
    return str(eval(expression))


class TestJPermOpsStillWork:
    def test_set_op(self):
        engine = build_agent_engine()
        spec = [{"op": "set", "path": "/name", "value": "Alice"}]
        result = engine.apply(spec, source={}, dest={})
        assert result == {"name": "Alice"}

    def test_foreach_op(self):
        engine = build_agent_engine()
        spec = [
            {
                "op": "foreach",
                "in": "/items",
                "as": "item",
                "do": [{"op": "set", "path": "/last", "value": "${/item}"}],
            }
        ]
        result = engine.apply(spec, source={"items": [1, 2, 3]}, dest={})
        assert result["last"] == 3

    def test_if_op(self):
        engine = build_agent_engine()
        spec = [
            {
                "op": "if",
                "cond": "${/flag}",
                "then": [{"op": "set", "path": "/r", "value": "yes"}],
                "else": [{"op": "set", "path": "/r", "value": "no"}],
            }
        ]
        assert engine.apply(spec, source={"flag": True}, dest={})["r"] == "yes"
        assert engine.apply(spec, source={"flag": False}, dest={})["r"] == "no"

    def test_shorthand_assign(self):
        engine = build_agent_engine()
        spec = {"/name": "${/user}"}
        result = engine.apply(spec, source={"user": "Bob"}, dest={})
        assert result == {"name": "Bob"}


class TestToolThenTransform:
    def test_tool_result_then_set(self):
        engine = build_agent_engine(tools={"search": search})
        spec = [
            {"op": "tool", "name": "search", "args": {"query": "python"}, "path": "/raw"},
            {"op": "set", "path": "/processed", "value": True},
        ]
        result = engine.apply(spec, source={}, dest={})
        assert result["raw"] == "Result: python"
        assert result["processed"] is True

    def test_tool_then_foreach(self):
        engine = build_agent_engine(tools={"calculate": calculate})
        spec = [
            {"op": "tool", "name": "calculate", "args": {"expression": "2+2"}, "path": "/sum"},
            {
                "op": "foreach",
                "in": "/tags",
                "as": "tag",
                "do": [{"op": "set", "path": "/last_tag", "value": "${/tag}"}],
            },
        ]
        result = engine.apply(spec, source={"tags": ["a", "b"]}, dest={})
        assert result["sum"] == "4"
        assert result["last_tag"] == "b"


class TestLlmWithJPermOps:
    @patch("litellm.completion")
    def test_llm_then_if(self, mock_completion):
        mock_completion.return_value = _make_text_response("positive")
        engine = build_agent_engine()

        spec = [
            {
                "op": "llm",
                "model": "openai/gpt-4o",
                "messages": [{"role": "user", "content": "Classify: ${/text}"}],
                "path": "/sentiment",
            },
            {
                "op": "if",
                "path": "@:/sentiment",
                "equals": "positive",
                "then": [{"op": "set", "path": "/action", "value": "celebrate"}],
                "else": [{"op": "set", "path": "/action", "value": "investigate"}],
            },
        ]
        result = engine.apply(spec, source={"text": "great day"}, dest={})
        assert result["sentiment"] == "positive"
        assert result["action"] == "celebrate"

    @patch("litellm.completion")
    def test_foreach_with_llm(self, mock_completion):
        mock_completion.side_effect = [
            _make_text_response("Summary of doc1"),
            _make_text_response("Summary of doc2"),
        ]
        engine = build_agent_engine()

        spec = [
            {"op": "set", "path": "/summaries", "value": []},
            {
                "op": "foreach",
                "in": "/docs",
                "as": "doc",
                "do": [
                    {
                        "op": "llm",
                        "model": "openai/gpt-4o",
                        "messages": [{"role": "user", "content": "Summarize: ${/doc}"}],
                        "path": "/current_summary",
                    },
                    {
                        "op": "set",
                        "path": "/summaries/-",
                        "value": "${@:/current_summary}",
                    },
                ],
            },
        ]
        result = engine.apply(spec, source={"docs": ["doc1", "doc2"]}, dest={})
        assert result["summaries"] == ["Summary of doc1", "Summary of doc2"]


class TestHandoffWorkflow:
    def test_multi_agent_pipeline(self):
        extractor = [{"/extracted": "${/raw}"}]
        formatter = [{"/formatted": "${/extracted}"}]

        engine = build_agent_engine(agent_specs={
            "extractor": extractor,
            "formatter": formatter,
        })

        spec = [
            {"op": "handoff", "to": "extractor", "input": {"raw": "${/data}"}, "path": "/step1"},
            {"op": "set", "path": "/final", "value": "${@:/step1/extracted}"},
        ]
        result = engine.apply(spec, source={"data": "hello"}, dest={})
        assert result["step1"] == {"extracted": "hello"}
        assert result["final"] == "hello"
