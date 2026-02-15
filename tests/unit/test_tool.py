import pytest


class TestToolHandler:
    def test_tool_with_dict_args(self, agent_engine):
        spec = [{"op": "tool", "name": "search", "args": {"query": "hello"}, "path": "/result"}]
        result = agent_engine.apply(spec, source={}, dest={})
        assert result == {"result": "Result for: hello"}

    def test_tool_with_template_args(self, agent_engine):
        spec = [{"op": "tool", "name": "search", "args": {"query": "${/q}"}, "path": "/result"}]
        result = agent_engine.apply(spec, source={"q": "test query"}, dest={})
        assert result == {"result": "Result for: test query"}

    def test_tool_calculate(self, agent_engine):
        spec = [
            {"op": "tool", "name": "calculate", "args": {"expression": "2 + 3"}, "path": "/answer"}
        ]
        result = agent_engine.apply(spec, source={}, dest={})
        assert result == {"answer": "5"}

    def test_tool_without_path(self, agent_engine):
        spec = [{"op": "tool", "name": "search", "args": {"query": "test"}}]
        result = agent_engine.apply(spec, source={}, dest={})
        assert result == "Result for: test"

    def test_tool_missing_raises(self, agent_engine):
        spec = [{"op": "tool", "name": "nonexistent", "args": {}}]
        with pytest.raises(KeyError, match="not registered"):
            agent_engine.apply(spec, source={}, dest={})

    def test_tool_template_name(self, agent_engine):
        spec = [{"op": "tool", "name": "${/tool_name}", "args": {"query": "test"}, "path": "/r"}]
        result = agent_engine.apply(spec, source={"tool_name": "search"}, dest={})
        assert result == {"r": "Result for: test"}
