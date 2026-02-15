import pytest

from perm_agent import build_agent_engine
from perm_agent.exceptions import HandoffError


class TestHandoffHandler:
    def test_basic_handoff(self):
        summarizer_spec = [{"/summary": "${/text}"}]
        engine = build_agent_engine(agent_specs={"summarizer": summarizer_spec})

        spec = [
            {
                "op": "handoff",
                "to": "summarizer",
                "input": {"text": "hello world"},
                "path": "/result",
            }
        ]
        result = engine.apply(spec, source={}, dest={})
        assert result == {"result": {"summary": "hello world"}}

    def test_handoff_with_template_input(self):
        echo_spec = [{"/echo": "${/msg}"}]
        engine = build_agent_engine(agent_specs={"echo": echo_spec})

        spec = [{"op": "handoff", "to": "echo", "input": {"msg": "${/data}"}, "path": "/out"}]
        result = engine.apply(spec, source={"data": "test"}, dest={})
        assert result == {"out": {"echo": "test"}}

    def test_handoff_missing_spec_raises(self):
        engine = build_agent_engine()
        spec = [{"op": "handoff", "to": "nonexistent", "input": {}}]
        with pytest.raises(HandoffError, match="not found"):
            engine.apply(spec, source={}, dest={})

    def test_handoff_template_target(self):
        upper_spec = [{"/upper": "${/val}"}]
        engine = build_agent_engine(agent_specs={"upper": upper_spec})

        spec = [{"op": "handoff", "to": "${/agent}", "input": {"val": "test"}, "path": "/r"}]
        result = engine.apply(spec, source={"agent": "upper"}, dest={})
        assert result == {"r": {"upper": "test"}}

    def test_handoff_without_path(self):
        copy_spec = [{"/copy": "${/x}"}]
        engine = build_agent_engine(agent_specs={"copy": copy_spec})

        spec = [{"op": "handoff", "to": "copy", "input": {"x": "val"}}]
        result = engine.apply(spec, source={}, dest={})
        assert result == {"copy": "val"}

    def test_chained_handoff(self):
        step1_spec = [{"/step1": "done"}]
        step2_spec = [{"/step2": "${/step1}"}]

        engine = build_agent_engine(
            agent_specs={
                "step1": step1_spec,
                "step2": step2_spec,
            }
        )

        spec = [
            {"op": "handoff", "to": "step1", "input": {}, "path": "/intermediate"},
            {
                "op": "handoff",
                "to": "step2",
                "input": {"step1": "@:/intermediate/step1"},
                "path": "/final",
            },
        ]

        result = engine.apply(spec, source={}, dest={})
        assert result["intermediate"] == {"step1": "done"}
