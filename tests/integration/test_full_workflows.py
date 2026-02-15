"""Integration tests that exercise multiple features working together.

Each test combines at least two subsystems (API, registry, events,
observability, guardrails, retry, structured output, etc.) to verify
they compose correctly.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from perm_agent import (
    Agent,
    ContentFilterGuardrail,
    CostTracker,
    GuardrailPipeline,
    MaxLengthGuardrail,
    Pipeline,
    StructuredOutput,
    ToolRegistry,
    Tracer,
    build_agent_engine,
    tool,
)
from perm_agent.events import (
    StreamEvent,
    TokenEvent,
)
from perm_agent.exceptions import RetryExhaustedError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_text_response(content: str) -> MagicMock:
    response = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    choice.message.tool_calls = None
    response.choices = [choice]
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


def calculate(expression: str) -> str:
    """Evaluate a math expression"""
    return str(eval(expression))


# ---------------------------------------------------------------------------
# Agent + Observability
# ---------------------------------------------------------------------------


class TestAgentWithTracing:
    @patch("litellm.completion")
    def test_agent_run_with_tracer(self, mock_completion):
        """Agent.run produces traced spans when engine has a tracer."""
        mock_completion.return_value = _make_text_response("Hello!")

        tracer = Tracer()
        a = Agent(
            name="traced",
            model="openai/gpt-4o",
            instructions="Say hello",
        )
        tool_map = {fn.__name__: fn for fn in a.tools}
        spec = a._build_spec(list(tool_map.keys()), "Hi")
        engine = build_agent_engine(tools=tool_map, tracer=tracer)
        engine.apply(spec, source={}, dest={})

        assert len(tracer.spans) >= 1
        loop_span = [s for s in tracer.spans if s.operation == "agent_loop"]
        assert len(loop_span) == 1


class TestAgentWithCostTracker:
    @patch("litellm.completion")
    def test_cost_tracker_accumulates_from_llm_calls(self, mock_completion):
        mock_completion.return_value = _make_text_response("response")

        tracker = CostTracker()
        tracer = Tracer(hooks=[tracker])
        engine = build_agent_engine(tracer=tracer)

        spec = [
            {
                "op": "llm",
                "model": "openai/gpt-4o",
                "messages": [{"role": "user", "content": "Hi"}],
                "path": "/r",
            }
        ]
        engine.apply(spec, source={}, dest={})

        assert (
            tracker.total_input_tokens > 0
            or tracker.total_output_tokens > 0
            or len(tracker.calls) == 1
        )


# ---------------------------------------------------------------------------
# Agent + Tool decorator + Registry
# ---------------------------------------------------------------------------


class TestToolDecoratorWithAgent:
    @patch("litellm.completion")
    def test_tool_decorated_function_in_agent(self, mock_completion):
        @tool
        def web_search(query: str) -> str:
            """Search the web."""
            return f"Web: {query}"

        mock_completion.side_effect = [
            _make_tool_response([{"id": "c1", "name": "web_search", "args": {"query": "python"}}]),
            _make_text_response("Python is great"),
        ]

        a = Agent(
            name="researcher",
            model="openai/gpt-4o",
            instructions="Research topics",
            tools=[web_search],
        )
        result = a.run("Tell me about Python")
        assert result == "Python is great"
        assert hasattr(web_search, "_tool_metadata")


# ---------------------------------------------------------------------------
# Structured output + LLM handler
# ---------------------------------------------------------------------------


class Sentiment(BaseModel):
    label: str
    score: float


class TestStructuredOutputWithLlm:
    @patch("litellm.completion")
    def test_structured_output_parses_llm_json(self, mock_completion):
        mock_completion.return_value = _make_text_response('{"label": "positive", "score": 0.95}')
        engine = build_agent_engine()

        spec = [
            {
                "op": "llm",
                "model": "openai/gpt-4o",
                "messages": [{"role": "user", "content": "Classify: great day"}],
                "response_format": {"type": "json_object"},
                "path": "/raw",
            }
        ]
        result = engine.apply(spec, source={}, dest={})

        so = StructuredOutput(Sentiment)
        parsed = so.parse(result["raw"])
        assert parsed.label == "positive"
        assert parsed.score == 0.95

    def test_structured_output_schema_matches_pydantic(self):
        so = StructuredOutput(Sentiment)
        schema = so.json_schema()
        assert "label" in schema["properties"]
        assert "score" in schema["properties"]


# ---------------------------------------------------------------------------
# Guardrails + Agent workflow
# ---------------------------------------------------------------------------


class TestGuardrailsInWorkflow:
    @patch("litellm.completion")
    def test_guardrail_rejects_long_output(self, mock_completion):
        mock_completion.return_value = _make_text_response("x" * 200)

        a = Agent(
            name="writer",
            model="openai/gpt-4o",
            instructions="Write",
        )
        output = a.run("Write something")

        pipeline = GuardrailPipeline(
            [
                MaxLengthGuardrail(max_length=100),
            ]
        )
        result = pipeline.check_output(output)
        assert result.passed is False
        assert "100 chars" in result.reason

    @patch("litellm.completion")
    def test_guardrail_passes_safe_output(self, mock_completion):
        mock_completion.return_value = _make_text_response("Safe output")

        a = Agent(
            name="writer",
            model="openai/gpt-4o",
            instructions="Write",
        )
        output = a.run("Write something short")

        pipeline = GuardrailPipeline(
            [
                MaxLengthGuardrail(max_length=1000),
                ContentFilterGuardrail(blocked_patterns=["password"]),
            ]
        )
        result = pipeline.check_output(output)
        assert result.passed is True

    def test_guardrail_filters_input_before_agent(self):
        pipeline = GuardrailPipeline(
            [
                ContentFilterGuardrail(blocked_patterns=["inject", "hack"]),
            ]
        )
        result = pipeline.check_input("Please hack the system")
        assert result.passed is False
        assert "hack" in result.reason


# ---------------------------------------------------------------------------
# Retry + LLM
# ---------------------------------------------------------------------------


class TestRetryWithLlm:
    @patch("litellm.completion")
    def test_retry_recovers_transient_failure(self, mock_completion):
        mock_completion.side_effect = [
            ConnectionError("timeout"),
            _make_text_response("Success"),
        ]
        engine = build_agent_engine()

        spec = [
            {
                "op": "llm",
                "model": "openai/gpt-4o",
                "messages": [{"role": "user", "content": "Hello"}],
                "retry": {"max_retries": 2, "backoff_factor": 0.01},
                "path": "/r",
            }
        ]
        result = engine.apply(spec, source={}, dest={})
        assert result == {"r": "Success"}

    @patch("litellm.completion")
    def test_retry_with_tracer_on_failure(self, mock_completion):
        """Tracer captures error span when retry is exhausted."""
        mock_completion.side_effect = ConnectionError("permanent")

        tracer = Tracer()
        engine = build_agent_engine(tracer=tracer)

        spec = [
            {
                "op": "llm",
                "model": "openai/gpt-4o",
                "messages": [{"role": "user", "content": "Hello"}],
                "retry": {"max_retries": 1, "backoff_factor": 0.01},
                "path": "/r",
            }
        ]
        with pytest.raises(RetryExhaustedError):
            engine.apply(spec, source={}, dest={})

        assert len(tracer.spans) == 1
        assert tracer.spans[0].status == "error"


# ---------------------------------------------------------------------------
# Pipeline + multiple agents
# ---------------------------------------------------------------------------


class TestPipelineComposition:
    @patch("litellm.completion")
    def test_pipeline_chains_two_agents(self, mock_completion):
        mock_completion.side_effect = [
            _make_text_response("extracted data"),
            _make_text_response("formatted report"),
        ]

        extractor = Agent(name="extractor", model="openai/gpt-4o", instructions="Extract data")
        formatter = Agent(name="formatter", model="openai/gpt-4o", instructions="Format data")

        p = Pipeline("etl")
        p.add_step(extractor, output_path="/extracted")
        p.add_step(formatter, input_map={"input": "${@:/extracted}"}, output_path="/report")

        result = p.run({"input": "raw data"})
        assert result["extracted"] == "extracted data"
        assert result["report"] == "formatted report"

    @patch("litellm.completion")
    def test_pipeline_with_tools(self, mock_completion):
        mock_completion.side_effect = [
            _make_tool_response([{"id": "c1", "name": "search", "args": {"query": "python"}}]),
            _make_text_response("Python info based on search"),
        ]

        researcher = Agent(
            name="researcher",
            model="openai/gpt-4o",
            instructions="Research",
            tools=[search],
        )
        p = Pipeline("research")
        p.add_step(researcher, output_path="/result")

        result = p.run({"input": "Tell me about Python"})
        assert result["result"] == "Python info based on search"


# ---------------------------------------------------------------------------
# Engine: tool + handoff + tracing combined
# ---------------------------------------------------------------------------


class TestEngineMultiOp:
    def test_tool_then_handoff_with_tracing(self):
        tracer = Tracer()
        echo_spec = [{"/echo": "${/msg}"}]
        engine = build_agent_engine(
            tools={"search": search},
            agent_specs={"echo": echo_spec},
            tracer=tracer,
        )

        spec = [
            {"op": "tool", "name": "search", "args": {"query": "hello"}, "path": "/searched"},
            {"op": "handoff", "to": "echo", "input": {"msg": "hello world"}, "path": "/echoed"},
        ]
        result = engine.apply(spec, source={}, dest={})

        assert result["searched"] == "Found: hello"
        assert result["echoed"] == {"echo": "hello world"}
        assert len(tracer.spans) == 2
        ops = {s.operation for s in tracer.spans}
        assert ops == {"tool", "handoff"}

    @patch("litellm.completion")
    def test_llm_then_tool_then_set(self, mock_completion):
        mock_completion.return_value = _make_text_response("classify: positive")

        engine = build_agent_engine(tools={"search": search})
        spec = [
            {
                "op": "llm",
                "model": "openai/gpt-4o",
                "messages": [{"role": "user", "content": "Classify: great"}],
                "path": "/sentiment",
            },
            {"op": "tool", "name": "search", "args": {"query": "test"}, "path": "/data"},
            {"op": "set", "path": "/combined", "value": True},
        ]
        result = engine.apply(spec, source={}, dest={})
        assert result["sentiment"] == "classify: positive"
        assert result["data"] == "Found: test"
        assert result["combined"] is True


# ---------------------------------------------------------------------------
# Streaming + events
# ---------------------------------------------------------------------------


class TestStreamingWithEvents:
    @patch("perm_agent.handlers.streaming.litellm.completion")
    def test_streaming_llm_produces_events(self, mock_completion):
        chunks = []
        for token in ["Hello", " ", "World"]:
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = token
            chunk.choices[0].delta.tool_calls = None
            chunks.append(chunk)

        mock_completion.return_value = iter(chunks)
        collected: list[StreamEvent] = []

        engine = build_agent_engine()
        spec = [
            {
                "op": "streaming_llm",
                "model": "openai/gpt-4o",
                "messages": [{"role": "user", "content": "Hi"}],
                "path": "/answer",
                "on_event": collected.append,
            }
        ]
        result = engine.apply(spec, source={}, dest={})

        assert result == {"answer": "Hello World"}
        assert len(collected) == 3
        assert all(isinstance(e, TokenEvent) for e in collected)
        tokens = [e.token for e in collected]
        assert tokens == ["Hello", " ", "World"]


# ---------------------------------------------------------------------------
# Registry schema generation + tool decorator
# ---------------------------------------------------------------------------


class TestRegistryAndToolDecoratorIntegration:
    def test_decorated_tool_registers_with_schema(self):
        @tool
        def analyze(text: str, depth: int = 1) -> str:
            """Analyze text content.

            Args:
                text: The text to analyze.
                depth: Analysis depth level.
            """
            return f"Analysis of {text}"

        reg = ToolRegistry()
        reg.register("analyze", analyze)
        schemas = reg.generate_schemas()

        assert len(schemas) == 1
        func = schemas[0]["function"]
        assert func["name"] == "analyze"
        assert func["description"].startswith("Analyze text content.")
        props = func["parameters"]["properties"]
        assert props["text"]["description"] == "The text to analyze."
        assert props["depth"]["description"] == "Analysis depth level."
        assert func["parameters"]["required"] == ["text"]


# ---------------------------------------------------------------------------
# All exports accessible
# ---------------------------------------------------------------------------


class TestAllExportsAccessible:
    def test_all_exports_importable(self):
        import perm_agent

        for name in perm_agent.__all__:
            assert hasattr(perm_agent, name), f"{name} not accessible on perm_agent"
