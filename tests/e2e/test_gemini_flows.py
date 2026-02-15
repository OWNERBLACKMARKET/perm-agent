"""End-to-end tests with a real Gemini model.

These tests exercise the full perm-agent stack against the Gemini API
via litellm. They cover:
  1. Simple agent (no tools)
  2. Agent with tool calls
  3. Multi-step JSON workflow (llm → set → if)
  4. Pipeline (two agents chained)
  5. Streaming LLM
  6. Streaming agent loop with tools
  7. Structured output (JSON mode)
  8. Observability (tracer + cost tracker)
  9. Agent decorator
 10. Guardrails on real LLM output
"""

from __future__ import annotations

import json
import os

import pytest
from pydantic import BaseModel

from perm_agent import (
    Agent,
    AgentCompleteEvent,
    ConsoleTracerHook,
    CostTracker,
    GuardrailPipeline,
    MaxLengthGuardrail,
    Pipeline,
    StructuredOutput,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
    Tracer,
    agent,
    build_agent_engine,
    tool,
)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL = "gemini/gemini-2.0-flash"

pytestmark = pytest.mark.skipif(not GEMINI_API_KEY, reason="GEMINI_API_KEY not set")


# ---------------------------------------------------------------------------
# Shared tools
# ---------------------------------------------------------------------------


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A Python math expression, e.g. '2 + 2'.
    """
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        return "Error: expression contains invalid characters"
    return str(eval(expression))  # noqa: S307


@tool
def get_weather(city: str) -> str:
    """Get current weather for a city.

    Args:
        city: Name of the city.
    """
    forecasts = {
        "kyiv": "Sunny, 22C",
        "london": "Cloudy, 14C",
        "tokyo": "Rainy, 18C",
    }
    return forecasts.get(city.lower(), f"No data for {city}")


@tool
def translate(text: str, language: str) -> str:
    """Translate text to a target language (stub).

    Args:
        text: Text to translate.
        language: Target language.
    """
    return f"[{language}] {text}"


# ---------------------------------------------------------------------------
# 1. Simple agent — no tools
# ---------------------------------------------------------------------------


class TestSimpleAgent:
    def test_agent_answers_question(self) -> None:
        a = Agent(
            name="qa",
            model=MODEL,
            instructions="Answer questions in one sentence. Be factual.",
        )
        result = a.run("What is the capital of France?")
        assert isinstance(result, str)
        assert len(result) > 0
        assert "paris" in result.lower()


# ---------------------------------------------------------------------------
# 2. Agent with tool calls
# ---------------------------------------------------------------------------


class TestAgentWithTools:
    def test_calculator_tool(self) -> None:
        a = Agent(
            name="math-agent",
            model=MODEL,
            instructions=(
                "You are a math assistant. "
                "Use the calculator tool to solve math problems. "
                "Return only the final numeric answer."
            ),
            tools=[calculator],
        )
        result = a.run("What is 123 * 456?")
        assert "56088" in result

    def test_weather_tool(self) -> None:
        a = Agent(
            name="weather-agent",
            model=MODEL,
            instructions=(
                "You are a weather assistant. "
                "Use the get_weather tool to answer weather questions. "
                "Report the result exactly as received."
            ),
            tools=[get_weather],
        )
        result = a.run("What's the weather in Kyiv?")
        assert "22" in result or "sunny" in result.lower()


# ---------------------------------------------------------------------------
# 3. Multi-step JSON workflow
# ---------------------------------------------------------------------------


class TestJsonWorkflow:
    def test_llm_then_set_then_if(self) -> None:
        engine = build_agent_engine()

        workflow = [
            {
                "op": "llm",
                "model": MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Respond with exactly one word: 'yes' if 2+2=4, 'no' otherwise."
                        ),
                    }
                ],
                "path": "/answer",
            },
            {
                "op": "set",
                "path": "/checked",
                "value": True,
            },
            {
                "op": "if",
                "cond": "${@:/checked}",
                "then": [{"op": "set", "path": "/status", "value": "verified"}],
                "else": [{"op": "set", "path": "/status", "value": "failed"}],
            },
        ]

        result = engine.apply(workflow, source={}, dest={})
        assert result["answer"] is not None
        assert "yes" in result["answer"].lower()
        assert result["checked"] is True
        assert result["status"] == "verified"


# ---------------------------------------------------------------------------
# 4. Pipeline — two agents chained
# ---------------------------------------------------------------------------


class TestPipeline:
    def test_research_then_summarize(self) -> None:
        researcher = Agent(
            name="researcher",
            model=MODEL,
            instructions=("Provide 3 key facts about the given topic. Format: numbered list."),
        )
        summarizer = Agent(
            name="summarizer",
            model=MODEL,
            instructions="Summarize the research into one paragraph.",
        )

        pipe = Pipeline("research-pipeline")
        pipe.add_step(researcher, output_path="/research")
        pipe.add_step(
            summarizer,
            input_map={"input": "@:/research"},
            output_path="/summary",
        )

        result = pipe.run({"input": "Python programming language"})
        assert "research" in result
        assert "summary" in result
        assert len(result["summary"]) > 20


# ---------------------------------------------------------------------------
# 5. Streaming LLM
# ---------------------------------------------------------------------------


class TestStreamingLlm:
    def test_streaming_tokens(self) -> None:
        engine = build_agent_engine()
        tokens: list[str] = []

        spec = [
            {
                "op": "streaming_llm",
                "model": MODEL,
                "messages": [{"role": "user", "content": "Count from 1 to 5."}],
                "on_event": lambda e: tokens.append(e.token) if isinstance(e, TokenEvent) else None,
                "path": "/result",
            }
        ]

        result = engine.apply(spec, source={}, dest={})
        full_text = result["result"]

        assert len(tokens) > 0, "Should have received streaming tokens"
        assert "".join(tokens) == full_text
        assert "3" in full_text


# ---------------------------------------------------------------------------
# 6. Streaming agent loop with tools
# ---------------------------------------------------------------------------


class TestStreamingAgentLoop:
    def test_streaming_with_tool_events(self) -> None:
        events: list = []

        def search_db(query: str) -> str:
            """Search a database for information.

            Args:
                query: The search query.
            """
            return f"Found: {query} is a popular framework"

        engine = build_agent_engine(tools={"search_db": search_db})

        spec = [
            {
                "op": "streaming_agent_loop",
                "model": MODEL,
                "instructions": (
                    "Use the search_db tool to find information, then summarize the result."
                ),
                "input": "Tell me about perm-agent",
                "tools": ["search_db"],
                "on_event": lambda e: events.append(e),
                "path": "/result",
            }
        ]

        result = engine.apply(spec, source={}, dest={})
        assert result["result"]

        event_types = {type(e) for e in events}
        assert TokenEvent in event_types, "Should emit token events"
        assert AgentCompleteEvent in event_types, "Should emit complete event"

        # Tool events may or may not appear depending on model decision
        tool_calls = [e for e in events if isinstance(e, ToolCallEvent)]
        tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
        if tool_calls:
            assert len(tool_results) == len(tool_calls)


# ---------------------------------------------------------------------------
# 7. Structured output via JSON mode
# ---------------------------------------------------------------------------


class TestStructuredOutput:
    def test_parse_llm_json_response(self) -> None:
        class CityInfo(BaseModel):
            name: str
            country: str
            population_millions: float

        engine = build_agent_engine()

        spec = [
            {
                "op": "llm",
                "model": MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Return JSON about Kyiv with fields: "
                            "name (str), country (str), population_millions (float). "
                            "Return only valid JSON, no markdown."
                        ),
                    }
                ],
                "response_format": "json",
                "path": "/raw",
            }
        ]

        result = engine.apply(spec, source={}, dest={})
        raw = result["raw"]

        # Gemini may return a list or dict; normalize
        if isinstance(raw, list):
            raw = raw[0]
        data = raw if isinstance(raw, dict) else json.loads(raw)
        if isinstance(data, list):
            data = data[0]

        parser = StructuredOutput(CityInfo)
        city = parser.parse(data)

        assert city.name.lower() == "kyiv"
        assert city.country.lower() in ("ukraine", "ua")
        assert city.population_millions > 0


# ---------------------------------------------------------------------------
# 8. Observability — tracer + cost tracker
# ---------------------------------------------------------------------------


class TestObservability:
    def test_tracer_records_agent_loop_spans(self) -> None:
        tracer = Tracer(hooks=[ConsoleTracerHook()])

        engine = build_agent_engine(
            tools={"calculator": calculator},
            tracer=tracer,
        )

        spec = [
            {
                "op": "agent_loop",
                "model": MODEL,
                "instructions": "Use calculator to compute 7 * 8. Return the answer.",
                "input": "What is 7 * 8?",
                "tools": ["calculator"],
                "path": "/answer",
            }
        ]

        result = engine.apply(spec, source={}, dest={})
        assert "56" in result["answer"]

        # Tracer should have collected spans
        assert len(tracer.spans) > 0

        # At least one agent_loop span
        ops = [s.operation for s in tracer.spans]
        assert "agent_loop" in ops

        # Spans should have iteration events
        loop_span = next(s for s in tracer.spans if s.operation == "agent_loop")
        event_names = [e.name for e in loop_span.events]
        assert "agent_loop.iteration" in event_names

        # Export works
        exported = tracer.to_dict()
        assert len(exported) > 0
        assert all("span_id" in s for s in exported)

    def test_cost_tracker_with_llm_op(self) -> None:
        cost = CostTracker()
        tracer = Tracer(hooks=[cost])

        engine = build_agent_engine(tracer=tracer)

        spec = [
            {
                "op": "llm",
                "model": MODEL,
                "messages": [{"role": "user", "content": "Say hello"}],
                "path": "/greeting",
            }
        ]

        result = engine.apply(spec, source={}, dest={})
        assert result["greeting"]

        # CostTracker tracks "llm" operation spans
        assert len(cost.calls) == 1
        assert cost.calls[0]["model"] == MODEL


# ---------------------------------------------------------------------------
# 9. Agent decorator
# ---------------------------------------------------------------------------


class TestAgentDecorator:
    def test_decorated_function(self) -> None:
        @agent(model=MODEL, tools=[calculator])
        def math_helper(question: str) -> str:
            """You are a math helper. Use calculator to solve problems."""

        result = math_helper("What is 15 + 27?")
        assert "42" in result


# ---------------------------------------------------------------------------
# 10. Guardrails on real output
# ---------------------------------------------------------------------------


class TestGuardrails:
    def test_guardrail_passes_normal_output(self) -> None:
        a = Agent(
            name="safe-agent",
            model=MODEL,
            instructions="Answer in one sentence.",
        )
        result = a.run("What color is the sky?")

        pipeline = GuardrailPipeline([MaxLengthGuardrail(max_length=5000)])
        check = pipeline.check(result)
        assert check.passed


# ---------------------------------------------------------------------------
# 11. Agent with multiple tools
# ---------------------------------------------------------------------------


class TestMultiToolAgent:
    def test_agent_selects_correct_tool(self) -> None:
        a = Agent(
            name="multi-tool",
            model=MODEL,
            instructions=(
                "You have access to calculator and get_weather tools. "
                "Use the appropriate tool for each question. "
                "Return the tool result directly."
            ),
            tools=[calculator, get_weather],
        )
        result = a.run("What is the weather in Tokyo?")
        assert "18" in result or "rainy" in result.lower()


# ---------------------------------------------------------------------------
# 12. Spec export / roundtrip
# ---------------------------------------------------------------------------


class TestSpecRoundtrip:
    def test_agent_spec_is_valid_json(self) -> None:
        a = Agent(
            name="test-agent",
            model=MODEL,
            instructions="Test instructions.",
            tools=[calculator],
            max_iterations=5,
            temperature=0.3,
        )
        spec = a.to_spec()
        serialized = json.dumps(spec)
        loaded = json.loads(serialized)

        assert loaded[0]["op"] == "agent_loop"
        assert loaded[0]["model"] == MODEL
        assert loaded[0]["temperature"] == 0.3
        assert loaded[0]["max_iterations"] == 5
        assert "calculator" in loaded[0]["tools"]

        # Roundtrip
        rebuilt = Agent.from_spec(
            {
                "name": loaded[0].get("name", "test-agent"),
                "model": loaded[0]["model"],
                "instructions": loaded[0]["instructions"],
                "tools": loaded[0]["tools"],
                "max_iterations": loaded[0]["max_iterations"],
            },
            tools={"calculator": calculator},
        )
        assert rebuilt.name == "test-agent"
        assert rebuilt.model == MODEL
