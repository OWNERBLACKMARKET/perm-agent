import json
from unittest.mock import MagicMock, patch

from perm_agent.api import Agent, Pipeline, agent


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
    return response


def search(query: str) -> str:
    """Search for information"""
    return f"Found: {query}"


def calculate(expression: str) -> str:
    """Evaluate a math expression"""
    return str(eval(expression))


class TestAgentRunSimple:
    @patch("litellm.completion")
    def test_agent_run_returns_text(self, mock_completion):
        mock_completion.return_value = _make_text_response("The answer is 42")

        a = Agent(
            name="helper",
            model="openai/gpt-4o",
            instructions="You are helpful",
        )
        result = a.run("What is the meaning of life?")

        assert result == "The answer is 42"
        mock_completion.assert_called_once()
        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["model"] == "openai/gpt-4o"
        msgs = call_kwargs["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "You are helpful"
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "What is the meaning of life?"


class TestAgentWithTools:
    @patch("litellm.completion")
    def test_agent_uses_tools(self, mock_completion):
        mock_completion.side_effect = [
            _make_tool_response([{"id": "call_1", "name": "search", "args": {"query": "python"}}]),
            _make_text_response("Python is a programming language"),
        ]

        a = Agent(
            name="researcher",
            model="openai/gpt-4o",
            instructions="Use search to answer questions",
            tools=[search],
        )
        result = a.run("What is Python?")

        assert result == "Python is a programming language"
        assert mock_completion.call_count == 2


class TestAgentDecorator:
    @patch("litellm.completion")
    def test_decorator_creates_callable(self, mock_completion):
        mock_completion.return_value = _make_text_response("Quantum computing is cool")

        @agent(model="openai/gpt-4o", tools=[search])
        def researcher(question: str) -> str:
            """You are a research assistant."""
            ...

        result = researcher("What is quantum computing?")

        assert result == "Quantum computing is cool"
        mock_completion.assert_called_once()
        call_kwargs = mock_completion.call_args[1]
        msgs = call_kwargs["messages"]
        assert msgs[0]["content"] == "You are a research assistant."

    @patch("litellm.completion")
    def test_decorator_preserves_agent_reference(self, mock_completion):
        @agent(model="openai/gpt-4o")
        def my_agent(q: str) -> str:
            """Instructions here."""
            ...

        assert hasattr(my_agent, "_agent")
        assert my_agent._agent.name == "my_agent"
        assert my_agent._agent.model == "openai/gpt-4o"
        assert my_agent._agent.instructions == "Instructions here."


class TestAgentToSpecRoundtrip:
    def test_to_spec_produces_valid_spec(self):
        a = Agent(
            name="bot",
            model="openai/gpt-4o",
            instructions="Be helpful",
            tools=[search],
            max_iterations=5,
            memory_limit=15,
        )
        spec = a.to_spec()

        assert len(spec) == 1
        step = spec[0]
        assert step["op"] == "agent_loop"
        assert step["model"] == "openai/gpt-4o"
        assert step["instructions"] == "Be helpful"
        assert step["input"] == "${/input}"
        assert step["tools"] == ["search"]
        assert step["max_iterations"] == 5
        assert step["memory_limit"] == 15
        assert step["path"] == "/result"

    @patch("litellm.completion")
    def test_roundtrip_from_spec(self, mock_completion):
        mock_completion.return_value = _make_text_response("roundtrip works")

        original = Agent(
            name="bot",
            model="openai/gpt-4o",
            instructions="Be helpful",
            tools=[search],
            max_iterations=5,
            memory_limit=15,
        )

        exported = {
            "name": original.name,
            "model": original.model,
            "instructions": original.instructions,
            "tools": [fn.__name__ for fn in original.tools],
            "max_iterations": original.max_iterations,
            "memory_limit": original.memory_limit,
        }

        restored = Agent.from_spec(exported, tools={"search": search})

        assert restored.name == "bot"
        assert restored.model == "openai/gpt-4o"
        assert restored.instructions == "Be helpful"
        assert restored.max_iterations == 5
        assert restored.memory_limit == 15
        assert len(restored.tools) == 1

        result = restored.run("test")
        assert result == "roundtrip works"


class TestAgentFromSpec:
    def test_from_spec_basic(self):
        spec = {
            "name": "tester",
            "model": "openai/gpt-4o",
            "instructions": "Test instructions",
            "tools": ["search"],
            "max_iterations": 3,
            "memory_limit": 10,
        }
        a = Agent.from_spec(spec, tools={"search": search})

        assert a.name == "tester"
        assert a.model == "openai/gpt-4o"
        assert a.instructions == "Test instructions"
        assert a.max_iterations == 3
        assert a.memory_limit == 10
        assert len(a.tools) == 1

    def test_from_spec_missing_tools_graceful(self):
        spec = {
            "name": "minimal",
            "model": "openai/gpt-4o",
            "instructions": "Minimal",
        }
        a = Agent.from_spec(spec)

        assert a.name == "minimal"
        assert a.tools == []

    def test_from_spec_with_temperature(self):
        spec = {
            "name": "creative",
            "model": "openai/gpt-4o",
            "instructions": "Be creative",
            "temperature": 0.9,
        }
        a = Agent.from_spec(spec)
        assert a.temperature == 0.9


class TestPipelineBasic:
    @patch("litellm.completion")
    def test_single_step_pipeline(self, mock_completion):
        mock_completion.return_value = _make_text_response("pipeline result")

        a = Agent(
            name="step1",
            model="openai/gpt-4o",
            instructions="Process input",
        )

        p = Pipeline("test-pipeline")
        p.add_step(a, output_path="/result")

        result = p.run({"input": "hello"})
        assert result["result"] == "pipeline result"


class TestPipelineMultiStep:
    @patch("litellm.completion")
    def test_two_step_pipeline(self, mock_completion):
        mock_completion.side_effect = [
            _make_text_response("intermediate"),
            _make_text_response("final answer"),
        ]

        step1 = Agent(name="analyzer", model="openai/gpt-4o", instructions="Analyze")
        step2 = Agent(name="writer", model="openai/gpt-4o", instructions="Write")

        p = Pipeline("multi-step")
        p.add_step(step1, output_path="/analysis")
        p.add_step(step2, input_map={"input": "${@:/analysis}"}, output_path="/output")

        result = p.run({"input": "raw data"})
        assert result["analysis"] == "intermediate"
        assert result["output"] == "final answer"
        assert mock_completion.call_count == 2


class TestPipelineToSpec:
    def test_pipeline_exports_spec(self):
        a = Agent(
            name="bot",
            model="openai/gpt-4o",
            instructions="Help",
            tools=[search],
        )

        p = Pipeline("export-test")
        p.add_step(a, output_path="/result")

        spec = p.to_spec()
        assert len(spec) == 1
        assert spec[0]["op"] == "agent_loop"
        assert spec[0]["path"] == "/result"
        assert spec[0]["model"] == "openai/gpt-4o"


class TestAgentCustomParams:
    @patch("litellm.completion")
    def test_custom_max_iterations(self, mock_completion):
        mock_completion.return_value = _make_tool_response(
            [{"id": "call_x", "name": "search", "args": {"query": "loop"}}]
        )

        a = Agent(
            name="looper",
            model="openai/gpt-4o",
            instructions="Keep going",
            tools=[search],
            max_iterations=2,
        )
        a.run("loop forever")

        assert mock_completion.call_count == 2

    @patch("litellm.completion")
    def test_custom_memory_limit(self, mock_completion):
        mock_completion.return_value = _make_text_response("ok")

        a = Agent(
            name="limited",
            model="openai/gpt-4o",
            instructions="Help",
            memory_limit=5,
        )
        a.run("test")

        call_kwargs = mock_completion.call_args[1]
        # With memory_limit=5, messages are sliced to last 5
        assert len(call_kwargs["messages"]) <= 5

    def test_temperature_in_spec(self):
        a = Agent(
            name="creative",
            model="openai/gpt-4o",
            instructions="Be creative",
            temperature=0.9,
        )
        spec = a.to_spec()
        assert spec[0].get("temperature") == 0.9

    def test_no_temperature_omitted_from_spec(self):
        a = Agent(
            name="default",
            model="openai/gpt-4o",
            instructions="Help",
        )
        spec = a.to_spec()
        assert "temperature" not in spec[0]


class TestAgentRunWithContext:
    @patch("litellm.completion")
    def test_context_passed_as_source(self, mock_completion):
        mock_completion.return_value = _make_text_response("Hello Alice")

        a = Agent(
            name="greeter",
            model="openai/gpt-4o",
            instructions="Greet the user",
        )
        result = a.run("Say hello", user_name="Alice")

        assert result == "Hello Alice"
        mock_completion.assert_called_once()
