import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from perm_agent import build_agent_engine
from perm_agent.exceptions import (
    GuardrailError,
    HandoffError,
    MaxIterationsError,
    PermAgentError,
    RetryExhaustedError,
    ToolExecutionError,
)
from perm_agent.guardrails import (
    ContentFilterGuardrail,
    GuardrailPipeline,
    MaxLengthGuardrail,
)
from perm_agent.handlers.async_agent_loop import AsyncAgentLoopHandler
from perm_agent.handlers.async_handoff import AsyncHandoffHandler
from perm_agent.registry import ToolRegistry
from perm_agent.retry import RetryConfig, with_retry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _make_assistant_tool_response(tool_calls: list[dict], content: str) -> MagicMock:
    """Tool response where the assistant message also has text content."""
    response = MagicMock()
    choice = MagicMock()
    choice.message.content = content
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
        "content": content,
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


def failing_tool(query: str) -> str:
    """A tool that always fails"""
    raise ValueError("Something went wrong")


# ---------------------------------------------------------------------------
# Tool error handling tests
# ---------------------------------------------------------------------------


class TestToolErrorHandling:
    @patch("litellm.completion")
    def test_tool_error_sent_back_to_llm(self, mock_completion):
        """When a tool raises, the error message is sent back as a tool result."""
        mock_completion.side_effect = [
            _make_tool_response(
                [
                    {"id": "call_1", "name": "failing_tool", "args": {"query": "test"}},
                ]
            ),
            _make_text_response("I encountered an error"),
        ]
        engine = build_agent_engine(
            tools={
                "search": search,
                "failing_tool": failing_tool,
            }
        )

        spec = [
            {
                "op": "agent_loop",
                "model": "openai/gpt-4o",
                "instructions": "Help",
                "input": "test",
                "tools": ["search", "failing_tool"],
                "path": "/result",
            }
        ]
        result = engine.apply(spec, source={}, dest={})

        assert result == {"result": "I encountered an error"}
        assert mock_completion.call_count == 2

        # Verify the error was sent back as a tool message
        second_call_messages = mock_completion.call_args_list[1][1]["messages"]
        tool_msg = [m for m in second_call_messages if m.get("role") == "tool"]
        assert len(tool_msg) == 1
        assert "Error executing tool 'failing_tool'" in tool_msg[0]["content"]
        assert "ValueError" in tool_msg[0]["content"]
        assert "Something went wrong" in tool_msg[0]["content"]

    @patch("litellm.completion")
    def test_tool_error_does_not_crash_loop(self, mock_completion):
        """The agent loop continues after a tool error, doesn't raise."""
        mock_completion.side_effect = [
            _make_tool_response(
                [
                    {"id": "call_1", "name": "failing_tool", "args": {"query": "x"}},
                ]
            ),
            _make_text_response("Recovered"),
        ]
        engine = build_agent_engine(
            tools={
                "search": search,
                "failing_tool": failing_tool,
            }
        )

        spec = [
            {
                "op": "agent_loop",
                "model": "openai/gpt-4o",
                "instructions": "Help",
                "input": "test",
                "tools": ["search", "failing_tool"],
                "path": "/result",
            }
        ]
        # Should not raise
        result = engine.apply(spec, source={}, dest={})
        assert result == {"result": "Recovered"}


# ---------------------------------------------------------------------------
# Max iterations fix tests
# ---------------------------------------------------------------------------


class TestMaxIterationsFix:
    @patch("litellm.completion")
    def test_max_iterations_returns_last_assistant_message(self, mock_completion):
        """On max iterations, return last assistant content, not last tool result."""
        mock_completion.return_value = _make_assistant_tool_response(
            [{"id": "call_x", "name": "search", "args": {"query": "loop"}}],
            content="I'm searching for more info",
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

        # Should return the assistant's content, not the tool result
        assert result == {"result": "I'm searching for more info"}

    @patch("litellm.completion")
    def test_max_iterations_no_assistant_content_returns_empty(self, mock_completion):
        """When no assistant message has content, return empty string."""
        mock_completion.return_value = _make_tool_response(
            [{"id": "call_x", "name": "search", "args": {"query": "loop"}}],
        )
        engine = build_agent_engine(tools={"search": search})

        spec = [
            {
                "op": "agent_loop",
                "model": "openai/gpt-4o",
                "instructions": "Keep searching",
                "input": "Loop forever",
                "tools": ["search"],
                "max_iterations": 2,
                "path": "/result",
            }
        ]
        result = engine.apply(spec, source={}, dest={})
        assert result == {"result": ""}


# ---------------------------------------------------------------------------
# Retry tests
# ---------------------------------------------------------------------------


class TestRetryConfig:
    def test_retry_config_basic(self):
        config = RetryConfig(max_retries=5, backoff_factor=0.5, max_backoff=10.0)
        assert config.max_retries == 5
        assert config.backoff_factor == 0.5
        assert config.max_backoff == 10.0

    def test_retry_config_defaults(self):
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.backoff_factor == 1.0
        assert config.max_backoff == 30.0

    def test_delay_for_attempt(self):
        config = RetryConfig(backoff_factor=1.0, max_backoff=10.0)
        assert config.delay_for_attempt(0) == 1.0  # 1 * 2^0
        assert config.delay_for_attempt(1) == 2.0  # 1 * 2^1
        assert config.delay_for_attempt(2) == 4.0  # 1 * 2^2
        assert config.delay_for_attempt(3) == 8.0  # 1 * 2^3
        assert config.delay_for_attempt(4) == 10.0  # capped at max_backoff


class TestWithRetry:
    def test_retry_succeeds_on_second_attempt(self):
        call_count = 0

        def flaky() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("transient")
            return "ok"

        config = RetryConfig(max_retries=3, backoff_factor=0.01)
        result = with_retry(flaky, config)
        assert result == "ok"
        assert call_count == 2

    def test_retry_exhausted_raises(self):
        def always_fails() -> str:
            raise ConnectionError("permanent")

        config = RetryConfig(max_retries=2, backoff_factor=0.01)
        with pytest.raises(RetryExhaustedError) as exc_info:
            with_retry(always_fails, config)

        assert exc_info.value.attempts == 3  # initial + 2 retries
        assert isinstance(exc_info.value.last_error, ConnectionError)

    def test_retry_only_catches_specified_exceptions(self):
        def type_error_fn() -> str:
            raise TypeError("wrong type")

        config = RetryConfig(
            max_retries=3,
            retry_on=(ConnectionError,),
            backoff_factor=0.01,
        )
        with pytest.raises(TypeError):
            with_retry(type_error_fn, config)

    def test_retry_backoff_timing(self):
        call_count = 0

        def always_fails() -> str:
            nonlocal call_count
            call_count += 1
            raise ConnectionError("fail")

        config = RetryConfig(max_retries=2, backoff_factor=0.05, max_backoff=1.0)

        start = time.monotonic()
        with pytest.raises(RetryExhaustedError):
            with_retry(always_fails, config)
        elapsed = time.monotonic() - start

        # backoff: 0.05 * 2^0 + 0.05 * 2^1 = 0.05 + 0.1 = 0.15s minimum
        assert elapsed >= 0.10  # generous lower bound
        assert call_count == 3

    def test_retry_no_retries_succeeds(self):
        config = RetryConfig(max_retries=0)
        result = with_retry(lambda: 42, config)
        assert result == 42

    def test_retry_no_retries_fails(self):
        def fails() -> None:
            raise RuntimeError("boom")

        config = RetryConfig(max_retries=0)
        with pytest.raises(RetryExhaustedError):
            with_retry(fails, config)


# ---------------------------------------------------------------------------
# Guardrails tests
# ---------------------------------------------------------------------------


class TestMaxLengthGuardrail:
    def test_guardrail_max_length_pass(self):
        g = MaxLengthGuardrail(max_length=100)
        result = g.check("short text")
        assert result.passed is True
        assert result.reason is None

    def test_guardrail_max_length_fail(self):
        g = MaxLengthGuardrail(max_length=5)
        result = g.check("this is too long")
        assert result.passed is False
        assert "5 chars" in result.reason

    def test_guardrail_max_length_exact_boundary(self):
        g = MaxLengthGuardrail(max_length=5)
        assert g.check("12345").passed is True
        assert g.check("123456").passed is False


class TestContentFilterGuardrail:
    def test_guardrail_content_filter_pass(self):
        g = ContentFilterGuardrail(blocked_patterns=["password", "secret"])
        result = g.check("This is safe content")
        assert result.passed is True

    def test_guardrail_content_filter_fail(self):
        g = ContentFilterGuardrail(blocked_patterns=["password", "secret"])
        result = g.check("The password is 12345")
        assert result.passed is False
        assert "password" in result.reason

    def test_guardrail_content_filter_case_insensitive(self):
        g = ContentFilterGuardrail(blocked_patterns=["secret"])
        result = g.check("This is a SECRET value")
        assert result.passed is False

    def test_guardrail_content_filter_regex(self):
        g = ContentFilterGuardrail(blocked_patterns=[r"SSN:\s*\d{3}-\d{2}-\d{4}"])
        assert g.check("SSN: 123-45-6789").passed is False
        assert g.check("No SSN here").passed is True


class TestGuardrailPipeline:
    def test_guardrail_pipeline_all_pass(self):
        pipeline = GuardrailPipeline(
            [
                MaxLengthGuardrail(max_length=1000),
                ContentFilterGuardrail(blocked_patterns=["banned"]),
            ]
        )
        result = pipeline.check("safe content")
        assert result.passed is True

    def test_guardrail_pipeline_one_fails(self):
        pipeline = GuardrailPipeline(
            [
                MaxLengthGuardrail(max_length=1000),
                ContentFilterGuardrail(blocked_patterns=["banned"]),
            ]
        )
        result = pipeline.check("this has banned word")
        assert result.passed is False
        assert "banned" in result.reason

    def test_guardrail_pipeline_first_fails_short_circuits(self):
        pipeline = GuardrailPipeline(
            [
                MaxLengthGuardrail(max_length=5),
                ContentFilterGuardrail(blocked_patterns=["banned"]),
            ]
        )
        result = pipeline.check("this is too long and also banned")
        assert result.passed is False
        assert "5 chars" in result.reason  # length check fails first

    def test_guardrail_pipeline_check_input_alias(self):
        pipeline = GuardrailPipeline([MaxLengthGuardrail(max_length=10)])
        assert pipeline.check_input("short").passed is True
        assert pipeline.check_input("this is way too long").passed is False

    def test_guardrail_pipeline_check_output_alias(self):
        pipeline = GuardrailPipeline([MaxLengthGuardrail(max_length=10)])
        assert pipeline.check_output("short").passed is True
        assert pipeline.check_output("this is way too long").passed is False

    def test_guardrail_pipeline_empty(self):
        pipeline = GuardrailPipeline([])
        assert pipeline.check("anything").passed is True


# ---------------------------------------------------------------------------
# Exception hierarchy tests
# ---------------------------------------------------------------------------


class TestCustomExceptionsHierarchy:
    def test_all_inherit_from_perm_agent_error(self):
        assert issubclass(MaxIterationsError, PermAgentError)
        assert issubclass(ToolExecutionError, PermAgentError)
        assert issubclass(GuardrailError, PermAgentError)
        assert issubclass(RetryExhaustedError, PermAgentError)
        assert issubclass(HandoffError, PermAgentError)

    def test_perm_agent_error_inherits_from_exception(self):
        assert issubclass(PermAgentError, Exception)

    def test_max_iterations_error_attributes(self):
        err = MaxIterationsError(10, "last content")
        assert err.max_iterations == 10
        assert err.last_content == "last content"
        assert "10" in str(err)

    def test_tool_execution_error_attributes(self):
        original = ValueError("bad value")
        err = ToolExecutionError("my_tool", original)
        assert err.tool_name == "my_tool"
        assert err.original is original
        assert "my_tool" in str(err)
        assert "ValueError" in str(err)

    def test_guardrail_error_attributes(self):
        err = GuardrailError("too long")
        assert err.reason == "too long"
        assert "too long" in str(err)

    def test_retry_exhausted_error_attributes(self):
        original = ConnectionError("timeout")
        err = RetryExhaustedError(3, original)
        assert err.attempts == 3
        assert err.last_error is original
        assert "3" in str(err)

    def test_handoff_error_attributes(self):
        err = HandoffError("agent_b", "not found")
        assert err.target == "agent_b"
        assert err.reason == "not found"
        assert "agent_b" in str(err)

    def test_catching_by_base_class(self):
        with pytest.raises(PermAgentError):
            raise MaxIterationsError(5)

        with pytest.raises(PermAgentError):
            raise ToolExecutionError("t", ValueError("x"))

        with pytest.raises(PermAgentError):
            raise GuardrailError("r")

        with pytest.raises(PermAgentError):
            raise RetryExhaustedError(1, RuntimeError("e"))

        with pytest.raises(PermAgentError):
            raise HandoffError("a", "b")


# ---------------------------------------------------------------------------
# LLM handler with retry integration test
# ---------------------------------------------------------------------------


class TestLlmHandlerWithRetry:
    @patch("litellm.completion")
    def test_llm_handler_with_retry(self, mock_completion):
        """LlmHandler respects retry config in spec."""
        mock_completion.side_effect = [
            ConnectionError("transient"),
            _make_text_response("Success after retry"),
        ]

        engine = build_agent_engine()
        spec = [
            {
                "op": "llm",
                "model": "openai/gpt-4o",
                "messages": [{"role": "user", "content": "Hello"}],
                "retry": {"max_retries": 2, "backoff_factor": 0.01},
                "path": "/result",
            }
        ]
        result = engine.apply(spec, source={}, dest={})
        assert result == {"result": "Success after retry"}
        assert mock_completion.call_count == 2

    @patch("litellm.completion")
    def test_llm_handler_retry_exhausted(self, mock_completion):
        """LlmHandler raises RetryExhaustedError when all retries fail."""
        mock_completion.side_effect = ConnectionError("permanent")

        engine = build_agent_engine()
        spec = [
            {
                "op": "llm",
                "model": "openai/gpt-4o",
                "messages": [{"role": "user", "content": "Hello"}],
                "retry": {"max_retries": 1, "backoff_factor": 0.01},
                "path": "/result",
            }
        ]
        with pytest.raises(RetryExhaustedError):
            engine.apply(spec, source={}, dest={})

    @patch("litellm.completion")
    def test_llm_handler_without_retry(self, mock_completion):
        """LlmHandler works normally without retry config."""
        mock_completion.return_value = _make_text_response("Direct")

        engine = build_agent_engine()
        spec = [
            {
                "op": "llm",
                "model": "openai/gpt-4o",
                "messages": [{"role": "user", "content": "Hello"}],
                "path": "/result",
            }
        ]
        result = engine.apply(spec, source={}, dest={})
        assert result == {"result": "Direct"}
        assert mock_completion.call_count == 1


# ---------------------------------------------------------------------------
# Streaming tool error recovery tests
# ---------------------------------------------------------------------------


def _make_stream_chunks(tokens: list[str]) -> list[MagicMock]:
    chunks = []
    for token in tokens:
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = token
        chunk.choices[0].delta.tool_calls = None
        chunks.append(chunk)
    return chunks


def _make_stream_tool_call_chunks(tool_calls: list[dict]) -> list[MagicMock]:
    chunks = []
    for i, tc in enumerate(tool_calls):
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = None
        tc_mock = MagicMock()
        tc_mock.index = i
        tc_mock.id = tc["id"]
        tc_mock.function.name = tc["name"]
        tc_mock.function.arguments = tc.get("arguments", json.dumps(tc.get("args", {})))
        chunk.choices[0].delta.tool_calls = [tc_mock]
        chunks.append(chunk)
    return chunks


class TestStreamingToolErrorRecovery:
    @patch("perm_agent.handlers.streaming.litellm.completion")
    def test_tool_error_sent_back_to_llm(self, mock_completion):
        """When a tool raises in streaming handler, the error is sent back to LLM."""
        tool_chunks = _make_stream_tool_call_chunks(
            [{"id": "call_1", "name": "failing_tool", "args": {"query": "test"}}]
        )
        text_chunks = _make_stream_chunks(["Recovered"])

        mock_completion.side_effect = [iter(tool_chunks), iter(text_chunks)]

        engine = build_agent_engine(tools={"search": search, "failing_tool": failing_tool})
        spec = [
            {
                "op": "streaming_agent_loop",
                "model": "openai/gpt-4o",
                "instructions": "Help",
                "input": "test",
                "tools": ["search", "failing_tool"],
                "path": "/result",
            }
        ]
        result = engine.apply(spec, source={}, dest={})
        assert result == {"result": "Recovered"}

        second_call_messages = mock_completion.call_args_list[1][1]["messages"]
        tool_msg = [m for m in second_call_messages if m.get("role") == "tool"]
        assert len(tool_msg) == 1
        assert "Error executing tool 'failing_tool'" in tool_msg[0]["content"]
        assert "ValueError" in tool_msg[0]["content"]


class TestStreamingJsonParseRecovery:
    @patch("perm_agent.handlers.streaming.litellm.completion")
    def test_invalid_json_arguments_handled(self, mock_completion):
        """When tool arguments are invalid JSON, the error is sent back to LLM."""
        tool_chunks = _make_stream_tool_call_chunks(
            [{"id": "call_1", "name": "search", "arguments": "{invalid json}"}]
        )
        text_chunks = _make_stream_chunks(["Recovered from bad JSON"])

        mock_completion.side_effect = [iter(tool_chunks), iter(text_chunks)]

        engine = build_agent_engine(tools={"search": search})
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
        result = engine.apply(spec, source={}, dest={})
        assert result == {"result": "Recovered from bad JSON"}

        second_call_messages = mock_completion.call_args_list[1][1]["messages"]
        tool_msg = [m for m in second_call_messages if m.get("role") == "tool"]
        assert len(tool_msg) == 1
        assert "Error parsing arguments for tool 'search'" in tool_msg[0]["content"]


# ---------------------------------------------------------------------------
# Async agent loop error handling tests
# ---------------------------------------------------------------------------


def _make_async_tool_response(tool_calls: list[dict]) -> MagicMock:
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


def _make_async_text_response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.choices[0].message.tool_calls = None
    return response


class TestAsyncAgentLoopToolError:
    @patch("litellm.acompletion", new_callable=AsyncMock)
    async def test_tool_error_caught_and_sent_back(self, mock_acompletion):
        """Async handler catches tool errors and sends them back to LLM."""
        mock_acompletion.side_effect = [
            _make_async_tool_response(
                [{"id": "call_1", "name": "failing_tool", "args": {"query": "x"}}]
            ),
            _make_async_text_response("Recovered"),
        ]

        registry = ToolRegistry()
        registry.register("failing_tool", failing_tool)
        handler = AsyncAgentLoopHandler(registry)

        result = await handler.execute(
            model="openai/gpt-4o",
            instructions="Help",
            input="test",
            tools=["failing_tool"],
        )
        assert result == "Recovered"

        second_call_messages = mock_acompletion.call_args_list[1][1]["messages"]
        tool_msg = [m for m in second_call_messages if m.get("role") == "tool"]
        assert len(tool_msg) == 1
        assert "Error executing tool 'failing_tool'" in tool_msg[0]["content"]


# ---------------------------------------------------------------------------
# Handoff uses HandoffError tests
# ---------------------------------------------------------------------------


class TestHandoffUsesHandoffError:
    def test_handoff_raises_handoff_error(self):
        engine = build_agent_engine()
        spec = [{"op": "handoff", "to": "nonexistent", "input": {}}]
        with pytest.raises(HandoffError, match="not found"):
            engine.apply(spec, source={}, dest={})


class TestAsyncHandoffUsesHandoffError:
    async def test_async_handoff_raises_handoff_error(self):
        handler = AsyncHandoffHandler(agents={})
        with pytest.raises(HandoffError, match="not found"):
            await handler.execute(to="nonexistent", input={})


# ---------------------------------------------------------------------------
# Retry config validation tests
# ---------------------------------------------------------------------------


class TestRetryConfigValidation:
    def test_negative_max_retries_raises(self):
        with pytest.raises(ValueError, match="max_retries must be >= 0"):
            RetryConfig(max_retries=-1)

    def test_zero_max_retries_allowed(self):
        config = RetryConfig(max_retries=0)
        assert config.max_retries == 0


class TestRetryExceptionChaining:
    def test_cause_is_set(self):
        def always_fails() -> str:
            raise ConnectionError("timeout")

        config = RetryConfig(max_retries=1, backoff_factor=0.01)
        with pytest.raises(RetryExhaustedError) as exc_info:
            with_retry(always_fails, config)

        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ConnectionError)


# ---------------------------------------------------------------------------
# Content filter invalid regex tests
# ---------------------------------------------------------------------------


class TestContentFilterInvalidRegex:
    def test_invalid_regex_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid regex pattern"):
            ContentFilterGuardrail(["[invalid("])
