import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from perm_agent.exceptions import HandoffError
from perm_agent.handlers.async_agent_loop import AsyncAgentLoopHandler
from perm_agent.handlers.async_handoff import AsyncHandoffHandler
from perm_agent.handlers.async_llm import AsyncLlmHandler
from perm_agent.handlers.async_tool import AsyncToolHandler
from perm_agent.registry import ToolRegistry


def _make_response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.choices[0].message.tool_calls = None
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


async def async_search(query: str) -> str:
    """Async search for information"""
    return f"Async found: {query}"


class TestAsyncLlmHandler:
    @patch("litellm.acompletion", new_callable=AsyncMock)
    async def test_basic_call(self, mock_acompletion: AsyncMock) -> None:
        mock_acompletion.return_value = _make_response("Hello!")
        handler = AsyncLlmHandler()

        result = await handler.execute(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert result == "Hello!"
        mock_acompletion.assert_awaited_once()
        call_kwargs = mock_acompletion.call_args[1]
        assert call_kwargs["model"] == "openai/gpt-4o"
        assert call_kwargs["temperature"] == 0.7

    @patch("litellm.acompletion", new_callable=AsyncMock)
    async def test_json_response_format(self, mock_acompletion: AsyncMock) -> None:
        mock_acompletion.return_value = _make_response('{"name": "Alice"}')
        handler = AsyncLlmHandler()

        result = await handler.execute(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "Extract"}],
            response_format={"name": "str"},
        )

        assert result == {"name": "Alice"}
        call_kwargs = mock_acompletion.call_args[1]
        assert call_kwargs["response_format"] == {"type": "json_object"}

    @patch("litellm.acompletion", new_callable=AsyncMock)
    async def test_custom_temperature(self, mock_acompletion: AsyncMock) -> None:
        mock_acompletion.return_value = _make_response("ok")
        handler = AsyncLlmHandler()

        await handler.execute(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "test"}],
            temperature=0.2,
        )

        call_kwargs = mock_acompletion.call_args[1]
        assert call_kwargs["temperature"] == 0.2


class TestAsyncToolHandler:
    async def test_sync_function(self) -> None:
        registry = ToolRegistry()
        registry.register("search", search)
        handler = AsyncToolHandler(registry)

        result = await handler.execute(name="search", args={"query": "hello"})
        assert result == "Found: hello"

    async def test_async_function(self) -> None:
        registry = ToolRegistry()
        registry.register("search", async_search)
        handler = AsyncToolHandler(registry)

        result = await handler.execute(name="search", args={"query": "hello"})
        assert result == "Async found: hello"

    async def test_missing_tool_raises(self) -> None:
        registry = ToolRegistry()
        handler = AsyncToolHandler(registry)

        with pytest.raises(KeyError, match="not registered"):
            await handler.execute(name="nonexistent", args={})


class TestAsyncAgentLoopHandler:
    @patch("litellm.acompletion", new_callable=AsyncMock)
    async def test_direct_response(self, mock_acompletion: AsyncMock) -> None:
        mock_acompletion.return_value = _make_response("The answer is 42")
        registry = ToolRegistry()
        registry.register("search", search)
        handler = AsyncAgentLoopHandler(registry)

        result = await handler.execute(
            model="openai/gpt-4o",
            instructions="You are helpful",
            input="What is the answer?",
            tools=["search"],
        )

        assert result == "The answer is 42"

    @patch("litellm.acompletion", new_callable=AsyncMock)
    async def test_with_tool_calls(self, mock_acompletion: AsyncMock) -> None:
        mock_acompletion.side_effect = [
            _make_tool_response([{"id": "call_1", "name": "search", "args": {"query": "test"}}]),
            _make_response("Based on search: Found: test"),
        ]
        registry = ToolRegistry()
        registry.register("search", search)
        handler = AsyncAgentLoopHandler(registry)

        result = await handler.execute(
            model="openai/gpt-4o",
            instructions="Use search when needed",
            input="Find test",
            tools=["search"],
        )

        assert result == "Based on search: Found: test"
        assert mock_acompletion.await_count == 2

    @patch("litellm.acompletion", new_callable=AsyncMock)
    async def test_with_async_tool(self, mock_acompletion: AsyncMock) -> None:
        mock_acompletion.side_effect = [
            _make_tool_response([{"id": "call_1", "name": "search", "args": {"query": "test"}}]),
            _make_response("Got async result"),
        ]
        registry = ToolRegistry()
        registry.register("search", async_search)
        handler = AsyncAgentLoopHandler(registry)

        result = await handler.execute(
            model="openai/gpt-4o",
            instructions="Help",
            input="Search",
            tools=["search"],
        )

        assert result == "Got async result"

    @patch("litellm.acompletion", new_callable=AsyncMock)
    async def test_max_iterations(self, mock_acompletion: AsyncMock) -> None:
        mock_acompletion.return_value = _make_tool_response(
            [{"id": "call_x", "name": "search", "args": {"query": "loop"}}]
        )
        registry = ToolRegistry()
        registry.register("search", search)
        handler = AsyncAgentLoopHandler(registry)

        await handler.execute(
            model="openai/gpt-4o",
            instructions="Keep searching",
            input="Loop forever",
            tools=["search"],
            max_iterations=3,
        )

        assert mock_acompletion.await_count == 3


class TestAsyncHandoffHandler:
    async def test_basic_handoff(self) -> None:
        target_agent = MagicMock()
        target_agent.arun = AsyncMock(return_value="handoff result")
        handler = AsyncHandoffHandler(agents={"summarizer": target_agent})

        result = await handler.execute(to="summarizer", input={"input": "hello"})
        assert result == "handoff result"
        target_agent.arun.assert_awaited_once_with(input="hello")

    async def test_missing_agent_raises(self) -> None:
        handler = AsyncHandoffHandler(agents={})

        with pytest.raises(HandoffError, match="not found"):
            await handler.execute(to="nonexistent", input={})
