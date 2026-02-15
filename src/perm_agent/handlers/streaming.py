from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import litellm
from j_perm import ActionHandler, ExecutionContext

from perm_agent.events import (
    AgentCompleteEvent,
    StreamEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class StreamingLlmHandler(ActionHandler):
    """LLM handler that streams tokens via a callback."""

    def execute(self, step: Any, ctx: ExecutionContext) -> Any:
        model = ctx.engine.process_value(step["model"], ctx)
        messages = ctx.engine.process_value(step["messages"], ctx)
        temperature = step.get("temperature", 0.7)
        on_event: Callable[[StreamEvent], None] | None = step.get("on_event")

        tracer = ctx.metadata.get("_tracer")
        span_id = None
        if tracer:
            span_id = tracer.start_span("streaming_llm", model, metadata={"model": model})
            tracer.add_event(span_id, "streaming.start", {"model": model})

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }

        try:
            response = litellm.completion(**kwargs)
            collected: list[str] = []

            for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content:
                    collected.append(delta.content)
                    if on_event:
                        on_event(TokenEvent(token=delta.content))

            content = "".join(collected)
        except Exception as exc:
            if tracer and span_id:
                tracer.end_span(span_id, status="error", error=str(exc))
            raise

        if tracer and span_id:
            tracer.add_event(span_id, "streaming.complete", {"token_count": len(collected)})
            tracer.end_span(span_id)

        path = step.get("path")
        if path:
            resolved_path = ctx.engine.process_value(path, ctx)
            ctx.engine.processor.set(resolved_path, ctx, content)
        else:
            ctx.dest = content

        return ctx.dest


class StreamingAgentLoopHandler(ActionHandler):
    """Agent loop that streams LLM tokens and emits tool events."""

    def execute(self, step: Any, ctx: ExecutionContext) -> Any:
        model = ctx.engine.process_value(step["model"], ctx)
        instructions = ctx.engine.process_value(step["instructions"], ctx)
        user_input = ctx.engine.process_value(step["input"], ctx)
        tool_names = step.get("tools", [])
        max_iterations = step.get("max_iterations", 10)
        memory_limit = step.get("memory_limit", 20)
        on_event: Callable[[StreamEvent], None] | None = step.get("on_event")

        registry = ctx.metadata["_tool_registry"]
        schemas = ctx.metadata["_tool_schemas"]

        tracer = ctx.metadata.get("_tracer")
        loop_span_id = None
        if tracer:
            loop_span_id = tracer.start_span(
                "streaming_agent_loop", model, metadata={"model": model}
            )

        if tool_names:
            tool_schemas = [s for s in schemas if s["function"]["name"] in tool_names]
        else:
            tool_schemas = schemas

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": str(user_input)},
        ]

        for iteration in range(max_iterations):
            if tracer and loop_span_id:
                tracer.add_event(
                    loop_span_id,
                    "streaming_agent_loop.iteration",
                    {"iteration": iteration + 1},
                )
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages[-memory_limit:],
                "stream": True,
            }
            if tool_schemas:
                kwargs["tools"] = tool_schemas

            response = litellm.completion(**kwargs)

            collected_content: list[str] = []
            tool_calls_by_index: dict[int, dict[str, Any]] = {}

            for chunk in response:
                delta = chunk.choices[0].delta

                if delta.content:
                    collected_content.append(delta.content)
                    if on_event:
                        on_event(TokenEvent(token=delta.content))

                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_by_index:
                            tool_calls_by_index[idx] = {
                                "id": tc.id or "",
                                "name": "",
                                "arguments": "",
                            }
                        entry = tool_calls_by_index[idx]
                        if tc.id:
                            entry["id"] = tc.id
                        if hasattr(tc, "function") and tc.function:
                            if tc.function.name:
                                entry["name"] = tc.function.name
                            if tc.function.arguments:
                                entry["arguments"] += tc.function.arguments

            if not tool_calls_by_index:
                result = "".join(collected_content)
                break

            # Build the assistant message with tool calls for history
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": "".join(collected_content) or None,
                "tool_calls": [
                    {
                        "id": tc_data["id"],
                        "type": "function",
                        "function": {
                            "name": tc_data["name"],
                            "arguments": tc_data["arguments"],
                        },
                    }
                    for tc_data in tool_calls_by_index.values()
                ],
            }
            messages.append(assistant_msg)

            for tc_data in tool_calls_by_index.values():
                fn_name = tc_data["name"]

                try:
                    fn_args = json.loads(tc_data["arguments"])
                except json.JSONDecodeError as e:
                    fn_result = f"Error parsing arguments for tool '{fn_name}': {e}"
                    if on_event:
                        on_event(ToolCallEvent(tool_name=fn_name, arguments={}))
                        on_event(ToolResultEvent(tool_name=fn_name, result=fn_result))
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc_data["id"],
                            "content": fn_result,
                        }
                    )
                    continue

                if on_event:
                    on_event(ToolCallEvent(tool_name=fn_name, arguments=fn_args))

                fn = registry.get(fn_name)
                try:
                    fn_result = fn(**fn_args) if isinstance(fn_args, dict) else fn(fn_args)
                except Exception as e:
                    fn_result = f"Error executing tool '{fn_name}': {type(e).__name__}: {e}"

                if on_event:
                    on_event(ToolResultEvent(tool_name=fn_name, result=fn_result))

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc_data["id"],
                        "content": (
                            json.dumps(fn_result) if not isinstance(fn_result, str) else fn_result
                        ),
                    }
                )
        else:
            result = messages[-1].get("content", "")

        if on_event:
            on_event(AgentCompleteEvent(result=result))

        if tracer and loop_span_id:
            tracer.end_span(loop_span_id)

        path = step.get("path")
        if path:
            resolved_path = ctx.engine.process_value(path, ctx)
            ctx.engine.processor.set(resolved_path, ctx, result)
        else:
            ctx.dest = result

        return ctx.dest
