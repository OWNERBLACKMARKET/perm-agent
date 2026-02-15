from __future__ import annotations

import json
from typing import Any

import litellm
from j_perm import ActionHandler, ExecutionContext


def _last_assistant_content(messages: list[dict[str, Any]]) -> str:
    """Return the content of the last assistant message, or empty string."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            return msg["content"]
    return ""


class AgentLoopHandler(ActionHandler):
    def execute(self, step: Any, ctx: ExecutionContext) -> Any:
        model = ctx.engine.process_value(step["model"], ctx)
        instructions = ctx.engine.process_value(step["instructions"], ctx)
        user_input = ctx.engine.process_value(step["input"], ctx)
        tool_names = step.get("tools", [])
        max_iterations = step.get("max_iterations", 10)
        memory_limit = step.get("memory_limit", 20)

        registry = ctx.metadata["_tool_registry"]
        schemas = ctx.metadata["_tool_schemas"]

        if tool_names:
            tool_schemas = [s for s in schemas if s["function"]["name"] in tool_names]
        else:
            tool_schemas = schemas

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": str(user_input)},
        ]

        tracer = ctx.metadata.get("_tracer")
        loop_span_id = None
        if tracer:
            loop_span_id = tracer.start_span("agent_loop", model, metadata={"model": model})

        for iteration in range(max_iterations):
            if tracer and loop_span_id:
                tracer.add_event(
                    loop_span_id,
                    "agent_loop.iteration",
                    {
                        "iteration": iteration + 1,
                    },
                )

            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages[-memory_limit:],
            }
            if tool_schemas:
                kwargs["tools"] = tool_schemas

            llm_span_id = None
            if tracer:
                llm_span_id = tracer.start_span("llm", model, metadata={"model": model})

            response = litellm.completion(**kwargs)
            choice = response.choices[0].message

            if tracer and llm_span_id:
                usage = getattr(response, "usage", None)
                usage_data = {}
                if usage:
                    usage_data = {
                        "input_tokens": getattr(usage, "prompt_tokens", 0),
                        "output_tokens": getattr(usage, "completion_tokens", 0),
                    }
                span = tracer.get_span(llm_span_id)
                if span:
                    span.metadata["usage"] = usage_data
                tracer.end_span(llm_span_id)

            if not choice.tool_calls:
                result = choice.content
                break

            messages.append(choice.model_dump())

            for tool_call in choice.tool_calls:
                fn_name = tool_call.function.name

                try:
                    fn_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as e:
                    fn_result = f"Error parsing arguments for tool '{fn_name}': {e}"
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": fn_result,
                        }
                    )
                    continue

                fn = registry.get(fn_name)
                try:
                    fn_result = fn(**fn_args) if isinstance(fn_args, dict) else fn(fn_args)
                except Exception as e:
                    fn_result = f"Error executing tool '{fn_name}': {type(e).__name__}: {e}"

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": (
                            json.dumps(fn_result) if not isinstance(fn_result, str) else fn_result
                        ),
                    }
                )
        else:
            # On max iterations, return the last assistant message, not the last message
            result = _last_assistant_content(messages)

        if tracer and loop_span_id:
            tracer.end_span(loop_span_id)

        path = step.get("path")
        if path:
            resolved_path = ctx.engine.process_value(path, ctx)
            ctx.engine.processor.set(resolved_path, ctx, result)
        else:
            ctx.dest = result

        return ctx.dest
