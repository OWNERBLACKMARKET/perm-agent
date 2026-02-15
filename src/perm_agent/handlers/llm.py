from __future__ import annotations

import json
from typing import Any

import litellm
from j_perm import ActionHandler, ExecutionContext

from perm_agent.retry import RetryConfig, with_retry


class LlmHandler(ActionHandler):
    def execute(self, step: Any, ctx: ExecutionContext) -> Any:
        model = ctx.engine.process_value(step["model"], ctx)
        messages = ctx.engine.process_value(step["messages"], ctx)
        temperature = step.get("temperature", 0.7)
        response_format = step.get("response_format")

        tracer = ctx.metadata.get("_tracer")
        span_id = None
        if tracer:
            span_id = tracer.start_span("llm", model, metadata={"model": model})
            tracer.add_event(
                span_id,
                "llm.request",
                {
                    "model": model,
                    "message_count": len(messages),
                    "temperature": temperature,
                },
            )

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        if response_format:
            kwargs["response_format"] = {"type": "json_object"}

        retry_spec = step.get("retry")
        try:
            if retry_spec:
                retry_cfg = RetryConfig(**retry_spec)
                response = with_retry(litellm.completion, retry_cfg, **kwargs)
            else:
                response = litellm.completion(**kwargs)
        except Exception as exc:
            if tracer and span_id:
                tracer.end_span(span_id, status="error", error=str(exc))
            raise

        content = response.choices[0].message.content

        if tracer and span_id:
            usage_data = {}
            usage = getattr(response, "usage", None)
            if usage:
                usage_data = {
                    "input_tokens": getattr(usage, "prompt_tokens", 0),
                    "output_tokens": getattr(usage, "completion_tokens", 0),
                }
            tracer.add_event(span_id, "llm.response", usage_data)

            tool_calls = getattr(response.choices[0].message, "tool_calls", None)
            if tool_calls:
                tracer.add_event(
                    span_id,
                    "llm.tool_calls",
                    {
                        "tools": [tc.function.name for tc in tool_calls],
                    },
                )

            # Store usage in span metadata for CostTracker
            span = tracer.get_span(span_id)
            if span:
                span.metadata["usage"] = usage_data
            tracer.end_span(span_id)

        if response_format and content:
            content = json.loads(content)

        path = step.get("path")
        if path:
            resolved_path = ctx.engine.process_value(path, ctx)
            ctx.engine.processor.set(resolved_path, ctx, content)
        else:
            ctx.dest = content

        return ctx.dest
