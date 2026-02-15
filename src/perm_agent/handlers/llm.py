from __future__ import annotations

import json
from typing import Any

import litellm
from j_perm import ActionHandler, ExecutionContext


class LlmHandler(ActionHandler):
    def execute(self, step: Any, ctx: ExecutionContext) -> Any:
        model = ctx.engine.process_value(step["model"], ctx)
        messages = ctx.engine.process_value(step["messages"], ctx)
        temperature = step.get("temperature", 0.7)
        response_format = step.get("response_format")

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        if response_format:
            kwargs["response_format"] = {"type": "json_object"}

        response = litellm.completion(**kwargs)
        content = response.choices[0].message.content

        if response_format and content:
            content = json.loads(content)

        path = step.get("path")
        if path:
            resolved_path = ctx.engine.process_value(path, ctx)
            ctx.engine.processor.set(resolved_path, ctx, content)
        else:
            ctx.dest = content

        return ctx.dest
