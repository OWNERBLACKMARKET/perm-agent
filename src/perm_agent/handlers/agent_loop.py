from __future__ import annotations

import json
from typing import Any

import litellm
from j_perm import ActionHandler, ExecutionContext


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

        for _ in range(max_iterations):
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages[-memory_limit:],
            }
            if tool_schemas:
                kwargs["tools"] = tool_schemas

            response = litellm.completion(**kwargs)
            choice = response.choices[0].message

            if not choice.tool_calls:
                result = choice.content
                break

            messages.append(choice.model_dump())

            for tool_call in choice.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)

                fn = registry.get(fn_name)
                if isinstance(fn_args, dict):
                    fn_result = fn(**fn_args)
                else:
                    fn_result = fn(fn_args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(fn_result) if not isinstance(fn_result, str) else fn_result,
                })
        else:
            result = messages[-1].get("content", "")

        path = step.get("path")
        if path:
            resolved_path = ctx.engine.process_value(path, ctx)
            ctx.engine.processor.set(resolved_path, ctx, result)
        else:
            ctx.dest = result

        return ctx.dest
