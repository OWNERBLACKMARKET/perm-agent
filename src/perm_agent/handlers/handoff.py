from __future__ import annotations

from typing import Any

from j_perm import ActionHandler, ExecutionContext


class HandoffHandler(ActionHandler):
    def execute(self, step: Any, ctx: ExecutionContext) -> Any:
        target = ctx.engine.process_value(step["to"], ctx)
        raw_input = step.get("input", {})
        resolved_input = ctx.engine.process_value(raw_input, ctx)

        specs = ctx.metadata.get("_agent_specs", {})
        if target not in specs:
            raise KeyError(f"Agent spec '{target}' not found")

        spec = specs[target]
        result = ctx.engine.apply(spec, source=resolved_input, dest={})

        path = step.get("path")
        if path:
            resolved_path = ctx.engine.process_value(path, ctx)
            ctx.engine.processor.set(resolved_path, ctx, result)
        else:
            ctx.dest = result

        return ctx.dest
