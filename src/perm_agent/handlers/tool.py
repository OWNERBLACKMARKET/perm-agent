from __future__ import annotations

from typing import Any

from j_perm import ActionHandler, ExecutionContext


class ToolHandler(ActionHandler):
    def execute(self, step: Any, ctx: ExecutionContext) -> Any:
        name = ctx.engine.process_value(step["name"], ctx)
        raw_args = step.get("args", {})
        args = ctx.engine.process_value(raw_args, ctx)

        registry = ctx.metadata["_tool_registry"]
        fn = registry.get(name)

        tracer = ctx.metadata.get("_tracer")
        span_id = None
        if tracer:
            span_id = tracer.start_span("tool", name, metadata={"tool": name})
            tracer.add_event(
                span_id,
                "tool.execution",
                {
                    "tool": name,
                    "args": args,
                },
            )

        try:
            if isinstance(args, dict):
                result = fn(**args)
            elif isinstance(args, list):
                result = fn(*args)
            else:
                result = fn(args)
        except Exception as exc:
            if tracer and span_id:
                tracer.end_span(span_id, status="error", error=str(exc))
            raise

        if tracer and span_id:
            tracer.add_event(
                span_id,
                "tool.result",
                {
                    "result_type": type(result).__name__,
                },
            )
            tracer.end_span(span_id)

        path = step.get("path")
        if path:
            resolved_path = ctx.engine.process_value(path, ctx)
            ctx.engine.processor.set(resolved_path, ctx, result)
        else:
            ctx.dest = result

        return ctx.dest
