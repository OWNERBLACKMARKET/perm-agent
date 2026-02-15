from __future__ import annotations

from typing import Any

from j_perm import ActionHandler, ExecutionContext

from perm_agent.exceptions import HandoffError


class HandoffHandler(ActionHandler):
    def execute(self, step: Any, ctx: ExecutionContext) -> Any:
        target = ctx.engine.process_value(step["to"], ctx)
        raw_input = step.get("input", {})
        resolved_input = ctx.engine.process_value(raw_input, ctx)

        specs = ctx.metadata.get("_agent_specs", {})
        if target not in specs:
            raise HandoffError(target, "not found in agent specs")

        tracer = ctx.metadata.get("_tracer")
        span_id = None
        if tracer:
            span_id = tracer.start_span("handoff", target, metadata={"target": target})
            tracer.add_event(span_id, "handoff.delegate", {"target": target})

        try:
            spec = specs[target]
            result = ctx.engine.apply(spec, source=resolved_input, dest={})
        except Exception as exc:
            if tracer and span_id:
                tracer.end_span(span_id, status="error", error=str(exc))
            raise

        if tracer and span_id:
            tracer.end_span(span_id)

        path = step.get("path")
        if path:
            resolved_path = ctx.engine.process_value(path, ctx)
            ctx.engine.processor.set(resolved_path, ctx, result)
        else:
            ctx.dest = result

        return ctx.dest
