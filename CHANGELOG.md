# Changelog

## 0.2.0 (2026-02-15)

### Added
- **Python-first API**: `Agent`, `Pipeline`, `@agent` decorator for simple usage
- **Async support**: `AsyncLlmHandler`, `AsyncToolHandler`, `AsyncAgentLoopHandler`, `AsyncHandoffHandler`
- **Streaming**: `StreamingLlmHandler`, `StreamingAgentLoopHandler` with event system
- **Observability**: `Tracer`, `Span`, `CostTracker`, `ConsoleTracerHook` with OpenTelemetry-inspired design
- **Structured output**: `StructuredOutput` with Pydantic model validation
- **Error handling**: `RetryConfig`, `with_retry` for transient failure recovery
- **Guardrails**: `MaxLengthGuardrail`, `ContentFilterGuardrail`, `GuardrailPipeline`
- **Custom exceptions**: `PermAgentError`, `MaxIterationsError`, `ToolExecutionError`, `GuardrailError`
- **Advanced ToolRegistry**: `Optional`, `list[T]`, `Literal`, `Enum`, Pydantic models, docstring parsing, `@tool` decorator

### Fixed
- Agent loop now sends tool execution errors back to LLM instead of crashing
- Max iterations returns last assistant message instead of last tool result

## 0.1.0 (2026-02-15)

### Added
- Initial release
- Core handlers: `LlmHandler`, `ToolHandler`, `AgentLoopHandler`, `HandoffHandler`
- `ToolRegistry` with auto schema generation
- `build_agent_engine` factory
- j-perm integration for declarative JSON workflows
