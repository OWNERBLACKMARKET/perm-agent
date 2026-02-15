# Changelog

## 0.3.0 (2026-02-15)

### Added
- **Inter-agent communication**: Blackboard + Mailbox system for agent coordination
  - `Blackboard`: shared key-value store visible to all agents in a pipeline
  - `Mailbox`: per-agent message queues for directed messaging between agents
  - `CommunicationHub`: facade combining Blackboard, Mailbox, and AgentContext
- **Communication tools** (LLM-callable):
  - `send_message`: send directed messages to other agents
  - `check_messages`: read inbox (consume or peek)
  - `post_to_board`: write to shared blackboard
  - `read_board`: read one key or all entries from blackboard
- **Pipeline communication**: `Pipeline(enable_communication=True)` to enable inter-agent communication
- `AgentContext`: tracks current agent identity across pipeline steps
- `build_comm_tools()` factory and `COMM_TOOL_NAMES` constant
- Full public API exports: `CommunicationHub`, `Blackboard`, `BoardEntry`, `Mailbox`, `Message`, `AgentContext`

### Design
- Zero changes to agent loop — communication happens through LLM tool calls
- `CommunicationHub` injected via existing middleware, no new dependencies
- Fully backward compatible: `enable_communication=False` by default

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
