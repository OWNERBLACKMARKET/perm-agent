from __future__ import annotations


class PermAgentError(Exception):
    """Base exception for all perm-agent errors."""


class MaxIterationsError(PermAgentError):
    """Raised when the agent loop exceeds the maximum number of iterations."""

    def __init__(self, max_iterations: int, last_content: str | None = None) -> None:
        self.max_iterations = max_iterations
        self.last_content = last_content
        super().__init__(f"Agent loop exceeded {max_iterations} iterations")


class ToolExecutionError(PermAgentError):
    """Raised when a tool execution fails."""

    def __init__(self, tool_name: str, original: Exception) -> None:
        self.tool_name = tool_name
        self.original = original
        super().__init__(f"Tool '{tool_name}' failed: {type(original).__name__}: {original}")


class GuardrailError(PermAgentError):
    """Raised when a guardrail check fails."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Guardrail check failed: {reason}")


class RetryExhaustedError(PermAgentError):
    """Raised when all retry attempts are exhausted."""

    def __init__(self, attempts: int, last_error: Exception) -> None:
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"All {attempts} retry attempts exhausted. "
            f"Last error: {type(last_error).__name__}: {last_error}"
        )


class HandoffError(PermAgentError):
    """Raised when an agent handoff fails."""

    def __init__(self, target: str, reason: str) -> None:
        self.target = target
        self.reason = reason
        super().__init__(f"Handoff to '{target}' failed: {reason}")
