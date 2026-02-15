from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class GuardrailResult:
    """Result of a guardrail check."""

    passed: bool
    reason: str | None = None


class Guardrail(Protocol):
    """Protocol for content guardrails."""

    def check(self, content: str) -> GuardrailResult: ...


class MaxLengthGuardrail:
    """Rejects content exceeding a character limit."""

    def __init__(self, max_length: int = 10_000) -> None:
        self._max_length = max_length

    def check(self, content: str) -> GuardrailResult:
        if len(content) > self._max_length:
            return GuardrailResult(
                passed=False,
                reason=f"Content exceeds {self._max_length} chars (got {len(content)})",
            )
        return GuardrailResult(passed=True)


class ContentFilterGuardrail:
    """Rejects content matching any blocked pattern (case-insensitive regex)."""

    def __init__(self, blocked_patterns: Sequence[str]) -> None:
        try:
            self._patterns = [re.compile(p, re.IGNORECASE) for p in blocked_patterns]
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}") from e
        self._raw_patterns = list(blocked_patterns)

    def check(self, content: str) -> GuardrailResult:
        for i, pattern in enumerate(self._patterns):
            if pattern.search(content):
                return GuardrailResult(
                    passed=False,
                    reason=f"Content matches blocked pattern: '{self._raw_patterns[i]}'",
                )
        return GuardrailResult(passed=True)


class GuardrailPipeline:
    """Runs a sequence of guardrails, failing fast on the first failure."""

    def __init__(self, guardrails: Sequence[Guardrail]) -> None:
        self._guardrails = list(guardrails)

    def check(self, content: str) -> GuardrailResult:
        for guardrail in self._guardrails:
            result = guardrail.check(content)
            if not result.passed:
                return result
        return GuardrailResult(passed=True)

    def check_input(self, content: str) -> GuardrailResult:
        """Alias for check — used for clarity at call sites."""
        return self.check(content)

    def check_output(self, content: str) -> GuardrailResult:
        """Alias for check — used for clarity at call sites."""
        return self.check(content)
