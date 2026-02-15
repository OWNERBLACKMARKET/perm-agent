from __future__ import annotations

from typing import Any, Protocol


class AsyncRunner(Protocol):
    """Protocol for anything that can run an agent spec asynchronously."""

    async def arun(self, input: str, **context: Any) -> Any: ...


class AsyncHandoffHandler:
    """Async version of HandoffHandler that delegates to async agent runners."""

    def __init__(self, agents: dict[str, AsyncRunner]) -> None:
        self._agents = agents

    async def execute(
        self,
        *,
        to: str,
        input: Any = None,
    ) -> Any:
        if to not in self._agents:
            from perm_agent.exceptions import HandoffError

            raise HandoffError(to, "not found")

        agent = self._agents[to]
        if isinstance(input, dict):
            return await agent.arun(**input)
        return await agent.arun(input=str(input) if input else "")
