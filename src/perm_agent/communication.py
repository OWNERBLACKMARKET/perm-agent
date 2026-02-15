"""Inter-agent communication primitives: Blackboard, Mailbox, and CommunicationHub."""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Message:
    """Directed message between agents in a pipeline run."""

    id: str
    sender: str
    recipient: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.monotonic)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "sender": self.sender,
            "recipient": self.recipient,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True, slots=True)
class BoardEntry:
    """Single entry on the shared blackboard."""

    key: str
    value: Any
    author: str
    timestamp: float = field(default_factory=time.monotonic)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "author": self.author,
            "timestamp": self.timestamp,
        }


class Blackboard:
    """Shared key-value store visible to all agents. Last-write-wins semantics."""

    def __init__(self) -> None:
        self._entries: dict[str, BoardEntry] = {}
        self._history: list[BoardEntry] = []

    def post(self, key: str, value: Any, author: str) -> None:
        entry = BoardEntry(key=key, value=value, author=author)
        self._entries[key] = entry
        self._history.append(entry)

    def read(self, key: str) -> BoardEntry | None:
        return self._entries.get(key)

    def read_all(self) -> dict[str, BoardEntry]:
        return dict(self._entries)

    def keys(self) -> list[str]:
        return list(self._entries.keys())

    @property
    def history(self) -> list[BoardEntry]:
        return list(self._history)

    def to_dict(self) -> dict[str, Any]:
        return {k: e.to_dict() for k, e in self._entries.items()}


class Mailbox:
    """Per-agent FIFO message queues. Consume-on-read semantics."""

    def __init__(self) -> None:
        self._queues: dict[str, list[Message]] = defaultdict(list)
        self._all_messages: list[Message] = []

    def send(
        self,
        sender: str,
        recipient: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        msg = Message(
            id=uuid.uuid4().hex[:12],
            sender=sender,
            recipient=recipient,
            content=content,
            metadata=metadata or {},
        )
        self._queues[recipient].append(msg)
        self._all_messages.append(msg)
        return msg

    def receive(self, agent_name: str) -> list[Message]:
        """Consume all pending messages for agent_name."""
        return self._queues.pop(agent_name, [])

    def peek(self, agent_name: str) -> list[Message]:
        """Non-destructive read of pending messages."""
        return list(self._queues.get(agent_name, []))

    def pending_count(self, agent_name: str) -> int:
        return len(self._queues.get(agent_name, []))

    @property
    def all_messages(self) -> list[Message]:
        return list(self._all_messages)


class AgentContext:
    """Mutable holder tracking which agent is currently executing."""

    __slots__ = ("current_agent",)

    def __init__(self) -> None:
        self.current_agent: str = "unknown"


class CommunicationHub:
    """Facade owning Blackboard + Mailbox + AgentContext for one pipeline run."""

    def __init__(self) -> None:
        self.blackboard = Blackboard()
        self.mailbox = Mailbox()
        self.agent_context = AgentContext()

    def to_dict(self) -> dict[str, Any]:
        return {
            "blackboard": self.blackboard.to_dict(),
            "messages": [m.to_dict() for m in self.mailbox.all_messages],
        }
