"""Communication tool factories for inter-agent messaging."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from .communication import CommunicationHub

COMM_TOOL_NAMES: frozenset[str] = frozenset({
    "send_message",
    "check_messages",
    "post_to_board",
    "read_board",
})


def build_comm_tools(hub: CommunicationHub) -> dict[str, Callable[..., Any]]:
    """Build all communication tools bound to the given hub.

    Returns a dict of {tool_name: callable} ready for ToolRegistry.
    """
    return {
        "send_message": _make_send_message(hub),
        "check_messages": _make_check_messages(hub),
        "post_to_board": _make_post_to_board(hub),
        "read_board": _make_read_board(hub),
    }


def _make_send_message(hub: CommunicationHub) -> Callable[..., str]:
    def send_message(to: str, content: str, metadata: str | None = None) -> str:
        """Send a directed message to another agent in the pipeline.

        Args:
            to: Name of the recipient agent.
            content: Message content.
            metadata: Optional JSON string with additional data.
        """
        meta = json.loads(metadata) if metadata else {}
        sender = hub.agent_context.current_agent
        msg = hub.mailbox.send(
            sender=sender, recipient=to, content=content, metadata=meta,
        )
        return json.dumps({"status": "sent", "message_id": msg.id, "to": to})

    send_message.__name__ = "send_message"
    send_message.__qualname__ = "send_message"
    return send_message


def _make_check_messages(hub: CommunicationHub) -> Callable[..., str]:
    def check_messages(peek: bool = False) -> str:
        """Check your inbox for messages from other agents.

        Args:
            peek: If true, messages remain in the queue (non-destructive).
        """
        agent = hub.agent_context.current_agent
        messages = hub.mailbox.peek(agent) if peek else hub.mailbox.receive(agent)
        return json.dumps({
            "messages": [m.to_dict() for m in messages],
            "count": len(messages),
        })

    check_messages.__name__ = "check_messages"
    check_messages.__qualname__ = "check_messages"
    return check_messages


def _make_post_to_board(hub: CommunicationHub) -> Callable[..., str]:
    def post_to_board(key: str, value: str) -> str:
        """Post a value to the shared blackboard visible to all agents.

        Args:
            key: Key to store the value under.
            value: Value to store (plain text or JSON string).
        """
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            parsed = value

        author = hub.agent_context.current_agent
        hub.blackboard.post(key=key, value=parsed, author=author)
        return json.dumps({"status": "posted", "key": key})

    post_to_board.__name__ = "post_to_board"
    post_to_board.__qualname__ = "post_to_board"
    return post_to_board


def _make_read_board(hub: CommunicationHub) -> Callable[..., str]:
    def read_board(key: str | None = None) -> str:
        """Read from the shared blackboard.

        Args:
            key: Specific key to read. If omitted, returns all entries.
        """
        if key:
            entry = hub.blackboard.read(key)
            if entry is None:
                return json.dumps({
                    "error": f"Key '{key}' not found",
                    "available_keys": hub.blackboard.keys(),
                })
            return json.dumps(entry.to_dict())
        return json.dumps(hub.blackboard.to_dict())

    read_board.__name__ = "read_board"
    read_board.__qualname__ = "read_board"
    return read_board
