import json

import pytest

from perm_agent.comm_tools import COMM_TOOL_NAMES, build_comm_tools
from perm_agent.communication import CommunicationHub


def _make_hub(agent_name: str = "test_agent") -> CommunicationHub:
    hub = CommunicationHub()
    hub.agent_context.current_agent = agent_name
    return hub


class TestBuildCommTools:
    def test_returns_four_tools(self):
        hub = _make_hub()
        tools = build_comm_tools(hub)
        assert len(tools) == 4

    def test_tool_names_match_constant(self):
        hub = _make_hub()
        tools = build_comm_tools(hub)
        assert set(tools.keys()) == COMM_TOOL_NAMES

    def test_tools_have_correct_names(self):
        hub = _make_hub()
        tools = build_comm_tools(hub)
        for name, fn in tools.items():
            assert fn.__name__ == name


class TestSendMessage:
    def test_sends_message_to_recipient(self):
        hub = _make_hub("alice")
        tools = build_comm_tools(hub)
        result = json.loads(tools["send_message"](to="bob", content="hello"))
        assert result["status"] == "sent"
        assert result["to"] == "bob"
        assert "message_id" in result
        # Verify message was actually queued
        messages = hub.mailbox.receive("bob")
        assert len(messages) == 1
        assert messages[0].sender == "alice"
        assert messages[0].content == "hello"

    def test_uses_current_agent_as_sender(self):
        hub = _make_hub("researcher")
        tools = build_comm_tools(hub)
        tools["send_message"](to="writer", content="data ready")
        messages = hub.mailbox.receive("writer")
        assert messages[0].sender == "researcher"

    def test_metadata_parsed_from_json_string(self):
        hub = _make_hub("a")
        tools = build_comm_tools(hub)
        tools["send_message"](to="b", content="hi", metadata='{"priority": "high"}')
        messages = hub.mailbox.receive("b")
        assert messages[0].metadata == {"priority": "high"}

    def test_no_metadata_defaults_to_empty(self):
        hub = _make_hub("a")
        tools = build_comm_tools(hub)
        tools["send_message"](to="b", content="hi")
        messages = hub.mailbox.receive("b")
        assert messages[0].metadata == {}


class TestCheckMessages:
    def test_returns_pending_messages(self):
        hub = _make_hub("bob")
        hub.mailbox.send("alice", "bob", "hello")
        tools = build_comm_tools(hub)
        result = json.loads(tools["check_messages"]())
        assert result["count"] == 1
        assert result["messages"][0]["content"] == "hello"

    def test_consume_empties_queue(self):
        hub = _make_hub("bob")
        hub.mailbox.send("alice", "bob", "hi")
        tools = build_comm_tools(hub)
        tools["check_messages"]()
        result = json.loads(tools["check_messages"]())
        assert result["count"] == 0

    def test_peek_preserves_queue(self):
        hub = _make_hub("bob")
        hub.mailbox.send("alice", "bob", "hi")
        tools = build_comm_tools(hub)
        tools["check_messages"](peek=True)
        result = json.loads(tools["check_messages"](peek=True))
        assert result["count"] == 1

    def test_empty_inbox_returns_empty(self):
        hub = _make_hub("lonely")
        tools = build_comm_tools(hub)
        result = json.loads(tools["check_messages"]())
        assert result == {"messages": [], "count": 0}


class TestPostToBoard:
    def test_posts_string_value(self):
        hub = _make_hub("writer")
        tools = build_comm_tools(hub)
        result = json.loads(tools["post_to_board"](key="draft", value="some text"))
        assert result == {"status": "posted", "key": "draft"}
        entry = hub.blackboard.read("draft")
        assert entry is not None
        assert entry.value == "some text"

    def test_posts_json_parsed_value(self):
        hub = _make_hub("analyst")
        tools = build_comm_tools(hub)
        tools["post_to_board"](key="data", value='{"score": 42}')
        entry = hub.blackboard.read("data")
        assert entry is not None
        assert entry.value == {"score": 42}

    def test_uses_current_agent_as_author(self):
        hub = _make_hub("researcher")
        tools = build_comm_tools(hub)
        tools["post_to_board"](key="k", value="v")
        entry = hub.blackboard.read("k")
        assert entry is not None
        assert entry.author == "researcher"


class TestReadBoard:
    def test_reads_specific_key(self):
        hub = _make_hub()
        hub.blackboard.post("findings", "important", author="a")
        tools = build_comm_tools(hub)
        result = json.loads(tools["read_board"](key="findings"))
        assert result["value"] == "important"

    def test_reads_all_keys(self):
        hub = _make_hub()
        hub.blackboard.post("a", 1, author="x")
        hub.blackboard.post("b", 2, author="y")
        tools = build_comm_tools(hub)
        result = json.loads(tools["read_board"]())
        assert "a" in result
        assert "b" in result

    def test_missing_key_returns_error_with_available_keys(self):
        hub = _make_hub()
        hub.blackboard.post("existing", "val", author="a")
        tools = build_comm_tools(hub)
        result = json.loads(tools["read_board"](key="nope"))
        assert "error" in result
        assert "existing" in result["available_keys"]


class TestToolFunctionMetadata:
    """Verify all tools have correct function metadata."""

    def test_send_message_has_correct_name(self):
        hub = _make_hub()
        tools = build_comm_tools(hub)
        assert tools["send_message"].__name__ == "send_message"

    def test_send_message_has_correct_qualname(self):
        hub = _make_hub()
        tools = build_comm_tools(hub)
        assert tools["send_message"].__qualname__ == "send_message"

    def test_send_message_has_docstring(self):
        hub = _make_hub()
        tools = build_comm_tools(hub)
        assert tools["send_message"].__doc__ is not None
        assert "recipient agent" in tools["send_message"].__doc__

    def test_check_messages_has_correct_name(self):
        hub = _make_hub()
        tools = build_comm_tools(hub)
        assert tools["check_messages"].__name__ == "check_messages"

    def test_check_messages_has_correct_qualname(self):
        hub = _make_hub()
        tools = build_comm_tools(hub)
        assert tools["check_messages"].__qualname__ == "check_messages"

    def test_check_messages_has_docstring(self):
        hub = _make_hub()
        tools = build_comm_tools(hub)
        assert tools["check_messages"].__doc__ is not None
        assert "inbox" in tools["check_messages"].__doc__

    def test_post_to_board_has_correct_name(self):
        hub = _make_hub()
        tools = build_comm_tools(hub)
        assert tools["post_to_board"].__name__ == "post_to_board"

    def test_post_to_board_has_correct_qualname(self):
        hub = _make_hub()
        tools = build_comm_tools(hub)
        assert tools["post_to_board"].__qualname__ == "post_to_board"

    def test_post_to_board_has_docstring(self):
        hub = _make_hub()
        tools = build_comm_tools(hub)
        assert tools["post_to_board"].__doc__ is not None
        assert "blackboard" in tools["post_to_board"].__doc__

    def test_read_board_has_correct_name(self):
        hub = _make_hub()
        tools = build_comm_tools(hub)
        assert tools["read_board"].__name__ == "read_board"

    def test_read_board_has_correct_qualname(self):
        hub = _make_hub()
        tools = build_comm_tools(hub)
        assert tools["read_board"].__qualname__ == "read_board"

    def test_read_board_has_docstring(self):
        hub = _make_hub()
        tools = build_comm_tools(hub)
        assert tools["read_board"].__doc__ is not None
        assert "blackboard" in tools["read_board"].__doc__


class TestSendMessageEdgeCases:
    """Edge cases for send_message tool."""

    def test_empty_content(self):
        hub = _make_hub("a")
        tools = build_comm_tools(hub)
        result = json.loads(tools["send_message"](to="b", content=""))
        assert result["status"] == "sent"
        messages = hub.mailbox.receive("b")
        assert messages[0].content == ""

    def test_very_long_content(self):
        hub = _make_hub("a")
        tools = build_comm_tools(hub)
        long_content = "x" * 10000
        tools["send_message"](to="b", content=long_content)
        messages = hub.mailbox.receive("b")
        assert messages[0].content == long_content

    def test_special_chars_in_content(self):
        hub = _make_hub("a")
        tools = build_comm_tools(hub)
        special = "Line1\nLine2\t\rSpecial: 你好 🚀 <>&\"'"
        tools["send_message"](to="b", content=special)
        messages = hub.mailbox.receive("b")
        assert messages[0].content == special

    def test_special_chars_in_recipient_name(self):
        hub = _make_hub("a")
        tools = build_comm_tools(hub)
        recipient = "agent-123_test.v2"
        tools["send_message"](to=recipient, content="hi")
        messages = hub.mailbox.receive(recipient)
        assert messages[0].recipient == recipient

    def test_send_to_self(self):
        hub = _make_hub("alice")
        tools = build_comm_tools(hub)
        tools["send_message"](to="alice", content="note to self")
        messages = hub.mailbox.receive("alice")
        assert len(messages) == 1
        assert messages[0].sender == "alice"
        assert messages[0].recipient == "alice"

    def test_metadata_with_nested_json(self):
        hub = _make_hub("a")
        tools = build_comm_tools(hub)
        meta = '{"data": {"nested": [1, 2, 3]}, "flag": true}'
        tools["send_message"](to="b", content="hi", metadata=meta)
        messages = hub.mailbox.receive("b")
        assert messages[0].metadata == {
            "data": {"nested": [1, 2, 3]},
            "flag": True,
        }

    def test_metadata_with_unicode(self):
        hub = _make_hub("a")
        tools = build_comm_tools(hub)
        meta = '{"emoji": "🎉", "lang": "中文"}'
        tools["send_message"](to="b", content="hi", metadata=meta)
        messages = hub.mailbox.receive("b")
        assert messages[0].metadata == {"emoji": "🎉", "lang": "中文"}

    def test_invalid_json_metadata_raises_error(self):
        hub = _make_hub("a")
        tools = build_comm_tools(hub)
        with pytest.raises(json.JSONDecodeError):
            tools["send_message"](to="b", content="hi", metadata="{invalid")

    def test_multiple_sends_same_recipient(self):
        hub = _make_hub("alice")
        tools = build_comm_tools(hub)
        tools["send_message"](to="bob", content="msg1")
        tools["send_message"](to="bob", content="msg2")
        tools["send_message"](to="bob", content="msg3")
        messages = hub.mailbox.receive("bob")
        assert len(messages) == 3
        assert messages[0].content == "msg1"
        assert messages[1].content == "msg2"
        assert messages[2].content == "msg3"

    def test_return_value_is_valid_json(self):
        hub = _make_hub("a")
        tools = build_comm_tools(hub)
        result = tools["send_message"](to="b", content="hi")
        parsed = json.loads(result)
        assert "status" in parsed
        assert "message_id" in parsed
        assert "to" in parsed

    def test_return_value_contains_message_id(self):
        hub = _make_hub("a")
        tools = build_comm_tools(hub)
        result = json.loads(tools["send_message"](to="b", content="hi"))
        assert isinstance(result["message_id"], str)
        assert len(result["message_id"]) > 0


class TestCheckMessagesEdgeCases:
    """Edge cases for check_messages tool."""

    def test_peek_then_consume_sequence(self):
        hub = _make_hub("bob")
        hub.mailbox.send("alice", "bob", "msg1")
        tools = build_comm_tools(hub)
        peek_result = json.loads(tools["check_messages"](peek=True))
        assert peek_result["count"] == 1
        consume_result = json.loads(tools["check_messages"](peek=False))
        assert consume_result["count"] == 1
        final = json.loads(tools["check_messages"](peek=False))
        assert final["count"] == 0

    def test_multiple_peeks_preserve_queue(self):
        hub = _make_hub("bob")
        hub.mailbox.send("alice", "bob", "msg")
        tools = build_comm_tools(hub)
        tools["check_messages"](peek=True)
        tools["check_messages"](peek=True)
        tools["check_messages"](peek=True)
        result = json.loads(tools["check_messages"](peek=True))
        assert result["count"] == 1

    def test_multiple_messages_from_different_senders(self):
        hub = _make_hub("bob")
        hub.mailbox.send("alice", "bob", "from alice")
        hub.mailbox.send("charlie", "bob", "from charlie")
        hub.mailbox.send("dave", "bob", "from dave")
        tools = build_comm_tools(hub)
        result = json.loads(tools["check_messages"]())
        assert result["count"] == 3
        senders = {msg["sender"] for msg in result["messages"]}
        assert senders == {"alice", "charlie", "dave"}

    def test_messages_include_all_fields(self):
        hub = _make_hub("bob")
        hub.mailbox.send("alice", "bob", "hi", metadata={"key": "val"})
        tools = build_comm_tools(hub)
        result = json.loads(tools["check_messages"]())
        msg = result["messages"][0]
        assert "id" in msg
        assert "sender" in msg
        assert "recipient" in msg
        assert "content" in msg
        assert "metadata" in msg
        assert "timestamp" in msg

    def test_return_value_is_valid_json(self):
        hub = _make_hub("bob")
        tools = build_comm_tools(hub)
        result = tools["check_messages"]()
        parsed = json.loads(result)
        assert "messages" in parsed
        assert "count" in parsed

    def test_empty_inbox_count_is_zero(self):
        hub = _make_hub("bob")
        tools = build_comm_tools(hub)
        result = json.loads(tools["check_messages"]())
        assert result["count"] == 0


class TestPostToBoardEdgeCases:
    """Edge cases for post_to_board tool."""

    def test_empty_string_value(self):
        hub = _make_hub("a")
        tools = build_comm_tools(hub)
        tools["post_to_board"](key="empty", value="")
        entry = hub.blackboard.read("empty")
        assert entry is not None
        assert entry.value == ""

    def test_very_long_value(self):
        hub = _make_hub("a")
        tools = build_comm_tools(hub)
        long_val = "x" * 10000
        tools["post_to_board"](key="long", value=long_val)
        entry = hub.blackboard.read("long")
        assert entry is not None
        assert entry.value == long_val

    def test_special_chars_in_key(self):
        hub = _make_hub("a")
        tools = build_comm_tools(hub)
        key = "key-123_test.v2:special"
        tools["post_to_board"](key=key, value="val")
        entry = hub.blackboard.read(key)
        assert entry is not None
        assert entry.key == key

    def test_special_chars_in_value(self):
        hub = _make_hub("a")
        tools = build_comm_tools(hub)
        special = "Line1\nLine2\t\rSpecial: 你好 🚀 <>&\"'"
        tools["post_to_board"](key="k", value=special)
        entry = hub.blackboard.read("k")
        assert entry is not None
        assert entry.value == special

    def test_json_array_value(self):
        hub = _make_hub("a")
        tools = build_comm_tools(hub)
        tools["post_to_board"](key="arr", value='[1, 2, 3, "four"]')
        entry = hub.blackboard.read("arr")
        assert entry is not None
        assert entry.value == [1, 2, 3, "four"]

    def test_json_null_value(self):
        hub = _make_hub("a")
        tools = build_comm_tools(hub)
        tools["post_to_board"](key="null", value="null")
        entry = hub.blackboard.read("null")
        assert entry is not None
        assert entry.value is None

    def test_json_boolean_value(self):
        hub = _make_hub("a")
        tools = build_comm_tools(hub)
        tools["post_to_board"](key="flag", value="true")
        entry = hub.blackboard.read("flag")
        assert entry is not None
        assert entry.value is True

    def test_json_number_value(self):
        hub = _make_hub("a")
        tools = build_comm_tools(hub)
        tools["post_to_board"](key="num", value="42")
        entry = hub.blackboard.read("num")
        assert entry is not None
        assert entry.value == 42

    def test_invalid_json_stored_as_string(self):
        hub = _make_hub("a")
        tools = build_comm_tools(hub)
        invalid = "{not valid json"
        tools["post_to_board"](key="k", value=invalid)
        entry = hub.blackboard.read("k")
        assert entry is not None
        assert entry.value == invalid

    def test_overwrite_existing_key(self):
        hub = _make_hub("a")
        tools = build_comm_tools(hub)
        tools["post_to_board"](key="k", value="v1")
        tools["post_to_board"](key="k", value="v2")
        entry = hub.blackboard.read("k")
        assert entry is not None
        assert entry.value == "v2"

    def test_different_agents_can_post_same_key(self):
        hub = _make_hub("alice")
        tools_alice = build_comm_tools(hub)
        tools_alice["post_to_board"](key="k", value="from alice")
        hub.agent_context.current_agent = "bob"
        tools_bob = build_comm_tools(hub)
        tools_bob["post_to_board"](key="k", value="from bob")
        entry = hub.blackboard.read("k")
        assert entry is not None
        assert entry.value == "from bob"
        assert entry.author == "bob"

    def test_return_value_is_valid_json(self):
        hub = _make_hub("a")
        tools = build_comm_tools(hub)
        result = tools["post_to_board"](key="k", value="v")
        parsed = json.loads(result)
        assert "status" in parsed
        assert "key" in parsed

    def test_return_value_contains_key(self):
        hub = _make_hub("a")
        tools = build_comm_tools(hub)
        result = json.loads(tools["post_to_board"](key="mykey", value="v"))
        assert result["key"] == "mykey"


class TestReadBoardEdgeCases:
    """Edge cases for read_board tool."""

    def test_read_all_with_empty_board(self):
        hub = _make_hub()
        tools = build_comm_tools(hub)
        result = json.loads(tools["read_board"]())
        assert result == {}

    def test_read_specific_key_from_empty_board(self):
        hub = _make_hub()
        tools = build_comm_tools(hub)
        result = json.loads(tools["read_board"](key="missing"))
        assert "error" in result
        assert result["available_keys"] == []

    def test_read_all_returns_all_entries(self):
        hub = _make_hub()
        hub.blackboard.post("k1", "v1", author="a")
        hub.blackboard.post("k2", "v2", author="b")
        hub.blackboard.post("k3", "v3", author="c")
        tools = build_comm_tools(hub)
        result = json.loads(tools["read_board"]())
        assert len(result) == 3
        assert "k1" in result
        assert "k2" in result
        assert "k3" in result

    def test_read_specific_key_returns_all_fields(self):
        hub = _make_hub()
        hub.blackboard.post("k", "v", author="alice")
        tools = build_comm_tools(hub)
        result = json.loads(tools["read_board"](key="k"))
        assert "key" in result
        assert "value" in result
        assert "author" in result
        assert "timestamp" in result

    def test_read_all_with_key_none(self):
        hub = _make_hub()
        hub.blackboard.post("k", "v", author="a")
        tools = build_comm_tools(hub)
        result = json.loads(tools["read_board"](key=None))
        assert "k" in result

    def test_missing_key_error_message_format(self):
        hub = _make_hub()
        hub.blackboard.post("existing", "val", author="a")
        tools = build_comm_tools(hub)
        result = json.loads(tools["read_board"](key="nope"))
        assert "Key 'nope' not found" in result["error"]

    def test_return_value_is_valid_json(self):
        hub = _make_hub()
        hub.blackboard.post("k", "v", author="a")
        tools = build_comm_tools(hub)
        result = tools["read_board"](key="k")
        parsed = json.loads(result)
        assert isinstance(parsed, dict)


class TestToolIntegration:
    """Integration scenarios involving multiple tools."""

    def test_post_and_read_round_trip(self):
        hub = _make_hub("alice")
        tools = build_comm_tools(hub)
        tools["post_to_board"](key="data", value='{"score": 100}')
        result = json.loads(tools["read_board"](key="data"))
        assert result["value"] == {"score": 100}
        assert result["author"] == "alice"

    def test_send_and_check_round_trip(self):
        hub = _make_hub("alice")
        tools = build_comm_tools(hub)
        send_result = json.loads(tools["send_message"](to="bob", content="hello bob"))
        message_id = send_result["message_id"]
        hub.agent_context.current_agent = "bob"
        tools_bob = build_comm_tools(hub)
        check_result = json.loads(tools_bob["check_messages"]())
        assert check_result["count"] == 1
        assert check_result["messages"][0]["id"] == message_id

    def test_multiple_agents_posting_reading_board(self):
        hub = _make_hub("alice")
        tools_alice = build_comm_tools(hub)
        tools_alice["post_to_board"](key="alice_data", value="from alice")
        hub.agent_context.current_agent = "bob"
        tools_bob = build_comm_tools(hub)
        tools_bob["post_to_board"](key="bob_data", value="from bob")
        result = json.loads(tools_bob["read_board"]())
        assert "alice_data" in result
        assert "bob_data" in result
        assert result["alice_data"]["author"] == "alice"
        assert result["bob_data"]["author"] == "bob"

    def test_broadcast_pattern(self):
        hub = _make_hub("orchestrator")
        tools = build_comm_tools(hub)
        for recipient in ["agent1", "agent2", "agent3"]:
            tools["send_message"](to=recipient, content="start")
        for agent in ["agent1", "agent2", "agent3"]:
            hub.agent_context.current_agent = agent
            agent_tools = build_comm_tools(hub)
            result = json.loads(agent_tools["check_messages"]())
            assert result["count"] == 1
            assert result["messages"][0]["content"] == "start"
