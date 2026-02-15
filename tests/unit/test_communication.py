import json

import pytest

from perm_agent.communication import (
    AgentContext,
    Blackboard,
    BoardEntry,
    CommunicationHub,
    Mailbox,
    Message,
)


class TestMessage:
    def test_frozen_immutable(self):
        msg = Message(id="1", sender="a", recipient="b", content="hi")
        with pytest.raises(AttributeError):
            msg.sender = "c"  # type: ignore[misc]

    def test_to_dict_serializable(self):
        msg = Message(id="1", sender="a", recipient="b", content="hi", metadata={"k": "v"})
        d = msg.to_dict()
        assert d["id"] == "1"
        assert d["sender"] == "a"
        assert d["recipient"] == "b"
        assert d["content"] == "hi"
        assert d["metadata"] == {"k": "v"}
        assert isinstance(d["timestamp"], float)
        json.dumps(d)  # must be JSON-serializable

    def test_default_metadata_empty(self):
        msg = Message(id="1", sender="a", recipient="b", content="hi")
        assert msg.metadata == {}

    def test_empty_string_content(self):
        msg = Message(id="1", sender="a", recipient="b", content="")
        assert msg.content == ""

    def test_very_long_content(self):
        long_text = "x" * 10_000
        msg = Message(id="1", sender="a", recipient="b", content=long_text)
        assert len(msg.content) == 10_000

    def test_special_characters_in_content(self):
        special = "Hello\n\tWorld\r\n🚀\\"
        msg = Message(id="1", sender="a", recipient="b", content=special)
        assert msg.content == special

    def test_timestamp_is_reasonable(self):
        msg = Message(id="1", sender="a", recipient="b", content="hi")
        assert msg.timestamp > 0
        assert msg.timestamp < 1e12

    def test_nested_metadata(self):
        meta = {"level1": {"level2": {"level3": "deep"}}}
        msg = Message(id="1", sender="a", recipient="b", content="hi", metadata=meta)
        assert msg.metadata["level1"]["level2"]["level3"] == "deep"

    def test_to_dict_json_roundtrip(self):
        msg = Message(id="1", sender="a", recipient="b", content="hi", metadata={"k": 1})
        d = msg.to_dict()
        json_str = json.dumps(d)
        restored = json.loads(json_str)
        assert restored["id"] == "1"
        assert restored["content"] == "hi"


class TestBoardEntry:
    def test_frozen_immutable(self):
        entry = BoardEntry(key="k", value="v", author="a")
        with pytest.raises(AttributeError):
            entry.key = "x"  # type: ignore[misc]

    def test_to_dict_serializable(self):
        entry = BoardEntry(key="k", value={"nested": True}, author="a")
        d = entry.to_dict()
        assert d["key"] == "k"
        assert d["value"] == {"nested": True}
        assert d["author"] == "a"
        assert isinstance(d["timestamp"], float)
        json.dumps(d)

    def test_timestamp_is_reasonable(self):
        entry = BoardEntry(key="k", value="v", author="a")
        assert entry.timestamp > 0
        assert entry.timestamp < 1e12

    def test_empty_string_key(self):
        entry = BoardEntry(key="", value="v", author="a")
        assert entry.key == ""

    def test_none_value(self):
        entry = BoardEntry(key="k", value=None, author="a")
        assert entry.value is None

    def test_complex_value_types(self):
        value = [1, "two", {"three": 3}, [4, 5]]
        entry = BoardEntry(key="k", value=value, author="a")
        assert entry.value == value


class TestBlackboard:
    def test_post_and_read(self):
        bb = Blackboard()
        bb.post("findings", "important data", author="researcher")
        entry = bb.read("findings")
        assert entry is not None
        assert entry.value == "important data"
        assert entry.author == "researcher"

    def test_read_nonexistent_returns_none(self):
        bb = Blackboard()
        assert bb.read("nope") is None

    def test_post_overwrites_same_key(self):
        bb = Blackboard()
        bb.post("status", "draft", author="a")
        bb.post("status", "final", author="b")
        entry = bb.read("status")
        assert entry is not None
        assert entry.value == "final"
        assert entry.author == "b"

    def test_read_all_returns_all_entries(self):
        bb = Blackboard()
        bb.post("a", 1, author="x")
        bb.post("b", 2, author="y")
        all_entries = bb.read_all()
        assert set(all_entries.keys()) == {"a", "b"}

    def test_keys(self):
        bb = Blackboard()
        bb.post("x", 1, author="a")
        bb.post("y", 2, author="a")
        assert set(bb.keys()) == {"x", "y"}

    def test_history_tracks_all_writes(self):
        bb = Blackboard()
        bb.post("k", "v1", author="a")
        bb.post("k", "v2", author="b")
        assert len(bb.history) == 2
        assert bb.history[0].value == "v1"
        assert bb.history[1].value == "v2"

    def test_to_dict_serializable(self):
        bb = Blackboard()
        bb.post("key", "value", author="agent")
        d = bb.to_dict()
        assert "key" in d
        json.dumps(d)

    def test_empty_blackboard_read_all(self):
        bb = Blackboard()
        assert bb.read_all() == {}

    def test_empty_blackboard_keys(self):
        bb = Blackboard()
        assert bb.keys() == []

    def test_empty_blackboard_history(self):
        bb = Blackboard()
        assert bb.history == []

    def test_read_all_returns_copy(self):
        bb = Blackboard()
        bb.post("k", "v", author="a")
        all1 = bb.read_all()
        all2 = bb.read_all()
        assert all1 is not all2
        assert all1 == all2

    def test_history_returns_copy(self):
        bb = Blackboard()
        bb.post("k", "v", author="a")
        hist1 = bb.history
        hist2 = bb.history
        assert hist1 is not hist2

    def test_keys_returns_copy(self):
        bb = Blackboard()
        bb.post("k", "v", author="a")
        keys1 = bb.keys()
        keys2 = bb.keys()
        assert keys1 is not keys2

    def test_many_overwrites_history_preserves_order(self):
        bb = Blackboard()
        for i in range(100):
            bb.post("counter", i, author=f"agent{i}")
        assert len(bb.history) == 100
        assert bb.history[0].value == 0
        assert bb.history[99].value == 99

    def test_history_timestamps_increase(self):
        bb = Blackboard()
        bb.post("k1", "v1", author="a")
        bb.post("k2", "v2", author="b")
        bb.post("k3", "v3", author="c")
        assert bb.history[0].timestamp <= bb.history[1].timestamp
        assert bb.history[1].timestamp <= bb.history[2].timestamp

    def test_post_empty_string_key(self):
        bb = Blackboard()
        bb.post("", "value", author="a")
        entry = bb.read("")
        assert entry is not None
        assert entry.value == "value"

    def test_post_none_value(self):
        bb = Blackboard()
        bb.post("k", None, author="a")
        entry = bb.read("k")
        assert entry is not None
        assert entry.value is None

    def test_to_dict_empty_blackboard(self):
        bb = Blackboard()
        d = bb.to_dict()
        assert d == {}

    def test_to_dict_after_overwrites(self):
        bb = Blackboard()
        bb.post("k", "v1", author="a")
        bb.post("k", "v2", author="b")
        d = bb.to_dict()
        assert len(d) == 1
        assert d["k"]["value"] == "v2"


class TestMailbox:
    def test_send_and_receive(self):
        mb = Mailbox()
        mb.send("alice", "bob", "hello")
        messages = mb.receive("bob")
        assert len(messages) == 1
        assert messages[0].sender == "alice"
        assert messages[0].content == "hello"

    def test_receive_empties_queue(self):
        mb = Mailbox()
        mb.send("a", "b", "msg")
        mb.receive("b")
        assert mb.receive("b") == []

    def test_receive_empty_returns_empty_list(self):
        mb = Mailbox()
        assert mb.receive("nobody") == []

    def test_peek_does_not_consume(self):
        mb = Mailbox()
        mb.send("a", "b", "msg")
        peeked = mb.peek("b")
        assert len(peeked) == 1
        # Still there after peek
        assert mb.pending_count("b") == 1

    def test_pending_count(self):
        mb = Mailbox()
        assert mb.pending_count("b") == 0
        mb.send("a", "b", "1")
        mb.send("a", "b", "2")
        assert mb.pending_count("b") == 2

    def test_multiple_recipients_isolated(self):
        mb = Mailbox()
        mb.send("a", "bob", "for bob")
        mb.send("a", "carol", "for carol")
        assert len(mb.receive("bob")) == 1
        assert len(mb.receive("carol")) == 1

    def test_fifo_order_preserved(self):
        mb = Mailbox()
        mb.send("a", "b", "first")
        mb.send("a", "b", "second")
        mb.send("a", "b", "third")
        messages = mb.receive("b")
        assert [m.content for m in messages] == ["first", "second", "third"]

    def test_all_messages_returns_complete_history(self):
        mb = Mailbox()
        mb.send("a", "b", "1")
        mb.send("c", "d", "2")
        assert len(mb.all_messages) == 2

    def test_send_with_metadata(self):
        mb = Mailbox()
        msg = mb.send("a", "b", "hello", metadata={"priority": "high"})
        assert msg.metadata == {"priority": "high"}

    def test_message_id_is_generated(self):
        mb = Mailbox()
        msg = mb.send("a", "b", "hi")
        assert len(msg.id) == 12

    def test_message_ids_are_unique(self):
        mb = Mailbox()
        ids = set()
        for i in range(100):
            msg = mb.send("a", "b", f"msg{i}")
            ids.add(msg.id)
        assert len(ids) == 100

    def test_empty_string_content(self):
        mb = Mailbox()
        msg = mb.send("a", "b", "")
        assert msg.content == ""

    def test_very_long_content(self):
        mb = Mailbox()
        long_text = "x" * 10_000
        msg = mb.send("a", "b", long_text)
        assert len(msg.content) == 10_000

    def test_peek_returns_copy(self):
        mb = Mailbox()
        mb.send("a", "b", "msg")
        peek1 = mb.peek("b")
        peek2 = mb.peek("b")
        assert peek1 is not peek2

    def test_all_messages_returns_copy(self):
        mb = Mailbox()
        mb.send("a", "b", "msg")
        all1 = mb.all_messages
        all2 = mb.all_messages
        assert all1 is not all2

    def test_pending_count_after_receive(self):
        mb = Mailbox()
        mb.send("a", "b", "1")
        mb.send("a", "b", "2")
        mb.receive("b")
        assert mb.pending_count("b") == 0

    def test_pending_count_nonexistent_agent(self):
        mb = Mailbox()
        assert mb.pending_count("nobody") == 0

    def test_peek_empty_queue(self):
        mb = Mailbox()
        assert mb.peek("nobody") == []

    def test_all_messages_persists_after_receive(self):
        mb = Mailbox()
        mb.send("a", "b", "msg")
        mb.receive("b")
        assert len(mb.all_messages) == 1

    def test_multiple_senders_to_same_recipient(self):
        mb = Mailbox()
        mb.send("alice", "bob", "from alice")
        mb.send("carol", "bob", "from carol")
        mb.send("dave", "bob", "from dave")
        messages = mb.receive("bob")
        assert len(messages) == 3
        senders = {m.sender for m in messages}
        assert senders == {"alice", "carol", "dave"}

    def test_rapid_send_receive_cycles(self):
        mb = Mailbox()
        for i in range(10):
            mb.send("a", "b", f"msg{i}")
            msgs = mb.receive("b")
            assert len(msgs) == 1
            assert msgs[0].content == f"msg{i}"

    def test_isolation_agent_a_never_sees_agent_b_messages(self):
        mb = Mailbox()
        mb.send("sender", "agent_a", "for a")
        mb.send("sender", "agent_b", "for b")
        a_msgs = mb.receive("agent_a")
        b_msgs = mb.receive("agent_b")
        assert len(a_msgs) == 1
        assert len(b_msgs) == 1
        assert a_msgs[0].content == "for a"
        assert b_msgs[0].content == "for b"

    def test_message_timestamps_increase(self):
        mb = Mailbox()
        msg1 = mb.send("a", "b", "1")
        msg2 = mb.send("a", "b", "2")
        msg3 = mb.send("a", "b", "3")
        assert msg1.timestamp <= msg2.timestamp
        assert msg2.timestamp <= msg3.timestamp

    def test_send_returns_message_with_correct_fields(self):
        mb = Mailbox()
        msg = mb.send("alice", "bob", "hello", metadata={"k": "v"})
        assert msg.sender == "alice"
        assert msg.recipient == "bob"
        assert msg.content == "hello"
        assert msg.metadata == {"k": "v"}
        assert isinstance(msg.id, str)
        assert isinstance(msg.timestamp, float)


class TestAgentContext:
    def test_default_is_unknown(self):
        ctx = AgentContext()
        assert ctx.current_agent == "unknown"

    def test_mutable_update(self):
        ctx = AgentContext()
        ctx.current_agent = "researcher"
        assert ctx.current_agent == "researcher"


class TestCommunicationHub:
    def test_creates_blackboard_and_mailbox(self):
        hub = CommunicationHub()
        assert isinstance(hub.blackboard, Blackboard)
        assert isinstance(hub.mailbox, Mailbox)
        assert isinstance(hub.agent_context, AgentContext)

    def test_to_dict_includes_both(self):
        hub = CommunicationHub()
        hub.blackboard.post("k", "v", author="a")
        hub.mailbox.send("a", "b", "msg")
        d = hub.to_dict()
        assert "blackboard" in d
        assert "messages" in d
        assert len(d["messages"]) == 1
        json.dumps(d)

    def test_to_dict_empty_hub(self):
        hub = CommunicationHub()
        d = hub.to_dict()
        assert d["blackboard"] == {}
        assert d["messages"] == []

    def test_to_dict_multiple_messages(self):
        hub = CommunicationHub()
        hub.mailbox.send("a", "b", "1")
        hub.mailbox.send("c", "d", "2")
        hub.mailbox.send("e", "f", "3")
        d = hub.to_dict()
        assert len(d["messages"]) == 3

    def test_to_dict_multiple_blackboard_entries(self):
        hub = CommunicationHub()
        hub.blackboard.post("k1", "v1", author="a")
        hub.blackboard.post("k2", "v2", author="b")
        hub.blackboard.post("k3", "v3", author="c")
        d = hub.to_dict()
        assert len(d["blackboard"]) == 3

    def test_agent_context_integration(self):
        hub = CommunicationHub()
        hub.agent_context.current_agent = "researcher"
        assert hub.agent_context.current_agent == "researcher"

    def test_to_dict_json_serializable(self):
        hub = CommunicationHub()
        hub.blackboard.post("k", {"nested": "value"}, author="a")
        hub.mailbox.send("a", "b", "msg", metadata={"priority": 1})
        d = hub.to_dict()
        json_str = json.dumps(d)
        restored = json.loads(json_str)
        assert restored["blackboard"]["k"]["value"]["nested"] == "value"
        assert restored["messages"][0]["metadata"]["priority"] == 1

    def test_to_dict_does_not_include_agent_context(self):
        hub = CommunicationHub()
        hub.agent_context.current_agent = "test_agent"
        d = hub.to_dict()
        assert "agent_context" not in d
        assert "current_agent" not in d
