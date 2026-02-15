"""End-to-end tests for inter-agent communication with real Gemini API.

These tests exercise the full Blackboard + Mailbox communication stack
against a live LLM. They verify that agents can:
  1. Post data to the shared blackboard and another agent reads it
  2. Send directed messages via mailbox
  3. Use both blackboard and mailbox in a multi-agent pipeline
  4. Combine custom tools with communication tools
  5. Inspect hub state after pipeline completes
"""

from __future__ import annotations

import os

import pytest

from perm_agent import (
    Agent,
    CommunicationHub,
    Pipeline,
    tool,
)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL = "gemini/gemini-2.0-flash"

pytestmark = pytest.mark.skipif(
    not GEMINI_API_KEY, reason="GEMINI_API_KEY not set",
)


# ---------------------------------------------------------------------------
# Shared tools
# ---------------------------------------------------------------------------


@tool
def lookup_price(item: str) -> str:
    """Look up the price of an item.

    Args:
        item: Name of the item to look up.
    """
    prices = {
        "laptop": "1200",
        "phone": "800",
        "headphones": "150",
        "keyboard": "75",
    }
    return prices.get(item.lower(), "unknown")


@tool
def format_report(data: str) -> str:
    """Format raw data into a clean report line.

    Args:
        data: Raw data string to format.
    """
    return f"REPORT: {data}"


# ---------------------------------------------------------------------------
# 1. Blackboard — one agent posts, another reads
# ---------------------------------------------------------------------------


class TestBlackboardE2E:
    def test_agent_posts_and_next_reads_board(self) -> None:
        researcher = Agent(
            name="researcher",
            model=MODEL,
            instructions=(
                "You are a researcher. Your task is to post a finding "
                "to the shared blackboard using the post_to_board tool. "
                "Post the key 'finding' with value 'Python was created "
                "by Guido van Rossum'. Then say 'Posted'."
            ),
            max_iterations=3,
        )
        writer = Agent(
            name="writer",
            model=MODEL,
            instructions=(
                "You are a writer. Read the key 'finding' from the "
                "shared blackboard using the read_board tool. "
                "Then write a single sentence summary that includes "
                "the information you read. Start with 'Summary:'."
            ),
            max_iterations=3,
        )

        p = Pipeline("board_e2e", enable_communication=True)
        p.add_step(researcher, output_path="/research")
        p.add_step(
            writer,
            input_map={"input": "Write a summary"},
            output_path="/summary",
        )
        result = p.run({"input": "Research Python"})

        assert p.communication is not None
        entry = p.communication.blackboard.read("finding")
        assert entry is not None
        assert "guido" in entry.value.lower() or "rossum" in entry.value.lower()
        assert entry.author == "researcher"

        assert isinstance(result.get("summary"), str)
        assert len(result["summary"]) > 10


# ---------------------------------------------------------------------------
# 2. Mailbox — directed message between agents
# ---------------------------------------------------------------------------


class TestMailboxE2E:
    def test_agent_sends_message_and_next_receives(self) -> None:
        sender = Agent(
            name="dispatcher",
            model=MODEL,
            instructions=(
                "You are a task dispatcher. Send a message to 'worker' "
                "using the send_message tool. The message content "
                "should be: 'Please process order #42'. "
                "Then say 'Dispatched'."
            ),
            max_iterations=3,
        )
        receiver = Agent(
            name="worker",
            model=MODEL,
            instructions=(
                "You are a worker. Check your messages using the "
                "check_messages tool. Read the message you received. "
                "Then confirm what order number was mentioned."
            ),
            max_iterations=3,
        )

        p = Pipeline("mailbox_e2e", enable_communication=True)
        p.add_step(sender, output_path="/dispatch")
        p.add_step(
            receiver,
            input_map={"input": "Check your tasks"},
            output_path="/confirmation",
        )
        result = p.run({"input": "Start dispatch"})

        hub = p.communication
        assert hub is not None
        assert len(hub.mailbox.all_messages) == 1
        msg = hub.mailbox.all_messages[0]
        assert msg.sender == "dispatcher"
        assert msg.recipient == "worker"
        assert "42" in msg.content

        assert "42" in result.get("confirmation", "")


# ---------------------------------------------------------------------------
# 3. Combined blackboard + mailbox in multi-agent pipeline
# ---------------------------------------------------------------------------


class TestCombinedCommunicationE2E:
    def test_three_agent_pipeline_with_board_and_mailbox(self) -> None:
        analyst = Agent(
            name="analyst",
            model=MODEL,
            instructions=(
                "You are a data analyst. Do two things:\n"
                "1. Post to the blackboard using post_to_board with "
                "key='analysis' and value='Revenue increased by 15%'\n"
                "2. Send a message to 'reporter' using send_message "
                "saying 'Analysis is ready on the board'\n"
                "Then say 'Analysis complete'."
            ),
            max_iterations=5,
        )
        reviewer = Agent(
            name="reviewer",
            model=MODEL,
            instructions=(
                "You are a reviewer. Read the 'analysis' key from the "
                "blackboard using read_board. Then post your review "
                "to the blackboard with key='review' and "
                "value='Approved: numbers look correct'. "
                "Say 'Review complete'."
            ),
            max_iterations=5,
        )
        reporter = Agent(
            name="reporter",
            model=MODEL,
            instructions=(
                "You are a reporter. Do two things:\n"
                "1. Check your messages using check_messages\n"
                "2. Read ALL entries from the blackboard using "
                "read_board (without a key argument)\n"
                "Then summarize what you found."
            ),
            max_iterations=5,
        )

        p = Pipeline("combined_e2e", enable_communication=True)
        p.add_step(analyst, output_path="/step1")
        p.add_step(reviewer, output_path="/step2")
        p.add_step(
            reporter,
            input_map={"input": "Generate report"},
            output_path="/step3",
        )
        result = p.run({"input": "Start analysis pipeline"})

        hub = p.communication
        assert hub is not None

        # Blackboard should have both entries
        analysis = hub.blackboard.read("analysis")
        assert analysis is not None
        assert "15%" in str(analysis.value) or "15" in str(analysis.value)

        review = hub.blackboard.read("review")
        assert review is not None
        assert "approved" in str(review.value).lower()

        # Mailbox should have the message from analyst to reporter
        assert len(hub.mailbox.all_messages) >= 1
        msgs_to_reporter = [
            m for m in hub.mailbox.all_messages
            if m.recipient == "reporter"
        ]
        assert len(msgs_to_reporter) >= 1

        # All steps produced output
        assert result.get("step1")
        assert result.get("step2")
        assert result.get("step3")


# ---------------------------------------------------------------------------
# 4. Communication + custom tools together
# ---------------------------------------------------------------------------


class TestCommunicationWithToolsE2E:
    def test_agent_uses_custom_tool_then_posts_to_board(self) -> None:
        pricer = Agent(
            name="pricer",
            model=MODEL,
            instructions=(
                "Step 1: Call lookup_price with item='laptop'.\n"
                "Step 2: Call post_to_board with key='laptop_price' "
                "and value being the price you got.\n"
                "Step 3: Say the price."
            ),
            tools=[lookup_price],
            max_iterations=5,
        )
        reader = Agent(
            name="reader",
            model=MODEL,
            instructions=(
                "Call read_board with key='laptop_price'. "
                "Report the price you found."
            ),
            max_iterations=3,
        )

        p = Pipeline("tools_comm_e2e", enable_communication=True)
        p.add_step(pricer, output_path="/price_check")
        p.add_step(
            reader,
            input_map={"input": "Read the laptop price"},
            output_path="/report",
        )
        result = p.run({"input": "Check laptop price"})

        hub = p.communication
        assert hub is not None
        entry = hub.blackboard.read("laptop_price")
        assert entry is not None
        assert "1200" in str(entry.value)

        assert result.get("report")


# ---------------------------------------------------------------------------
# 5. Hub state inspection after pipeline
# ---------------------------------------------------------------------------


class TestHubInspectionE2E:
    def test_hub_tracks_all_communication(self) -> None:
        agent_a = Agent(
            name="agent_a",
            model=MODEL,
            instructions=(
                "Do exactly these steps:\n"
                "1. Post to blackboard: key='status', value='started'\n"
                "2. Send message to 'agent_b': 'Hello from A'\n"
                "Say 'A done'."
            ),
            max_iterations=5,
        )
        agent_b = Agent(
            name="agent_b",
            model=MODEL,
            instructions=(
                "Do exactly these steps:\n"
                "1. Check your messages using check_messages\n"
                "2. Read 'status' from the blackboard using read_board\n"
                "3. Post to blackboard: key='status', value='completed'\n"
                "Say 'B done'."
            ),
            max_iterations=5,
        )

        p = Pipeline("inspect_e2e", enable_communication=True)
        p.add_step(agent_a, output_path="/a")
        p.add_step(
            agent_b,
            input_map={"input": "Continue work"},
            output_path="/b",
        )
        result = p.run({"input": "Start"})

        hub = p.communication
        assert hub is not None
        assert isinstance(hub, CommunicationHub)

        # Blackboard state: 'status' should be 'completed' (overwritten)
        status = hub.blackboard.read("status")
        assert status is not None
        assert status.value == "completed"
        assert status.author == "agent_b"

        # Blackboard history should have 2 entries for 'status'
        history = [
            e for e in hub.blackboard.history if e.key == "status"
        ]
        assert len(history) == 2
        assert history[0].value == "started"
        assert history[1].value == "completed"

        # Mailbox: one message sent
        assert len(hub.mailbox.all_messages) == 1
        msg = hub.mailbox.all_messages[0]
        assert msg.sender == "agent_a"
        assert msg.recipient == "agent_b"

        # Agent context should track last agent
        assert hub.agent_context.current_agent == "agent_b"

        # to_dict should be serializable
        import json
        serialized = json.dumps(hub.to_dict())
        assert len(serialized) > 0

        # Both steps produced results
        assert result.get("a")
        assert result.get("b")
