"""Integration tests for inter-agent communication via Pipeline."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from perm_agent import (
    Agent,
    CommunicationHub,
    Pipeline,
    build_agent_engine,
)
from perm_agent.comm_tools import COMM_TOOL_NAMES


def _make_text_response(content: str) -> MagicMock:
    response = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    choice.message.tool_calls = None
    response.choices = [choice]
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 5
    return response


def _make_tool_response(tool_calls: list[dict]) -> MagicMock:
    response = MagicMock()
    choice = MagicMock()
    choice.message.content = None
    calls = []
    for tc in tool_calls:
        call = MagicMock()
        call.id = tc["id"]
        call.function.name = tc["name"]
        call.function.arguments = json.dumps(tc["args"])
        calls.append(call)
    choice.message.tool_calls = calls
    choice.message.model_dump.return_value = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc["args"]),
                },
            }
            for tc in tool_calls
        ],
    }
    response.choices = [choice]
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 5
    return response


class TestPipelineWithoutCommunication:
    """Backward compatibility: existing pipelines unaffected."""

    @patch("litellm.completion")
    def test_pipeline_works_without_communication(self, mock_completion):
        mock_completion.return_value = _make_text_response("done")
        agent = Agent(name="worker", model="openai/gpt-4o", instructions="Work")

        p = Pipeline("basic")
        p.add_step(agent, output_path="/result")
        result = p.run({"input": "go"})

        assert result["result"] == "done"
        assert p.communication is None


class TestPipelineCommunicationEnabled:
    @patch("litellm.completion")
    def test_hub_created_when_enabled(self, mock_completion):
        mock_completion.return_value = _make_text_response("ok")
        agent = Agent(name="a1", model="openai/gpt-4o", instructions="Do")

        p = Pipeline("comm", enable_communication=True)
        p.add_step(agent, output_path="/r")
        p.run({"input": "go"})

        assert p.communication is not None
        assert isinstance(p.communication, CommunicationHub)

    @patch("litellm.completion")
    def test_comm_tools_in_schema_when_enabled(self, mock_completion):
        """Comm tool names should appear in the step spec."""
        agent = Agent(name="talker", model="openai/gpt-4o", instructions="Talk")

        p = Pipeline("comm", enable_communication=True)
        p.add_step(agent, output_path="/r")

        spec = p.to_spec()
        agent_step = [s for s in spec if s.get("op") == "agent_loop"][0]
        for tool_name in COMM_TOOL_NAMES:
            assert tool_name in agent_step["tools"]

    @patch("litellm.completion")
    def test_agent_posts_to_board_then_next_reads(self, mock_completion):
        """First agent posts to blackboard, second agent reads it."""
        mock_completion.side_effect = [
            # Agent 1: calls post_to_board
            _make_tool_response([{
                "id": "c1",
                "name": "post_to_board",
                "args": {"key": "findings", "value": "important data"},
            }]),
            # Agent 1: finishes with text
            _make_text_response("Posted findings"),
            # Agent 2: calls read_board
            _make_tool_response([{
                "id": "c2",
                "name": "read_board",
                "args": {"key": "findings"},
            }]),
            # Agent 2: finishes with text
            _make_text_response("Read the findings"),
        ]

        researcher = Agent(name="researcher", model="openai/gpt-4o", instructions="Research")
        writer = Agent(name="writer", model="openai/gpt-4o", instructions="Write")

        p = Pipeline("research_write", enable_communication=True)
        p.add_step(researcher, output_path="/research")
        p.add_step(writer, input_map={"input": "${@:/research}"}, output_path="/report")
        result = p.run({"input": "analyze topic"})

        assert result["research"] == "Posted findings"
        assert result["report"] == "Read the findings"
        # Verify blackboard has the data
        entry = p.communication.blackboard.read("findings")
        assert entry is not None
        assert entry.value == "important data"

    @patch("litellm.completion")
    def test_agent_sends_message_then_next_checks(self, mock_completion):
        """First agent sends a message, second agent checks inbox."""
        mock_completion.side_effect = [
            # Agent 1: sends message
            _make_tool_response([{
                "id": "c1",
                "name": "send_message",
                "args": {"to": "reviewer", "content": "please review draft"},
            }]),
            _make_text_response("Sent review request"),
            # Agent 2: checks messages
            _make_tool_response([{
                "id": "c2",
                "name": "check_messages",
                "args": {},
            }]),
            _make_text_response("Reviewed the draft"),
        ]

        drafter = Agent(name="drafter", model="openai/gpt-4o", instructions="Draft")
        reviewer = Agent(name="reviewer", model="openai/gpt-4o", instructions="Review")

        p = Pipeline("draft_review", enable_communication=True)
        p.add_step(drafter, output_path="/draft")
        p.add_step(reviewer, input_map={"input": "${@:/draft}"}, output_path="/review")
        result = p.run({"input": "write article"})

        assert result["draft"] == "Sent review request"
        assert result["review"] == "Reviewed the draft"

    @patch("litellm.completion")
    def test_agent_name_tracked_per_step(self, mock_completion):
        """Each step updates agent_context.current_agent."""
        mock_completion.side_effect = [
            _make_text_response("first done"),
            _make_text_response("second done"),
        ]

        a1 = Agent(name="alpha", model="openai/gpt-4o", instructions="First")
        a2 = Agent(name="beta", model="openai/gpt-4o", instructions="Second")

        p = Pipeline("tracking", enable_communication=True)
        p.add_step(a1, output_path="/r1")
        p.add_step(a2, output_path="/r2")
        p.run({"input": "go"})

        # After pipeline, the last agent_name should be "beta"
        assert p.communication.agent_context.current_agent == "beta"


class TestBuildAgentEngineWithCommunication:
    def test_comm_tools_registered_in_engine(self):
        hub = CommunicationHub()
        hub.agent_context.current_agent = "test"
        engine = build_agent_engine(communication=hub)

        spec = [
            {
                "op": "tool",
                "name": "post_to_board",
                "args": {"key": "k", "value": "v"},
                "path": "/r",
            },
        ]
        result = engine.apply(spec, source={}, dest={})
        parsed = json.loads(result["r"])
        assert parsed["status"] == "posted"

    def test_regular_tools_coexist_with_comm_tools(self):
        def my_tool(x: str) -> str:
            """Custom tool."""
            return f"custom:{x}"

        hub = CommunicationHub()
        hub.agent_context.current_agent = "agent"
        engine = build_agent_engine(tools={"my_tool": my_tool}, communication=hub)

        spec = [
            {
                "op": "tool",
                "name": "my_tool",
                "args": {"x": "hello"},
                "path": "/custom",
            },
            {
                "op": "tool",
                "name": "post_to_board",
                "args": {"key": "test", "value": "data"},
                "path": "/comm",
            },
        ]
        result = engine.apply(spec, source={}, dest={})
        assert result["custom"] == "custom:hello"
        assert json.loads(result["comm"])["status"] == "posted"

    def test_no_communication_no_comm_tools(self):
        engine = build_agent_engine()

        spec = [
            {
                "op": "tool",
                "name": "post_to_board",
                "args": {"key": "k", "value": "v"},
                "path": "/r",
            },
        ]
        with pytest.raises(KeyError, match="post_to_board"):
            engine.apply(spec, source={}, dest={})


class TestThreeAgentCommunication:
    """Three-agent pipeline using blackboard and mailbox."""

    @patch("litellm.completion")
    def test_three_agents_share_board_data(self, mock_completion):
        """First agent posts, second reads and posts, third reads both."""
        mock_completion.side_effect = [
            # Agent 1: posts initial data
            _make_tool_response([{
                "id": "c1",
                "name": "post_to_board",
                "args": {"key": "step1", "value": "initial"},
            }]),
            _make_text_response("Step 1 complete"),
            # Agent 2: reads and posts
            _make_tool_response([
                {"id": "c2", "name": "read_board", "args": {"key": "step1"}},
            ]),
            _make_tool_response([{
                "id": "c3",
                "name": "post_to_board",
                "args": {"key": "step2", "value": "processed"},
            }]),
            _make_text_response("Step 2 complete"),
            # Agent 3: reads all
            _make_tool_response([
                {"id": "c4", "name": "read_board", "args": {}},
            ]),
            _make_text_response("Step 3 complete"),
        ]

        a1 = Agent(name="collector", model="openai/gpt-4o", instructions="")
        a2 = Agent(name="processor", model="openai/gpt-4o", instructions="")
        a3 = Agent(name="reporter", model="openai/gpt-4o", instructions="")

        p = Pipeline("three_step", enable_communication=True)
        p.add_step(a1, output_path="/r1")
        p.add_step(a2, output_path="/r2")
        p.add_step(a3, output_path="/r3")
        result = p.run({"input": "start"})

        assert result["r1"] == "Step 1 complete"
        assert result["r2"] == "Step 2 complete"
        assert result["r3"] == "Step 3 complete"
        assert p.communication.blackboard.read("step1").value == "initial"
        assert p.communication.blackboard.read("step2").value == "processed"

    @patch("litellm.completion")
    def test_three_agents_mailbox_chain(self, mock_completion):
        """Agent 1 sends to 2, agent 2 sends to 3."""
        mock_completion.side_effect = [
            # Agent 1: sends message to agent 2
            _make_tool_response([{
                "id": "c1",
                "name": "send_message",
                "args": {"to": "middle", "content": "task for you"},
            }]),
            _make_text_response("Sent to middle"),
            # Agent 2: checks messages, sends to agent 3
            _make_tool_response([{"id": "c2", "name": "check_messages", "args": {}}]),
            _make_tool_response([{
                "id": "c3",
                "name": "send_message",
                "args": {"to": "final", "content": "forwarded task"},
            }]),
            _make_text_response("Forwarded to final"),
            # Agent 3: checks messages
            _make_tool_response([{"id": "c4", "name": "check_messages", "args": {}}]),
            _make_text_response("Received task"),
        ]

        a1 = Agent(name="starter", model="openai/gpt-4o", instructions="")
        a2 = Agent(name="middle", model="openai/gpt-4o", instructions="")
        a3 = Agent(name="final", model="openai/gpt-4o", instructions="")

        p = Pipeline("message_chain", enable_communication=True)
        p.add_step(a1, output_path="/r1")
        p.add_step(a2, output_path="/r2")
        p.add_step(a3, output_path="/r3")
        p.run({"input": "go"})

        assert len(p.communication.mailbox.all_messages) == 2


class TestMixedCommunication:
    """Agent uses both blackboard and mailbox in same step."""

    @patch("litellm.completion")
    def test_agent_posts_board_and_sends_message(self, mock_completion):
        """Single agent uses both blackboard and mailbox."""
        mock_completion.side_effect = [
            # Agent 1: posts to board and sends message
            _make_tool_response([
                {"id": "c1", "name": "post_to_board",
                 "args": {"key": "shared", "value": "data"}},
            ]),
            _make_tool_response([{
                "id": "c2",
                "name": "send_message",
                "args": {"to": "recipient", "content": "check board"},
            }]),
            _make_text_response("Done both"),
            # Agent 2: checks message and reads board
            _make_tool_response([{"id": "c3", "name": "check_messages", "args": {}}]),
            _make_tool_response([
                {"id": "c4", "name": "read_board", "args": {"key": "shared"}},
            ]),
            _make_text_response("Got both"),
        ]

        a1 = Agent(name="sender", model="openai/gpt-4o", instructions="")
        a2 = Agent(name="recipient", model="openai/gpt-4o", instructions="")

        p = Pipeline("mixed", enable_communication=True)
        p.add_step(a1, output_path="/r1")
        p.add_step(a2, output_path="/r2")
        result = p.run({"input": "start"})

        assert result["r1"] == "Done both"
        assert result["r2"] == "Got both"
        assert p.communication.blackboard.read("shared") is not None
        assert len(p.communication.mailbox.all_messages) == 1


class TestCommunicationWithCustomTools:
    """Pipeline combines custom tools and communication tools."""

    @patch("litellm.completion")
    def test_custom_tool_then_post_board(self, mock_completion):
        def calculate(expr: str) -> str:
            """Calculate math expression."""
            return str(eval(expr))

        mock_completion.side_effect = [
            # Agent calls custom tool, then posts result to board
            _make_tool_response([{
                "id": "c1", "name": "calculate", "args": {"expr": "10*5"},
            }]),
            _make_tool_response([{
                "id": "c2",
                "name": "post_to_board",
                "args": {"key": "result", "value": "50"},
            }]),
            _make_text_response("Calculated and posted"),
            # Agent 2 reads board
            _make_tool_response([
                {"id": "c3", "name": "read_board", "args": {"key": "result"}},
            ]),
            _make_text_response("Read result"),
        ]

        a1 = Agent(
            name="calc",
            model="openai/gpt-4o",
            instructions="",
            tools=[calculate],
        )
        a2 = Agent(name="reader", model="openai/gpt-4o", instructions="")

        p = Pipeline("custom_comm", enable_communication=True)
        p.add_step(a1, output_path="/r1")
        p.add_step(a2, output_path="/r2")
        result = p.run({"input": "compute"})

        assert result["r1"] == "Calculated and posted"
        assert result["r2"] == "Read result"
        board_val = p.communication.blackboard.read("result").value
        assert board_val == 50


class TestCommunicationWithInputOutputPaths:
    """Communication with input_map and output_path combinations."""

    @patch("litellm.completion")
    def test_input_map_with_board_communication(self, mock_completion):
        """Agent receives mapped input and uses communication."""
        mock_completion.side_effect = [
            # Agent 1: posts data
            _make_tool_response([{
                "id": "c1",
                "name": "post_to_board",
                "args": {"key": "config", "value": "settings"},
            }]),
            _make_text_response("Config saved"),
            # Agent 2: receives mapped input and reads board
            _make_tool_response([
                {"id": "c2", "name": "read_board", "args": {"key": "config"}},
            ]),
            _make_text_response("Applied config"),
        ]

        a1 = Agent(name="setup", model="openai/gpt-4o", instructions="")
        a2 = Agent(name="worker", model="openai/gpt-4o", instructions="")

        p = Pipeline("mapped", enable_communication=True)
        p.add_step(a1, output_path="/setup_result")
        p.add_step(
            a2,
            input_map={"input": "${@:/setup_result}"},
            output_path="/work_result",
        )
        result = p.run({"input": "initialize"})

        assert result["setup_result"] == "Config saved"
        assert result["work_result"] == "Applied config"


class TestPipelineSpecWithCommunication:
    """Pipeline.to_spec() includes comm tools when enabled."""

    def test_to_spec_includes_comm_tool_names(self):
        """Agent step spec includes all communication tools."""
        agent = Agent(name="talker", model="openai/gpt-4o", instructions="")

        p = Pipeline("spec_test", enable_communication=True)
        p.add_step(agent, output_path="/r")

        spec = p.to_spec()
        agent_steps = [s for s in spec if s.get("op") == "agent_loop"]
        assert len(agent_steps) == 1

        tools = agent_steps[0]["tools"]
        for tool_name in COMM_TOOL_NAMES:
            assert tool_name in tools

    def test_to_spec_no_comm_tools_when_disabled(self):
        """Without communication, no comm tools in spec."""
        agent = Agent(name="solo", model="openai/gpt-4o", instructions="")

        p = Pipeline("no_comm")
        p.add_step(agent, output_path="/r")

        spec = p.to_spec()
        agent_steps = [s for s in spec if s.get("op") == "agent_loop"]
        tools = agent_steps[0]["tools"]

        for tool_name in COMM_TOOL_NAMES:
            assert tool_name not in tools


class TestCommunicationBeforeRun:
    """Pipeline.communication is None before run()."""

    def test_communication_none_before_run(self):
        agent = Agent(name="test", model="openai/gpt-4o", instructions="")
        p = Pipeline("test", enable_communication=True)
        p.add_step(agent, output_path="/r")

        assert p.communication is None

    @patch("litellm.completion")
    def test_communication_available_after_run(self, mock_completion):
        mock_completion.return_value = _make_text_response("done")
        agent = Agent(name="test", model="openai/gpt-4o", instructions="")
        p = Pipeline("test", enable_communication=True)
        p.add_step(agent, output_path="/r")
        p.run({"input": "go"})

        assert p.communication is not None


class TestHubStateInspection:
    """Hub state inspection after complex multi-step pipeline."""

    @patch("litellm.completion")
    def test_inspect_blackboard_after_pipeline(self, mock_completion):
        """Blackboard history tracks all posts."""
        mock_completion.side_effect = [
            _make_tool_response([{
                "id": "c1",
                "name": "post_to_board",
                "args": {"key": "k1", "value": "v1"},
            }]),
            _make_text_response("posted 1"),
            _make_tool_response([{
                "id": "c2",
                "name": "post_to_board",
                "args": {"key": "k2", "value": "v2"},
            }]),
            _make_text_response("posted 2"),
            _make_tool_response([{
                "id": "c3",
                "name": "post_to_board",
                "args": {"key": "k1", "value": "v1-updated"},
            }]),
            _make_text_response("updated 1"),
        ]

        a1 = Agent(name="a1", model="openai/gpt-4o", instructions="")
        a2 = Agent(name="a2", model="openai/gpt-4o", instructions="")
        a3 = Agent(name="a3", model="openai/gpt-4o", instructions="")

        p = Pipeline("inspect", enable_communication=True)
        p.add_step(a1, output_path="/r1")
        p.add_step(a2, output_path="/r2")
        p.add_step(a3, output_path="/r3")
        p.run({"input": "go"})

        assert p.communication.blackboard.read("k1").value == "v1-updated"
        assert p.communication.blackboard.read("k2").value == "v2"
        assert len(p.communication.blackboard.history) == 3

    @patch("litellm.completion")
    def test_inspect_mailbox_all_messages(self, mock_completion):
        """All messages tracked even after consumption."""
        mock_completion.side_effect = [
            _make_tool_response([{
                "id": "c1",
                "name": "send_message",
                "args": {"to": "a2", "content": "msg1"},
            }]),
            _make_text_response("sent"),
            _make_tool_response([{"id": "c2", "name": "check_messages", "args": {}}]),
            _make_tool_response([{
                "id": "c3",
                "name": "send_message",
                "args": {"to": "a3", "content": "msg2"},
            }]),
            _make_text_response("forwarded"),
            _make_tool_response([{"id": "c4", "name": "check_messages", "args": {}}]),
            _make_text_response("received"),
        ]

        a1 = Agent(name="a1", model="openai/gpt-4o", instructions="")
        a2 = Agent(name="a2", model="openai/gpt-4o", instructions="")
        a3 = Agent(name="a3", model="openai/gpt-4o", instructions="")

        p = Pipeline("mailbox_inspect", enable_communication=True)
        p.add_step(a1, output_path="/r1")
        p.add_step(a2, output_path="/r2")
        p.add_step(a3, output_path="/r3")
        p.run({"input": "go"})

        assert len(p.communication.mailbox.all_messages) == 2
        assert p.communication.mailbox.pending_count("a2") == 0
        assert p.communication.mailbox.pending_count("a3") == 0


class TestPeekThenConsume:
    """Agent peeks messages then consumes them."""

    @patch("litellm.completion")
    def test_peek_does_not_consume(self, mock_completion):
        """Peek leaves messages in queue."""
        mock_completion.side_effect = [
            _make_tool_response([{
                "id": "c1",
                "name": "send_message",
                "args": {"to": "reader", "content": "important"},
            }]),
            _make_text_response("sent"),
            _make_tool_response([
                {"id": "c2", "name": "check_messages", "args": {"peek": True}},
            ]),
            _make_tool_response([{"id": "c3", "name": "check_messages", "args": {}}]),
            _make_text_response("consumed"),
        ]

        a1 = Agent(name="sender", model="openai/gpt-4o", instructions="")
        a2 = Agent(name="reader", model="openai/gpt-4o", instructions="")

        p = Pipeline("peek_consume", enable_communication=True)
        p.add_step(a1, output_path="/r1")
        p.add_step(a2, output_path="/r2")
        p.run({"input": "go"})

        assert p.communication.mailbox.pending_count("reader") == 0


class TestCommunicationDisabledErrors:
    """Error scenarios when communication disabled but agent tries comm."""

    def test_engine_without_comm_rejects_comm_tools(self):
        """Engine without communication cannot execute comm tools."""
        engine = build_agent_engine()

        spec = [
            {
                "op": "tool",
                "name": "send_message",
                "args": {"to": "other", "content": "hello"},
                "path": "/r",
            },
        ]
        with pytest.raises(KeyError, match="send_message"):
            engine.apply(spec, source={}, dest={})

    def test_engine_without_comm_rejects_read_board(self):
        """Cannot read blackboard without communication hub."""
        engine = build_agent_engine()

        spec = [
            {
                "op": "tool",
                "name": "read_board",
                "args": {"key": "test"},
                "path": "/r",
            },
        ]
        with pytest.raises(KeyError, match="read_board"):
            engine.apply(spec, source={}, dest={})


class TestBackwardCompatibility:
    """Existing pipeline tests work with communication disabled."""

    @patch("litellm.completion")
    def test_single_agent_no_communication(self, mock_completion):
        """Basic agent loop without communication."""
        mock_completion.return_value = _make_text_response("result")
        agent = Agent(name="worker", model="openai/gpt-4o", instructions="Work")

        p = Pipeline("basic")
        p.add_step(agent, output_path="/result")
        result = p.run({"input": "task"})

        assert result["result"] == "result"
        assert p.communication is None

    @patch("litellm.completion")
    def test_two_agents_no_communication(self, mock_completion):
        """Two agents without communication."""
        mock_completion.side_effect = [
            _make_text_response("step1"),
            _make_text_response("step2"),
        ]
        a1 = Agent(name="a1", model="openai/gpt-4o", instructions="")
        a2 = Agent(name="a2", model="openai/gpt-4o", instructions="")

        p = Pipeline("two_step")
        p.add_step(a1, output_path="/r1")
        p.add_step(a2, input_map={"input": "${@:/r1}"}, output_path="/r2")
        result = p.run({"input": "start"})

        assert result["r1"] == "step1"
        assert result["r2"] == "step2"
        assert p.communication is None

    @patch("litellm.completion")
    def test_agent_with_tools_no_communication(self, mock_completion):
        """Agent with custom tools, no communication."""
        def add(x: int, y: int) -> int:
            """Add numbers."""
            return x + y

        mock_completion.side_effect = [
            _make_tool_response([{
                "id": "c1", "name": "add", "args": {"x": 5, "y": 3},
            }]),
            _make_text_response("Sum is 8"),
        ]
        agent = Agent(
            name="calc",
            model="openai/gpt-4o",
            instructions="",
            tools=[add],
        )

        p = Pipeline("calc")
        p.add_step(agent, output_path="/result")
        result = p.run({"input": "add numbers"})

        assert result["result"] == "Sum is 8"
        assert p.communication is None


class TestComplexMultiStepPipeline:
    """Complex pipeline with multiple agents and communication patterns."""

    @patch("litellm.completion")
    def test_four_agents_complex_workflow(self, mock_completion):
        """Four agents with interleaved board and message communication."""
        mock_completion.side_effect = [
            # Agent 1: posts initial config to board
            _make_tool_response([{
                "id": "c1",
                "name": "post_to_board",
                "args": {"key": "config", "value": "{'mode': 'production'}"},
            }]),
            _make_text_response("Config posted"),
            # Agent 2: reads config, sends message to agent 3
            _make_tool_response([
                {"id": "c2", "name": "read_board", "args": {"key": "config"}},
            ]),
            _make_tool_response([{
                "id": "c3",
                "name": "send_message",
                "args": {"to": "processor", "content": "start processing"},
            }]),
            _make_text_response("Notified processor"),
            # Agent 3: checks messages, posts result to board
            _make_tool_response([{"id": "c4", "name": "check_messages", "args": {}}]),
            _make_tool_response([{
                "id": "c5",
                "name": "post_to_board",
                "args": {"key": "status", "value": "processing"},
            }]),
            _make_text_response("Processing started"),
            # Agent 4: reads board and sends final message to agent 2
            _make_tool_response([
                {"id": "c6", "name": "read_board", "args": {"key": "status"}},
            ]),
            _make_tool_response([{
                "id": "c7",
                "name": "send_message",
                "args": {"to": "coordinator", "content": "all done"},
            }]),
            _make_text_response("Workflow complete"),
        ]

        a1 = Agent(name="initializer", model="openai/gpt-4o", instructions="")
        a2 = Agent(name="coordinator", model="openai/gpt-4o", instructions="")
        a3 = Agent(name="processor", model="openai/gpt-4o", instructions="")
        a4 = Agent(name="finalizer", model="openai/gpt-4o", instructions="")

        p = Pipeline("complex_workflow", enable_communication=True)
        p.add_step(a1, output_path="/init")
        p.add_step(a2, output_path="/coord")
        p.add_step(a3, output_path="/process")
        p.add_step(a4, output_path="/final")
        result = p.run({"input": "start workflow"})

        assert result["init"] == "Config posted"
        assert result["coord"] == "Notified processor"
        assert result["process"] == "Processing started"
        assert result["final"] == "Workflow complete"
        assert len(p.communication.blackboard.history) == 2
        assert len(p.communication.mailbox.all_messages) == 2
        assert p.communication.agent_context.current_agent == "finalizer"
