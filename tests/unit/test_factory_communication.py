"""Unit tests for factory.py communication wiring and middleware."""


from perm_agent.comm_tools import COMM_TOOL_NAMES
from perm_agent.communication import CommunicationHub
from perm_agent.factory import _AgentMetadataMiddleware, build_agent_engine
from perm_agent.registry import ToolRegistry


def dummy_tool(x: str) -> str:
    """A simple tool."""
    return f"processed: {x}"


class TestBuildAgentEngineWithCommunication:
    """Test build_agent_engine registers comm tools when hub provided."""

    def test_registers_all_four_comm_tools(self):
        hub = CommunicationHub()
        engine = build_agent_engine(communication=hub)
        metadata = engine.main_pipeline._middlewares[0]
        registry = metadata._registry
        tool_names = set(registry.names())
        assert COMM_TOOL_NAMES.issubset(tool_names)

    def test_comm_tools_registered_with_correct_names(self):
        hub = CommunicationHub()
        engine = build_agent_engine(communication=hub)
        metadata = engine.main_pipeline._middlewares[0]
        registry = metadata._registry
        for name in COMM_TOOL_NAMES:
            assert name in registry.names()

    def test_comm_tool_callables_have_correct_name_attribute(self):
        hub = CommunicationHub()
        engine = build_agent_engine(communication=hub)
        metadata = engine.main_pipeline._middlewares[0]
        registry = metadata._registry
        for name in COMM_TOOL_NAMES:
            fn = registry.get(name)
            assert fn.__name__ == name

    def test_comm_tools_generate_valid_schemas(self):
        hub = CommunicationHub()
        engine = build_agent_engine(communication=hub)
        metadata = engine.main_pipeline._middlewares[0]
        schemas = metadata._schemas
        schema_names = {s["function"]["name"] for s in schemas}
        assert COMM_TOOL_NAMES.issubset(schema_names)

    def test_comm_tools_coexist_with_regular_tools(self):
        hub = CommunicationHub()
        engine = build_agent_engine(
            communication=hub,
            tools={"dummy_tool": dummy_tool},
        )
        metadata = engine.main_pipeline._middlewares[0]
        registry = metadata._registry
        names = set(registry.names())
        assert "dummy_tool" in names
        assert COMM_TOOL_NAMES.issubset(names)

    def test_comm_and_regular_tools_both_in_schemas(self):
        hub = CommunicationHub()
        engine = build_agent_engine(
            communication=hub,
            tools={"dummy_tool": dummy_tool},
        )
        metadata = engine.main_pipeline._middlewares[0]
        schemas = metadata._schemas
        schema_names = {s["function"]["name"] for s in schemas}
        assert "dummy_tool" in schema_names
        assert COMM_TOOL_NAMES.issubset(schema_names)


class TestBuildAgentEngineWithoutCommunication:
    """Test build_agent_engine does NOT register comm tools without hub."""

    def test_no_comm_tools_registered(self):
        engine = build_agent_engine()
        metadata = engine.main_pipeline._middlewares[0]
        registry = metadata._registry
        tool_names = set(registry.names())
        assert not COMM_TOOL_NAMES.intersection(tool_names)

    def test_only_regular_tools_registered(self):
        engine = build_agent_engine(tools={"dummy_tool": dummy_tool})
        metadata = engine.main_pipeline._middlewares[0]
        registry = metadata._registry
        assert registry.names() == ["dummy_tool"]

    def test_no_comm_tools_in_schemas(self):
        engine = build_agent_engine(tools={"dummy_tool": dummy_tool})
        metadata = engine.main_pipeline._middlewares[0]
        schemas = metadata._schemas
        schema_names = {s["function"]["name"] for s in schemas}
        assert not COMM_TOOL_NAMES.intersection(schema_names)

    def test_communication_none_explicitly(self):
        engine = build_agent_engine(communication=None)
        metadata = engine.main_pipeline._middlewares[0]
        registry = metadata._registry
        tool_names = set(registry.names())
        assert not COMM_TOOL_NAMES.intersection(tool_names)


class TestAgentMetadataMiddlewareSetsCommunication:
    """Test middleware sets ctx.metadata['_communication']."""

    def test_sets_communication_in_metadata(self):
        hub = CommunicationHub()
        registry = ToolRegistry()
        schemas = []
        specs = {}
        middleware = _AgentMetadataMiddleware(
            registry=registry,
            schemas=schemas,
            agent_specs=specs,
            communication=hub,
        )

        from unittest.mock import MagicMock

        from j_perm import ExecutionContext
        ctx = ExecutionContext(source={}, dest={}, engine=MagicMock())

        middleware.process({}, ctx)
        assert ctx.metadata["_communication"] is hub

    def test_does_not_set_communication_if_none(self):
        registry = ToolRegistry()
        schemas = []
        specs = {}
        middleware = _AgentMetadataMiddleware(
            registry=registry,
            schemas=schemas,
            agent_specs=specs,
            communication=None,
        )

        from unittest.mock import MagicMock

        from j_perm import ExecutionContext
        ctx = ExecutionContext(source={}, dest={}, engine=MagicMock())

        middleware.process({}, ctx)
        assert "_communication" not in ctx.metadata

    def test_always_sets_registry_schemas_specs(self):
        registry = ToolRegistry()
        schemas = [{"name": "tool1"}]
        specs = {"agent1": []}
        middleware = _AgentMetadataMiddleware(
            registry=registry,
            schemas=schemas,
            agent_specs=specs,
            communication=None,
        )

        from unittest.mock import MagicMock

        from j_perm import ExecutionContext
        ctx = ExecutionContext(source={}, dest={}, engine=MagicMock())

        middleware.process({}, ctx)
        assert ctx.metadata["_tool_registry"] is registry
        assert ctx.metadata["_tool_schemas"] == schemas
        assert ctx.metadata["_agent_specs"] == specs


class TestAgentMetadataMiddlewareUpdatesCurrentAgent:
    """Test middleware updates agent_context.current_agent."""

    def test_updates_current_agent_from_step_agent_name(self):
        hub = CommunicationHub()
        hub.agent_context.current_agent = "unknown"
        registry = ToolRegistry()
        middleware = _AgentMetadataMiddleware(
            registry=registry,
            schemas=[],
            agent_specs={},
            communication=hub,
        )

        from unittest.mock import MagicMock

        from j_perm import ExecutionContext
        ctx = ExecutionContext(source={}, dest={}, engine=MagicMock())

        step = {"agent_name": "researcher", "op": "llm"}
        middleware.process(step, ctx)
        assert hub.agent_context.current_agent == "researcher"

    def test_updates_to_different_agent_on_second_call(self):
        hub = CommunicationHub()
        registry = ToolRegistry()
        middleware = _AgentMetadataMiddleware(
            registry=registry,
            schemas=[],
            agent_specs={},
            communication=hub,
        )

        from unittest.mock import MagicMock

        from j_perm import ExecutionContext
        ctx = ExecutionContext(source={}, dest={}, engine=MagicMock())

        middleware.process({"agent_name": "alice"}, ctx)
        assert hub.agent_context.current_agent == "alice"

        middleware.process({"agent_name": "bob"}, ctx)
        assert hub.agent_context.current_agent == "bob"

    def test_ignores_step_without_agent_name(self):
        hub = CommunicationHub()
        hub.agent_context.current_agent = "initial"
        registry = ToolRegistry()
        middleware = _AgentMetadataMiddleware(
            registry=registry,
            schemas=[],
            agent_specs={},
            communication=hub,
        )

        from unittest.mock import MagicMock

        from j_perm import ExecutionContext
        ctx = ExecutionContext(source={}, dest={}, engine=MagicMock())

        step = {"op": "set", "path": "/x"}
        middleware.process(step, ctx)
        assert hub.agent_context.current_agent == "initial"

    def test_ignores_step_with_none_agent_name(self):
        hub = CommunicationHub()
        hub.agent_context.current_agent = "initial"
        registry = ToolRegistry()
        middleware = _AgentMetadataMiddleware(
            registry=registry,
            schemas=[],
            agent_specs={},
            communication=hub,
        )

        from unittest.mock import MagicMock

        from j_perm import ExecutionContext
        ctx = ExecutionContext(source={}, dest={}, engine=MagicMock())

        step = {"agent_name": None, "op": "llm"}
        middleware.process(step, ctx)
        assert hub.agent_context.current_agent == "initial"

    def test_ignores_step_with_empty_agent_name(self):
        hub = CommunicationHub()
        hub.agent_context.current_agent = "initial"
        registry = ToolRegistry()
        middleware = _AgentMetadataMiddleware(
            registry=registry,
            schemas=[],
            agent_specs={},
            communication=hub,
        )

        from unittest.mock import MagicMock

        from j_perm import ExecutionContext
        ctx = ExecutionContext(source={}, dest={}, engine=MagicMock())

        step = {"agent_name": "", "op": "llm"}
        middleware.process(step, ctx)
        assert hub.agent_context.current_agent == "initial"

    def test_ignores_non_dict_step(self):
        hub = CommunicationHub()
        hub.agent_context.current_agent = "initial"
        registry = ToolRegistry()
        middleware = _AgentMetadataMiddleware(
            registry=registry,
            schemas=[],
            agent_specs={},
            communication=hub,
        )

        from unittest.mock import MagicMock

        from j_perm import ExecutionContext
        ctx = ExecutionContext(source={}, dest={}, engine=MagicMock())

        middleware.process("string_step", ctx)
        assert hub.agent_context.current_agent == "initial"

        middleware.process(123, ctx)
        assert hub.agent_context.current_agent == "initial"

        middleware.process(None, ctx)
        assert hub.agent_context.current_agent == "initial"


class TestAgentMetadataMiddlewareSetdefault:
    """Test middleware uses setdefault and doesn't overwrite."""

    def test_does_not_overwrite_existing_registry(self):
        hub = CommunicationHub()
        registry1 = ToolRegistry()
        registry2 = ToolRegistry()
        registry1.register("tool1", lambda: "a")

        middleware = _AgentMetadataMiddleware(
            registry=registry2,
            schemas=[],
            agent_specs={},
            communication=hub,
        )

        from unittest.mock import MagicMock

        from j_perm import ExecutionContext
        ctx = ExecutionContext(source={}, dest={}, engine=MagicMock())
        ctx.metadata["_tool_registry"] = registry1

        middleware.process({}, ctx)
        assert ctx.metadata["_tool_registry"] is registry1
        assert ctx.metadata["_tool_registry"] is not registry2

    def test_does_not_overwrite_existing_schemas(self):
        hub = CommunicationHub()
        existing_schemas = [{"existing": "schema"}]
        new_schemas = [{"new": "schema"}]

        middleware = _AgentMetadataMiddleware(
            registry=ToolRegistry(),
            schemas=new_schemas,
            agent_specs={},
            communication=hub,
        )

        from unittest.mock import MagicMock

        from j_perm import ExecutionContext
        ctx = ExecutionContext(source={}, dest={}, engine=MagicMock())
        ctx.metadata["_tool_schemas"] = existing_schemas

        middleware.process({}, ctx)
        assert ctx.metadata["_tool_schemas"] == existing_schemas

    def test_does_not_overwrite_existing_agent_specs(self):
        hub = CommunicationHub()
        existing_specs = {"agent1": []}
        new_specs = {"agent2": []}

        middleware = _AgentMetadataMiddleware(
            registry=ToolRegistry(),
            schemas=[],
            agent_specs=new_specs,
            communication=hub,
        )

        from unittest.mock import MagicMock

        from j_perm import ExecutionContext
        ctx = ExecutionContext(source={}, dest={}, engine=MagicMock())
        ctx.metadata["_agent_specs"] = existing_specs

        middleware.process({}, ctx)
        assert ctx.metadata["_agent_specs"] == existing_specs

    def test_does_not_overwrite_existing_communication(self):
        hub1 = CommunicationHub()
        hub2 = CommunicationHub()

        middleware = _AgentMetadataMiddleware(
            registry=ToolRegistry(),
            schemas=[],
            agent_specs={},
            communication=hub2,
        )

        from unittest.mock import MagicMock

        from j_perm import ExecutionContext
        ctx = ExecutionContext(source={}, dest={}, engine=MagicMock())
        ctx.metadata["_communication"] = hub1

        middleware.process({}, ctx)
        assert ctx.metadata["_communication"] is hub1
        assert ctx.metadata["_communication"] is not hub2


class TestAgentMetadataMiddlewareWithTracer:
    """Test middleware handles tracer correctly."""

    def test_sets_tracer_when_provided(self):
        from unittest.mock import MagicMock

        tracer = MagicMock()
        registry = ToolRegistry()
        middleware = _AgentMetadataMiddleware(
            registry=registry,
            schemas=[],
            agent_specs={},
            tracer=tracer,
        )

        from unittest.mock import MagicMock

        from j_perm import ExecutionContext
        ctx = ExecutionContext(source={}, dest={}, engine=MagicMock())

        middleware.process({}, ctx)
        assert ctx.metadata["_tracer"] is tracer

    def test_does_not_set_tracer_when_none(self):
        registry = ToolRegistry()
        middleware = _AgentMetadataMiddleware(
            registry=registry,
            schemas=[],
            agent_specs={},
            tracer=None,
        )

        from unittest.mock import MagicMock

        from j_perm import ExecutionContext
        ctx = ExecutionContext(source={}, dest={}, engine=MagicMock())

        middleware.process({}, ctx)
        assert "_tracer" not in ctx.metadata

    def test_does_not_overwrite_existing_tracer(self):
        from unittest.mock import MagicMock

        existing_tracer = MagicMock()
        new_tracer = MagicMock()
        registry = ToolRegistry()
        middleware = _AgentMetadataMiddleware(
            registry=registry,
            schemas=[],
            agent_specs={},
            tracer=new_tracer,
        )

        from unittest.mock import MagicMock

        from j_perm import ExecutionContext
        ctx = ExecutionContext(source={}, dest={}, engine=MagicMock())
        ctx.metadata["_tracer"] = existing_tracer

        middleware.process({}, ctx)
        assert ctx.metadata["_tracer"] is existing_tracer


class TestAgentMetadataMiddlewareReturnsStep:
    """Test middleware returns step unchanged."""

    def test_returns_step_unchanged(self):
        registry = ToolRegistry()
        middleware = _AgentMetadataMiddleware(
            registry=registry,
            schemas=[],
            agent_specs={},
        )

        from unittest.mock import MagicMock

        from j_perm import ExecutionContext
        ctx = ExecutionContext(source={}, dest={}, engine=MagicMock())

        step = {"op": "llm", "model": "gpt-4"}
        result = middleware.process(step, ctx)
        assert result is step

    def test_returns_non_dict_step_unchanged(self):
        registry = ToolRegistry()
        middleware = _AgentMetadataMiddleware(
            registry=registry,
            schemas=[],
            agent_specs={},
        )

        from unittest.mock import MagicMock

        from j_perm import ExecutionContext
        ctx = ExecutionContext(source={}, dest={}, engine=MagicMock())

        assert middleware.process("string", ctx) == "string"
        assert middleware.process(123, ctx) == 123
        assert middleware.process(None, ctx) is None
