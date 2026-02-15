import pytest

from perm_agent import ToolRegistry, build_agent_engine


def search(query: str) -> str:
    """Search for information"""
    return f"Result for: {query}"


def calculate(expression: str) -> str:
    """Evaluate a math expression"""
    return str(eval(expression))


def read_file(path: str) -> str:
    """Read a file by path"""
    return f"Contents of {path}"


@pytest.fixture
def tool_registry():
    registry = ToolRegistry()
    registry.register("search", search)
    registry.register("calculate", calculate)
    registry.register("read_file", read_file)
    return registry


@pytest.fixture
def tools():
    return {
        "search": search,
        "calculate": calculate,
        "read_file": read_file,
    }


@pytest.fixture
def agent_engine(tools):
    return build_agent_engine(tools=tools)
