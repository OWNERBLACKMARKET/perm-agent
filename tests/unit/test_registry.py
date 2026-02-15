import dataclasses
from enum import Enum as StdEnum
from typing import Literal

import pytest
from pydantic import BaseModel

from perm_agent import ToolRegistry
from perm_agent.registry import tool


def greet(name: str) -> str:
    """Say hello"""
    return f"Hello, {name}"


def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


def optional_fn(x: str, y: int = 5) -> str:
    """Has optional param"""
    return f"{x}-{y}"


class TestToolRegistry:
    def test_register_and_get(self):
        reg = ToolRegistry()
        reg.register("greet", greet)
        assert reg.get("greet") is greet

    def test_get_missing_raises(self):
        reg = ToolRegistry()
        with pytest.raises(KeyError, match="not registered"):
            reg.get("missing")

    def test_names(self):
        reg = ToolRegistry()
        reg.register("greet", greet)
        reg.register("add", add)
        assert set(reg.names()) == {"greet", "add"}

    def test_generate_schemas_all(self):
        reg = ToolRegistry()
        reg.register("greet", greet)
        reg.register("add", add)
        schemas = reg.generate_schemas()
        assert len(schemas) == 2
        names = {s["function"]["name"] for s in schemas}
        assert names == {"greet", "add"}

    def test_generate_schemas_subset(self):
        reg = ToolRegistry()
        reg.register("greet", greet)
        reg.register("add", add)
        schemas = reg.generate_schemas(names=["greet"])
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "greet"

    def test_schema_structure(self):
        reg = ToolRegistry()
        reg.register("add", add)
        schema = reg.generate_schemas()[0]
        assert schema["type"] == "function"
        func = schema["function"]
        assert func["name"] == "add"
        assert func["description"] == "Add two numbers"
        params = func["parameters"]
        assert params["type"] == "object"
        assert "a" in params["properties"]
        assert "b" in params["properties"]
        assert params["properties"]["a"]["type"] == "integer"
        assert set(params["required"]) == {"a", "b"}

    def test_schema_optional_params(self):
        reg = ToolRegistry()
        reg.register("optional_fn", optional_fn)
        schema = reg.generate_schemas()[0]
        params = schema["function"]["parameters"]
        assert params["required"] == ["x"]

    def test_schema_type_mapping(self):
        def typed_fn(s: str, i: int, f: float, b: bool, lst: list, d: dict) -> None:
            pass

        reg = ToolRegistry()
        reg.register("typed", typed_fn)
        schema = reg.generate_schemas()[0]
        props = schema["function"]["parameters"]["properties"]
        assert props["s"]["type"] == "string"
        assert props["i"]["type"] == "integer"
        assert props["f"]["type"] == "number"
        assert props["b"]["type"] == "boolean"
        assert props["lst"]["type"] == "array"
        assert props["d"]["type"] == "object"


# ---------------------------------------------------------------------------
# New tests for advanced type handling, docstrings, Pydantic, Enum, @tool
# ---------------------------------------------------------------------------


class Color(StdEnum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class SearchParams(BaseModel):
    query: str
    max_results: int = 10


@dataclasses.dataclass
class Point:
    x: float
    y: float


class TestAdvancedTypeResolution:
    """Tests for Optional, list[T], Literal, Enum, Pydantic, dataclass schemas."""

    def _props(self, fn, name: str = "fn") -> dict:
        reg = ToolRegistry()
        reg.register(name, fn)
        schema = reg.generate_schemas()[0]
        return schema["function"]["parameters"]

    def test_optional_type_not_required(self) -> None:
        def fn(x: str, y: str | None = None) -> None:
            pass

        params = self._props(fn)
        assert params["required"] == ["x"]
        assert params["properties"]["y"]["type"] == "string"

    def test_optional_no_default_not_required(self) -> None:
        """Optional[str] without a default is still not required."""

        def fn(x: str | None) -> None:
            pass

        params = self._props(fn)
        assert params["required"] == []

    def test_list_str_type(self) -> None:
        def fn(tags: list[str]) -> None:
            pass

        params = self._props(fn)
        prop = params["properties"]["tags"]
        assert prop == {"type": "array", "items": {"type": "string"}}

    def test_list_int_type(self) -> None:
        def fn(ids: list[int]) -> None:
            pass

        params = self._props(fn)
        prop = params["properties"]["ids"]
        assert prop == {"type": "array", "items": {"type": "integer"}}

    def test_literal_type_enum(self) -> None:
        def fn(mode: Literal["fast", "slow", "auto"]) -> None:
            pass

        params = self._props(fn)
        prop = params["properties"]["mode"]
        assert prop == {"type": "string", "enum": ["fast", "slow", "auto"]}

    def test_enum_type(self) -> None:
        def fn(color: Color) -> None:
            pass

        params = self._props(fn)
        prop = params["properties"]["color"]
        assert prop == {"type": "string", "enum": ["red", "green", "blue"]}

    def test_pydantic_model_schema(self) -> None:
        def fn(params: SearchParams) -> None:
            pass

        schema_params = self._props(fn)
        prop = schema_params["properties"]["params"]
        # Pydantic model_json_schema produces a full object schema
        assert prop["type"] == "object"
        assert "query" in prop["properties"]
        assert "max_results" in prop["properties"]

    def test_dataclass_schema(self) -> None:
        def fn(point: Point) -> None:
            pass

        params = self._props(fn)
        prop = params["properties"]["point"]
        assert prop["type"] == "object"
        assert prop["title"] == "Point"
        assert prop["properties"]["x"]["type"] == "number"
        assert prop["properties"]["y"]["type"] == "number"

    def test_dict_str_any_type(self) -> None:
        from typing import Any

        def fn(metadata: dict[str, Any]) -> None:
            pass

        params = self._props(fn)
        assert params["properties"]["metadata"]["type"] == "object"

    def test_nested_optional_list(self) -> None:
        def fn(items: list[str] | None = None) -> None:
            pass

        params = self._props(fn)
        prop = params["properties"]["items"]
        assert prop == {"type": "array", "items": {"type": "string"}}
        assert params["required"] == []


class TestDocstringParsing:
    """Tests for parameter description extraction from docstrings."""

    def _props(self, fn, name: str = "fn") -> dict:
        reg = ToolRegistry()
        reg.register(name, fn)
        schema = reg.generate_schemas()[0]
        return schema["function"]["parameters"]

    def test_docstring_param_descriptions(self) -> None:
        def search(query: str, max_results: int = 10) -> str:
            """Search the web for information.

            Args:
                query: The search query string.
                max_results: Maximum number of results to return.
            """
            return ""

        params = self._props(search, "search")
        assert params["properties"]["query"]["description"] == "The search query string."
        max_desc = params["properties"]["max_results"]["description"]
        assert max_desc == "Maximum number of results to return."

    def test_docstring_google_style(self) -> None:
        def analyze(text: str, language: str = "en") -> str:
            """Analyze text content.

            Args:
                text: The text to analyze.
                language: Language code for analysis.
            """
            return ""

        params = self._props(analyze, "analyze")
        assert params["properties"]["text"]["description"] == "The text to analyze."
        assert params["properties"]["language"]["description"] == "Language code for analysis."

    def test_no_docstring(self) -> None:
        def fn(x: str) -> None:
            pass

        params = self._props(fn)
        assert "description" not in params["properties"]["x"]

    def test_docstring_without_args_section(self) -> None:
        def fn(x: str) -> None:
            """Just a simple function."""

        params = self._props(fn)
        assert "description" not in params["properties"]["x"]


class TestToolDecorator:
    """Tests for the @tool decorator."""

    def test_tool_decorator(self) -> None:
        @tool
        def search(query: str) -> str:
            """Search the web."""
            return f"Results for {query}"

        assert hasattr(search, "_tool_metadata")
        meta = search._tool_metadata
        assert meta["name"] == "search"
        assert meta["schema"]["function"]["name"] == "search"
        assert meta["schema"]["function"]["description"] == "Search the web."
        # Still callable
        assert search("test") == "Results for test"

    def test_tool_decorator_custom_name(self) -> None:
        @tool(name="web_search")
        def search(query: str) -> str:
            """Search the web."""
            return f"Results for {query}"

        meta = search._tool_metadata
        assert meta["name"] == "web_search"
        assert meta["schema"]["function"]["name"] == "web_search"

    def test_tool_decorator_preserves_function(self) -> None:
        @tool
        def greet(name: str) -> str:
            """Say hi."""
            return f"Hi, {name}"

        assert greet("Alice") == "Hi, Alice"
        assert greet.__name__ == "greet"
        assert greet.__doc__ == "Say hi."


class TestComplexToolSchema:
    """Integration test combining multiple advanced features."""

    def test_complex_tool_schema(self) -> None:
        @tool
        def process(
            query: str,
            tags: list[str],
            mode: Literal["fast", "slow"],
            color: Color,
            limit: int | None = None,
        ) -> str:
            """Process data with various options.

            Args:
                query: The main search query.
                tags: List of tags to filter by.
                mode: Processing speed mode.
                color: Color preference.
                limit: Max items to process.
            """
            return "done"

        meta = process._tool_metadata
        schema = meta["schema"]
        params = schema["function"]["parameters"]

        assert params["properties"]["query"] == {
            "type": "string",
            "description": "The main search query.",
        }
        assert params["properties"]["tags"] == {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of tags to filter by.",
        }
        assert params["properties"]["mode"] == {
            "type": "string",
            "enum": ["fast", "slow"],
            "description": "Processing speed mode.",
        }
        assert params["properties"]["color"] == {
            "type": "string",
            "enum": ["red", "green", "blue"],
            "description": "Color preference.",
        }
        assert params["properties"]["limit"] == {
            "type": "integer",
            "description": "Max items to process.",
        }
        # query, tags, mode, color are required; limit is Optional with default
        assert set(params["required"]) == {"query", "tags", "mode", "color"}
