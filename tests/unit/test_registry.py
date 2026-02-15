import pytest

from perm_agent import ToolRegistry


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
        def typed_fn(s: str, i: int, f: float, b: bool, l: list, d: dict) -> None:
            pass

        reg = ToolRegistry()
        reg.register("typed", typed_fn)
        schema = reg.generate_schemas()[0]
        props = schema["function"]["parameters"]["properties"]
        assert props["s"]["type"] == "string"
        assert props["i"]["type"] == "integer"
        assert props["f"]["type"] == "number"
        assert props["b"]["type"] == "boolean"
        assert props["l"]["type"] == "array"
        assert props["d"]["type"] == "object"
