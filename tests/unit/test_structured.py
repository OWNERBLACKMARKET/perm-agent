import json

import pytest
from pydantic import BaseModel, ValidationError

from perm_agent.structured import StructuredOutput


class Person(BaseModel):
    name: str
    age: int


class Address(BaseModel):
    street: str
    city: str
    zip_code: str | None = None


class TestStructuredOutputParse:
    def test_parse_json_string(self):
        so = StructuredOutput(Person)
        result = so.parse('{"name": "Alice", "age": 30}')
        assert result.name == "Alice"
        assert result.age == 30

    def test_parse_dict(self):
        so = StructuredOutput(Person)
        result = so.parse({"name": "Bob", "age": 25})
        assert result.name == "Bob"
        assert result.age == 25

    def test_parse_invalid_json_raises(self):
        so = StructuredOutput(Person)
        with pytest.raises(json.JSONDecodeError):
            so.parse("not json")

    def test_parse_invalid_data_raises(self):
        so = StructuredOutput(Person)
        with pytest.raises(ValidationError):
            so.parse('{"name": "Alice"}')  # missing age

    def test_parse_extra_fields_ignored(self):
        so = StructuredOutput(Person)
        result = so.parse('{"name": "Alice", "age": 30, "extra": true}')
        assert result.name == "Alice"
        assert result.age == 30

    def test_parse_with_optional_field_present(self):
        so = StructuredOutput(Address)
        result = so.parse('{"street": "Main St", "city": "NYC", "zip_code": "10001"}')
        assert result.zip_code == "10001"

    def test_parse_with_optional_field_absent(self):
        so = StructuredOutput(Address)
        result = so.parse('{"street": "Main St", "city": "NYC"}')
        assert result.zip_code is None

    def test_parse_wrong_type_coercion(self):
        so = StructuredOutput(Person)
        # Pydantic will coerce string "30" to int 30
        result = so.parse('{"name": "Alice", "age": "30"}')
        assert result.age == 30


class TestStructuredOutputParseSafe:
    def test_parse_safe_valid(self):
        so = StructuredOutput(Person)
        result = so.parse_safe('{"name": "Alice", "age": 30}')
        assert result is not None
        assert result.name == "Alice"

    def test_parse_safe_invalid_json(self):
        so = StructuredOutput(Person)
        assert so.parse_safe("not json") is None

    def test_parse_safe_invalid_data(self):
        so = StructuredOutput(Person)
        assert so.parse_safe('{"name": "Alice"}') is None

    def test_parse_safe_empty_string(self):
        so = StructuredOutput(Person)
        assert so.parse_safe("") is None


class TestStructuredOutputSchema:
    def test_json_schema(self):
        so = StructuredOutput(Person)
        schema = so.json_schema()
        assert schema["type"] == "object"
        assert "name" in schema["properties"]
        assert "age" in schema["properties"]

    def test_model_class_property(self):
        so = StructuredOutput(Person)
        assert so.model_class is Person


class TestStructuredOutputImport:
    def test_importable_from_package(self):
        from perm_agent import StructuredOutput as SO

        assert SO is StructuredOutput
