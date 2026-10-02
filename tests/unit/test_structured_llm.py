"""StructuredLLM: one place for schema-constrained generation, native structured output and the correction loop."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.router import ModelRouter
from app.providers.llm.structured import (
    SchemaValidationError,
    StructuredLLM,
    StructuredOutputError,
    extract_json,
    native_schema,
    parse_structured,
    validate_json_schema,
)
from app.schemas.common import ModelTier
from tests.unit.helpers import routing, scope


class Hook(BaseModel):
    title: str = Field(min_length=3)
    minutes: int = Field(ge=1, le=10)
    tags: list[str] = []


def structured(*faults, responder=None):
    llm = MockLLMProvider({"a": responder or (lambda r: {"title": "Fracciones", "minutes": 3})})
    if faults:
        llm.inject("a", *faults)
    return StructuredLLM(ModelRouter(routing(("mock", "m1")), {"mock": llm})), llm


async def generate(llm: StructuredLLM, output, **kwargs):
    sc, events = scope()
    result = await llm.generate(output, agent_id="a", tier=ModelTier.CHEAP, system="sys", prompt="go", scope=sc,
                                **kwargs)
    return result, events


async def test_pydantic_output_and_schema_reach_the_provider() -> None:
    s, llm = structured()
    result, events = await generate(s, Hook)
    assert result == Hook(title="Fracciones", minutes=3)
    assert llm.requests[0].response_schema == Hook.model_json_schema()
    assert not [e for e in events if e.type == "agent.validation_failed"]


async def test_json_schema_dict_output() -> None:
    schema = {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]}
    s, _ = structured()
    result, _ = await generate(s, schema)
    assert result == {"title": "Fracciones", "minutes": 3}


async def test_fenced_and_wrapped_json_is_parsed() -> None:
    s, _ = structured('Here you go:\n```json\n{"title": "Hola", "minutes": 2}\n```\nEnjoy.')
    result, events = await generate(s, Hook)
    assert result.title == "Hola" and not [e for e in events if e.type == "agent.validation_failed"]


async def test_correction_loop_sends_the_error_back() -> None:
    s, llm = structured("not json at all", '{"title": "ab", "minutes": 3}')
    result, events = await generate(s, Hook)
    assert result.title == "Fracciones"
    failures = [e for e in events if e.type == "agent.validation_failed"]
    assert [e.data["attempt"] for e in failures] == [1, 2]
    third = llm.requests[2].messages
    assert [m.role for m in third] == ["user", "assistant", "user", "assistant", "user"]
    assert "failed validation" in third[-1].content and "title" in third[-1].content


async def test_semantic_check_drives_a_correction() -> None:
    def check(hook: Hook) -> None:
        if "draft" in hook.tags:
            raise ValueError("tags must not contain 'draft'")

    s, _ = structured('{"title": "Hola", "minutes": 2, "tags": ["draft"]}')
    result, events = await generate(s, Hook, check=check)
    assert result.tags == [] and "draft" in events[[e.type for e in events].index("agent.validation_failed")].data[
        "error"]


async def test_exhausted_attempts_raise_a_typed_error() -> None:
    s, llm = structured("nope", "still nope", "never")
    with pytest.raises(StructuredOutputError) as err:
        await generate(s, Hook, validation_retries=2)
    assert err.value.attempts == 3 and "no valid output after 3 attempts" in str(err.value)
    assert llm.calls["a"] == 3


def test_extract_json_handles_prose_and_fences() -> None:
    assert extract_json('```\n{"a": 1}\n```') == '{"a": 1}'
    assert extract_json('Sure! {"a": {"b": [1]}} done') == '{"a": {"b": [1]}}'
    assert extract_json("[1, 2]") == "[1, 2]"


def test_parse_structured_validates_dict_schemas() -> None:
    schema = {"type": "object", "properties": {"n": {"type": "integer", "minimum": 1}}, "required": ["n"]}
    assert parse_structured('{"n": 2}', schema) == {"n": 2}
    with pytest.raises(SchemaValidationError, match=r"\$\.n"):
        parse_structured('{"n": 0}', schema)
    with pytest.raises(SchemaValidationError, match="required"):
        parse_structured("{}", schema)


def test_validate_json_schema_covers_the_common_keywords() -> None:
    schema = {
        "type": "object", "additionalProperties": False, "required": ["kind", "items"],
        "properties": {
            "kind": {"enum": ["a", "b"]},
            "items": {"type": "array", "minItems": 1, "items": {"$ref": "#/$defs/Item"}},
            "note": {"anyOf": [{"type": "string", "maxLength": 5}, {"type": "null"}]},
        },
        "$defs": {"Item": {"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"]}},
    }
    validate_json_schema({"kind": "a", "items": [{"x": 1.5}], "note": None}, schema)
    for bad, where in [({"kind": "c", "items": [{"x": 1}]}, "$.kind"), ({"kind": "a", "items": []}, "$.items"),
                       ({"kind": "a", "items": [{"x": "1"}]}, "$.items[0].x"),
                       ({"kind": "a", "items": [{"x": 1}], "extra": 1}, "extra"),
                       ({"kind": "a", "items": [{"x": 1}], "note": "toolong"}, "$.note"),
                       ({"kind": "a", "items": [{"x": True}]}, "$.items[0].x")]:
        with pytest.raises(SchemaValidationError, match=where.replace("$", r"\$").replace("[", r"\[")):
            validate_json_schema(bad, schema)


def test_native_schema_produces_the_strict_subset() -> None:
    native = native_schema(Hook.model_json_schema(), all_required=False)
    assert native["additionalProperties"] is False
    assert "minLength" not in str(native) and "maximum" not in str(native)
    # every property required in strict mode: an optional property must be nullable, or there is no native schema
    assert native_schema(Hook.model_json_schema(), all_required=True) is None

    class Strict(BaseModel):
        name: str
        note: str | None
        format: str  # a property called "format" is a name, not the keyword

    strict = native_schema(Strict.model_json_schema(), all_required=True)
    assert strict["required"] == ["name", "note", "format"] and "format" in strict["properties"]


def test_native_schema_rejects_what_strict_mode_cannot_express() -> None:
    assert native_schema({"type": "object", "properties": {"meta": {"type": "object"}}}) is None
    assert native_schema({"type": "object", "properties": {"m": {"type": "object",
                                                                 "additionalProperties": {"type": "string"}}}}) is None
    recursive = {"type": "object", "properties": {"n": {"$ref": "#/$defs/N"}},
                 "$defs": {"N": {"type": "object", "properties": {"kids": {"type": "array",
                                                                           "items": {"$ref": "#/$defs/N"}}}}}}
    assert native_schema(recursive) is None
    assert native_schema({"type": "array", "items": {"type": "string"}}) is None  # the root must be an object
    one_of = native_schema({"type": "object", "properties": {"v": {"oneOf": [{"type": "string"},
                                                                            {"type": "integer"}]}}})
    assert one_of["properties"]["v"] == {"anyOf": [{"type": "string"}, {"type": "integer"}]}
