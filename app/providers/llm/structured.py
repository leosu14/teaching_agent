"""StructuredLLM: structured output through the provider abstraction. The single place that turns model text into
validated data, so no agent parses JSON itself.

- The output is described by a Pydantic model or a plain JSON Schema.
- The schema goes with every request (`response_schema`); a provider with native structured output enforces it
  (see `native_schema`), any other provider gets it in the prompt and returns JSON text.
- Either way the text is parsed (fenced or bare JSON) and validated here, never trusted.
- Invalid output is sent back to the model with the validation error, up to `validation_retries` times.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from app.observability.scope import ExecutionScope
from app.providers.llm.base import LLMMessage
from app.providers.llm.router import ModelRouter
from app.schemas.common import ModelTier
from app.schemas.events import EventType

M = TypeVar("M", bound=BaseModel)
JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class StructuredOutputError(Exception):
    """The model never produced output that passed validation."""

    def __init__(self, message: str, *, attempts: int, last_error: str) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error


class SchemaValidationError(ValueError):
    """Data does not match a plain JSON Schema."""


def extract_json(text: str) -> str:
    """The JSON in a model reply: a fenced block, the whole reply, or the first JSON value inside prose."""
    match = JSON_BLOCK.search(text)
    text = (match.group(1) if match else text).strip()
    if text[:1] in ("{", "["):
        return text
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if starts:
        try:
            _, end = json.JSONDecoder().raw_decode(text, min(starts))
            return text[min(starts):end]
        except ValueError:
            pass
    return text


# --- Native structured output -------------------------------------------------------------------------------

# Keywords vendors' strict JSON-schema modes reject; validation still enforces them after parsing.
UNSUPPORTED_KEYWORDS = {"minLength", "maxLength", "pattern", "format", "minimum", "maximum", "exclusiveMinimum",
                        "exclusiveMaximum", "multipleOf", "minItems", "maxItems", "uniqueItems", "minProperties",
                        "maxProperties", "default", "examples", "discriminator"}


def native_schema(schema: dict, *, all_required: bool = False) -> dict | None:
    """The schema in the strict subset vendors' native structured-output modes accept, or None when it cannot be
    expressed there (free-form objects, tuples, recursion, or with `all_required` an optional non-nullable field).
    Constraints that are dropped are still enforced by validation after parsing."""
    schema = copy.deepcopy(schema)
    defs = schema.get("$defs", {})
    if schema.get("type") != "object" or _recursive(defs):  # vendors require an object at the root
        return None

    def walk(node: Any) -> bool:
        if not isinstance(node, dict):
            return True
        for key in UNSUPPORTED_KEYWORDS & node.keys():
            del node[key]
        if "oneOf" in node:
            node["anyOf"] = node.pop("oneOf")
        if "prefixItems" in node:
            return False
        is_object = node.get("type") == "object" or "properties" in node
        if is_object:
            extra = node.get("additionalProperties")
            if extra not in (None, False) or not node.get("properties"):
                return False  # a free-form dict cannot be expressed
            node["additionalProperties"] = False
            if all_required:
                required = set(node.get("required", []))
                for name, prop in node["properties"].items():
                    if name not in required:
                        if not _nullable(prop, defs):
                            return False
                        required.add(name)
                node["required"] = list(node["properties"])
        ok = True
        for key, value in node.items():
            if key in ("properties", "$defs", "definitions") and isinstance(value, dict):
                ok = all([walk(v) for v in value.values()]) and ok
            elif key in ("items", "additionalProperties", "not") and isinstance(value, dict):
                ok = walk(value) and ok
            elif key in ("anyOf", "allOf") and isinstance(value, list):
                ok = all([walk(v) for v in value]) and ok
        return ok

    return schema if walk(schema) else None


def _nullable(prop: dict, defs: dict) -> bool:
    if prop.get("type") == "null" or (isinstance(prop.get("type"), list) and "null" in prop["type"]):
        return True
    return any(_nullable(p, defs) for p in prop.get("anyOf", []) + prop.get("oneOf", []))


def _refs(node: Any) -> set[str]:
    if isinstance(node, dict):
        found = {node["$ref"].split("/")[-1]} if isinstance(node.get("$ref"), str) else set()
        return found.union(*(_refs(v) for v in node.values()))
    if isinstance(node, list):
        return set().union(*(_refs(v) for v in node))
    return set()


def _recursive(defs: dict) -> bool:
    graph = {name: _refs(body) for name, body in defs.items()}

    def reaches(start: str, target: str, seen: set[str]) -> bool:
        for nxt in graph.get(start, ()):
            if nxt == target or (nxt not in seen and reaches(nxt, target, seen | {nxt})):
                return True
        return False

    return any(reaches(name, name, {name}) for name in graph)


# --- Plain JSON Schema validation (for callers that describe output without a Pydantic model) ----------------

TYPES = {"object": dict, "array": list, "string": str, "integer": int, "number": (int, float), "boolean": bool,
         "null": type(None)}


def validate_json_schema(data: Any, schema: dict, *, root: dict | None = None, path: str = "$") -> None:
    """Validate `data` against the common JSON Schema keywords (type, properties, required, additionalProperties,
    items, enum, const, anyOf/oneOf/allOf, $ref, string/number/array bounds). Raises SchemaValidationError."""
    root = root if root is not None else schema
    if "$ref" in schema:
        name = schema["$ref"].split("/")[-1]
        return validate_json_schema(data, root.get("$defs", root.get("definitions", {}))[name], root=root, path=path)
    for key in ("anyOf", "oneOf"):
        if key in schema:
            for option in schema[key]:
                try:
                    validate_json_schema(data, option, root=root, path=path)
                    break
                except SchemaValidationError:
                    continue
            else:
                raise SchemaValidationError(f"{path}: matches none of the allowed schemas")
    for option in schema.get("allOf", []):
        validate_json_schema(data, option, root=root, path=path)
    if "type" in schema:
        allowed = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(isinstance(data, TYPES[t]) and not (t in ("integer", "number") and isinstance(data, bool))
                   for t in allowed):
            raise SchemaValidationError(f"{path}: expected {' or '.join(allowed)}, got {type(data).__name__}")
    if "enum" in schema and data not in schema["enum"]:
        raise SchemaValidationError(f"{path}: {data!r} is not one of {schema['enum']}")
    if "const" in schema and data != schema["const"]:
        raise SchemaValidationError(f"{path}: must be {schema['const']!r}")
    if isinstance(data, str):
        if len(data) < schema.get("minLength", 0) or len(data) > schema.get("maxLength", len(data)):
            raise SchemaValidationError(f"{path}: string length {len(data)} out of bounds")
        if "pattern" in schema and not re.search(schema["pattern"], data):
            raise SchemaValidationError(f"{path}: does not match {schema['pattern']!r}")
    if isinstance(data, (int, float)) and not isinstance(data, bool):
        if "minimum" in schema and data < schema["minimum"] or "maximum" in schema and data > schema["maximum"]:
            raise SchemaValidationError(f"{path}: {data} out of bounds")
    if isinstance(data, list):
        if len(data) < schema.get("minItems", 0) or len(data) > schema.get("maxItems", len(data)):
            raise SchemaValidationError(f"{path}: {len(data)} items out of bounds")
        if "items" in schema:
            for i, item in enumerate(data):
                validate_json_schema(item, schema["items"], root=root, path=f"{path}[{i}]")
    if isinstance(data, dict):
        props = schema.get("properties", {})
        missing = [k for k in schema.get("required", []) if k not in data]
        if missing:
            raise SchemaValidationError(f"{path}: missing required {missing}")
        extra = schema.get("additionalProperties", True)
        for key, value in data.items():
            if key in props:
                validate_json_schema(value, props[key], root=root, path=f"{path}.{key}")
            elif extra is False:
                raise SchemaValidationError(f"{path}: unexpected property {key!r}")
            elif isinstance(extra, dict):
                validate_json_schema(value, extra, root=root, path=f"{path}.{key}")


def parse_structured(text: str, output: type[M] | dict) -> M | dict:
    """Parse model text (bare or fenced JSON) and validate it against a Pydantic model or a JSON Schema."""
    raw = extract_json(text)
    if isinstance(output, dict):
        try:
            data = json.loads(raw)
        except ValueError as exc:
            raise SchemaValidationError(f"output is not valid JSON: {exc}") from exc
        validate_json_schema(data, output)
        return data
    return output.model_validate_json(raw)


class StructuredLLM:
    def __init__(self, router: ModelRouter) -> None:
        self._router = router

    async def generate(self, output: type[M] | dict, *, agent_id: str, tier: ModelTier, system: str, prompt: str,
                       scope: ExecutionScope, input_payload: dict | None = None, max_output_tokens: int = 4000,
                       timeout_seconds: float = 60.0, validation_retries: int = 2,
                       check: Callable[[Any], None] | None = None) -> M | dict:
        """One structured generation. `check` adds semantic validation: raise ValueError to send the model a
        correction. Raises StructuredOutputError when no attempt produced valid output."""
        schema = output if isinstance(output, dict) else output.model_json_schema()
        messages = [LLMMessage(role="user", content=prompt)]
        last_error = ""
        attempts = validation_retries + 1
        for attempt in range(1, attempts + 1):
            response = await self._router.complete(
                tier=tier, agent_id=agent_id, system=system, messages=messages, response_schema=schema,
                max_output_tokens=max_output_tokens, input_payload=input_payload or {}, scope=scope,
                timeout_seconds=timeout_seconds, attempt=attempt,
            )
            try:
                result = parse_structured(response.text, output)
                if check is not None:
                    check(result)
                return result
            except (ValidationError, ValueError) as exc:
                last_error = str(exc)
                scope.emit(EventType.AGENT_VALIDATION_FAILED, attempt=attempt, error=last_error[:2000])
                messages = [
                    *messages,
                    LLMMessage(role="assistant", content=response.text),
                    LLMMessage(role="user", content=(
                        "Your previous output failed validation:\n"
                        f"{last_error}\nReturn only corrected JSON that matches the schema."
                    )),
                ]
        raise StructuredOutputError(f"{agent_id}: no valid output after {attempts} attempts: {last_error[:500]}",
                                    attempts=attempts, last_error=last_error)
