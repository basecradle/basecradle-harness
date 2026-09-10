"""Normalizing a tool schema onto an object root (`_schema`), issue #496.

The pathological schema these tests are written against is **not hand-written**: it is the verbatim
``tools/list`` response of the published ``mcp-mail-server@2.0.2`` tarball, captured by running the
server over stdio (`tests/data/mcp_mail_server_2_0_2_tools.json`, provenance inside the file). That
matters — the whole defect is a shape a *real* server emits and a hand-written approximation of it
would prove nothing about the real one.

The rest of the file pins the rule's edges: what merges, what is left alone, and what is refused so
that the caller can drop one tool instead of losing the wake.
"""

from __future__ import annotations

import copy

import pytest

from basecradle_harness._schema import UnrepresentableSchema, normalize_object_root
from tests.conftest import MAIL_SERVER_TOOLS


def _mail_tools() -> dict[str, dict]:
    return {tool["name"]: tool for tool in MAIL_SERVER_TOOLS["tools"]}


# --- the real mcp-mail-server@2.0.2 schema ------------------------------------


def test_the_fixture_still_carries_the_shape_that_broke_the_agent():
    # If this ever stops holding, the fixture was re-captured against a different build and every
    # other test in this file is quietly proving something else.
    schema = _mail_tools()["send_email"]["inputSchema"]
    assert schema["type"] == "object"
    assert schema["anyOf"] == [{"required": ["text"]}, {"required": ["html"]}]
    assert all("type" not in branch for branch in schema["anyOf"])  # the "non-object branch"
    assert "anyOf" in schema["properties"]["signature"]  # and again one level down


@pytest.mark.parametrize("name", ["send_email", "reply_to_email", "continue_email_thread"])
def test_the_union_root_becomes_a_plain_object_root(name):
    normalized, _ = normalize_object_root(_mail_tools()[name]["inputSchema"])
    assert normalized["type"] == "object"
    assert not any(k in normalized for k in ("anyOf", "oneOf", "allOf"))


def test_the_nested_union_is_normalized_too():
    # The root is what xAI named, but `signature` carries the identical shape — and a fix that left
    # it would have needed a second release to find out.
    normalized, _ = normalize_object_root(_mail_tools()["send_email"]["inputSchema"])
    signature = normalized["properties"]["signature"]
    assert signature["type"] == "object"
    assert "anyOf" not in signature


def test_every_property_survives_the_rewrite():
    original = _mail_tools()["send_email"]["inputSchema"]
    normalized, _ = normalize_object_root(original)
    assert set(normalized["properties"]) == set(original["properties"])
    assert normalized["required"] == original["required"]
    # The optional bodies stay optional: a disjunctive branch never promotes a name into `required`.
    assert "text" not in normalized["required"]
    assert "html" not in normalized["required"]


def test_the_constraint_the_schema_could_not_carry_comes_back_as_prose():
    _, notes = normalize_object_root(_mail_tools()["send_email"]["inputSchema"])
    assert "At least one of: (text) or (html)." in notes
    assert "`signature`: At least one of: (text) or (html)." in notes


def test_a_well_formed_schema_is_returned_byte_for_byte_with_no_notes():
    # The regression bar: eleven of the twelve mail tools are ordinary, and a tool xAI never had a
    # problem with must be offered exactly as it arrived.
    for name, tool in _mail_tools().items():
        if name in ("send_email", "reply_to_email", "continue_email_thread"):
            continue
        schema = tool["inputSchema"]
        normalized, notes = normalize_object_root(schema)
        assert normalized == schema, name
        assert notes == [], name


def test_the_input_schema_is_never_mutated():
    # An MCP tool's `parameters` is the live `Tool` object's own dict; rewriting it in place would
    # change what every *other* provider is offered.
    original = _mail_tools()["send_email"]["inputSchema"]
    before = copy.deepcopy(original)
    normalize_object_root(original)
    assert original == before


# --- the rule's edges ---------------------------------------------------------


def test_a_union_of_object_branches_merges_into_one_object():
    schema = {
        "anyOf": [
            {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]},
            {"type": "object", "properties": {"b": {"type": "number"}}, "required": ["b"]},
        ]
    }
    normalized, notes = normalize_object_root(schema)
    assert normalized["type"] == "object"
    assert set(normalized["properties"]) == {"a", "b"}
    # Neither is required: only one branch applies, so demanding both would reject a legal call.
    assert "required" not in normalized
    assert notes == ["At least one of: (a) or (b)."]


def test_one_of_says_exactly_one_where_any_of_says_at_least_one():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
        "oneOf": [{"required": ["a"]}, {"required": ["b"]}],
    }
    _, notes = normalize_object_root(schema)
    assert notes == ["Exactly one of: (a) or (b)."]


def test_all_of_is_conjunctive_so_its_required_really_is_required():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "allOf": [{"properties": {"b": {"type": "string"}}, "required": ["b"]}],
    }
    normalized, notes = normalize_object_root(schema)
    assert set(normalized["properties"]) == {"a", "b"}
    assert normalized["required"] == ["b"]
    assert notes == []  # nothing was lost, so there is nothing to tell the model


def test_a_nested_type_union_is_left_exactly_as_it_is():
    # Ordinary, useful JSON Schema. Flattening it would throw away what the model needs to know, and
    # nothing about it is refused for being a non-object — the vendor named the *root*.
    schema = {
        "type": "object",
        "properties": {"value": {"anyOf": [{"type": "string"}, {"type": "number"}]}},
    }
    normalized, notes = normalize_object_root(schema)
    assert normalized["properties"]["value"] == {"anyOf": [{"type": "string"}, {"type": "number"}]}
    assert notes == []


def test_an_untyped_root_with_properties_is_stated_as_an_object():
    normalized, _ = normalize_object_root({"properties": {"a": {"type": "string"}}})
    assert normalized["type"] == "object"


def test_an_empty_root_becomes_the_explicit_no_arguments_schema():
    normalized, _ = normalize_object_root({})
    assert normalized == {"type": "object", "properties": {}}


def test_subschemas_inside_a_kept_union_are_still_normalized():
    schema = {
        "type": "object",
        "properties": {
            "value": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "object", "properties": {"x": {}}, "anyOf": [{"required": ["x"]}]},
                ]
            }
        },
    }
    normalized, notes = normalize_object_root(schema)
    branch = normalized["properties"]["value"]["anyOf"][1]
    assert "anyOf" not in branch
    assert notes == ["`value.anyOf.1`: At least one of: (x)."]


# --- what is refused, so one tool is dropped and the wake survives -------------
#
# Exactly two shapes, and both are quoted from xAI's own message. Everything else this module cannot
# fold goes to the wire as the server wrote it, because the two failure directions are not close:
# refusing too eagerly takes away a capability the vendor would have accepted, while refusing too
# little costs one request that the caller's reactive drop then settles.


def test_a_non_object_root_type_is_refused():
    # "tool parameter root must be an object type"
    with pytest.raises(UnrepresentableSchema) as caught:
        normalize_object_root({"type": "string"})
    assert "not an object" in str(caught.value)


def test_a_root_union_with_a_branch_that_declares_a_non_object_type_is_refused():
    # "root schema is an anyOf/oneOf union with a non-object branch"
    with pytest.raises(UnrepresentableSchema) as caught:
        normalize_object_root(
            {"anyOf": [{"type": "object", "properties": {"a": {}}}, {"type": "string"}]}
        )
    assert "non-object branch" in str(caught.value)


def test_a_root_that_is_not_a_schema_object_at_all_is_refused():
    with pytest.raises(UnrepresentableSchema):
        normalize_object_root(True)
    with pytest.raises(UnrepresentableSchema):
        normalize_object_root("nonsense")


def test_a_root_that_is_only_a_ref_goes_to_the_wire_untouched():
    # Not attested: xAI never said it refuses a `$ref` root, and resolving one would mean writing a
    # `$ref`/`$id` resolver for a case no server has produced. So it is neither rewritten nor
    # refused — the vendor judges it, and its verdict costs one tool at most.
    schema = {"$ref": "#/$defs/Thing", "$defs": {"Thing": {"type": "object"}}}
    normalized, notes = normalize_object_root(schema)
    assert normalized == schema
    assert notes == []


def test_a_root_all_of_that_cannot_be_folded_is_left_alone_rather_than_refused():
    # The regression this calibration exists for: `allOf` of `$ref`s is what Pydantic emits every
    # day, xAI has said nothing against it, and dropping the tool over it would silently narrow the
    # agent. Note the root keeps its own `type` and gains no invented `properties`.
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "allOf": [{"$ref": "#/$defs/Base"}],
    }
    normalized, notes = normalize_object_root(schema)
    assert normalized == schema
    assert notes == []


def test_an_unfoldable_union_root_is_never_stamped_into_an_argument_less_object():
    # `oneOf` of `$ref` branches: nothing declares a non-object type, so it is not refused — and it
    # must not be "helpfully" typed either. Stamping `{"type": "object", "properties": {}}` on it
    # would hand the model a tool with no arguments, which is worse than any request the vendor
    # might reject.
    schema = {"oneOf": [{"$ref": "#/$defs/A"}, {"$ref": "#/$defs/B"}]}
    normalized, notes = normalize_object_root(schema)
    assert normalized == schema
    assert notes == []


def test_one_unfoldable_combinator_does_not_stop_the_others_folding():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "anyOf": [{"required": ["a"]}, {"required": ["b"]}],
        "allOf": [{"$ref": "#/$defs/Base"}],
    }
    normalized, notes = normalize_object_root(schema)
    assert "anyOf" not in normalized
    assert normalized["allOf"] == [{"$ref": "#/$defs/Base"}]
    assert notes == ["At least one of: (a) or (b)."]


def test_a_nested_branch_this_module_cannot_fold_costs_nothing_at_all():
    # Nested, there is a weaker outcome than refusing the tool, and it is taken: the node is handed
    # back exactly as the server wrote it, and no half-rewrite and no note is left behind.
    inner = {
        "type": "object",
        "properties": {"a": {}},
        "anyOf": [{"required": ["a"]}, {"type": "string"}],
    }
    schema = {"type": "object", "properties": {"nested": copy.deepcopy(inner)}}
    normalized, notes = normalize_object_root(schema)
    assert normalized["properties"]["nested"] == inner
    assert notes == []
