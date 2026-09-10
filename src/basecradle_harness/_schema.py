"""Make a tool's JSON Schema acceptable to a vendor that demands an **object root** (issue #496).

The recurrence guard this closes: **one tool's schema a vendor refuses took the whole agent down.**
`@briggs` (`xai-sdk`, `grok-4.6`) died on *every* wake with

    xAI gRPC error (INVALID_ARGUMENT): Failed to start sampling: [invalid_client_tool_schema]
    workmail__send_email: tool parameter root must be an object type
    (root schema is an anyOf/oneOf union with a non-object branch)

— retried by the router, `posted=0` every time. The same MCP server (`mcp-mail-server@2.0.2`), the
same schema, was accepted by OpenAI (`@jt`) and OpenRouter (`@glm-5.2`) in the same verify, so the
server is not broken: it is an **adapter difference**, and the harness's job at that boundary is to
translate what it honestly can and to fail **per tool**, never per wake — the "stall is a drop"
class, and "don't show a locked door" applied at the vendor boundary.

What the observed schema actually is
------------------------------------
Narrower than "a union root", and the difference is the whole design. The root **is** an object —
``type: "object"`` with ``properties`` and ``required`` — and it carries a *sibling* combinator::

    {"type": "object", "properties": {...}, "required": ["to", "subject"],
     "anyOf": [{"required": ["text"]}, {"required": ["html"]}]}

`zod`'s ``.refine(…)`` cannot be expressed in JSON Schema, so `mcp-mail-server` states the
"at least one of text or html" rule through ``.meta({anyOf: […]})`` — branches carrying **only** a
``required`` list and no ``type``. That is valid JSON Schema and every other endpoint honors it;
xAI's validator reads the root's ``anyOf`` as a union, finds a branch that is not an object, and
refuses the whole request. The same shape appears one level down under ``properties.signature``,
which is why the transform is not root-only: a root-only fix would have needed a second release to
discover that.

The rule, in one sentence
-------------------------
**Fold a combinator into the object it sits beside, refuse only what the vendor has actually said it
refuses, and leave everything else exactly as the server wrote it.**

Concretely, at a node this module will rewrite (`_normalize`):

- ``allOf`` — every branch applies, so its branches are merged **whole**: properties union,
  ``required`` union. That is the standard reading and loses nothing.
- ``anyOf`` / ``oneOf`` — at most one branch applies, so a branch's properties are merged in as
  **optional** and a branch's ``required`` never becomes the parent's ``required``. Requiring a
  key the model may legitimately omit would turn a schema the vendor merely disliked into one that
  is *wrong*.
- The constraint the merge cannot express is not thrown away: it becomes a human-readable **note**
  ("At least one of: (text) or (html)"), returned to the caller to append to the *function*
  description — the one string every vendor puts in front of the model. The MCP server still
  enforces the real rule (that is what `zod`'s ``.refine`` is), so the model reads the requirement
  in prose and a mistake comes back as the server's own error, not as a silent wrong send.

**Only two things are refused, and both are quoted from the vendor** (`_refuse_if_not_an_object`):
a root that declares a ``type`` other than ``object`` (*"tool parameter root must be an object
type"*), and a root ``anyOf``/``oneOf`` carrying a branch that declares a non-object ``type``
(*"root schema is an anyOf/oneOf union with a non-object branch"*).

Why the refusal set is that narrow, and not "anything I could not fold"
-----------------------------------------------------------------------
Because the two failure directions are wildly asymmetric, and the first draft of this module had
them backwards. A pre-flight that refuses **too eagerly** takes away a capability the vendor would
have accepted — a root ``allOf`` of ``$ref`` branches is what Pydantic emits every day, and nothing
xAI said suggests it minds — and the agent then simply cannot do a thing it could do yesterday. A
pre-flight that is **too lenient** costs one refused request, after which the caller's reactive drop
(keyed on the vendor naming the tool) removes it anyway. One wasted call against a silently narrowed
agent is not a close contest. So a combinator this module cannot fold is **left in place**, per
keyword, and the vendor decides — the same principle the reranker's fault classes already turn on:
*the vendor's error text is the authority on its own faults.*

Where it applies, and where it deliberately does not
----------------------------------------------------
The **root** is where the vendor's demand lives, so it is the only place anything is refused. A
**nested** node is folded only when it is *already* object-shaped (``type: "object"``, or it has
``properties``) and a combinator is sitting beside it — precisely the pathological shape above. A
nested node whose combinator is a genuine type union
(``anyOf: [{"type": "string"}, {"type": "number"}]``) is **left exactly as it is**: that is ordinary,
useful JSON Schema, nothing about it is refused for being a non-object, and flattening it would
throw away information the model needs. The transform is a repair of one specific pathology, not a
general schema rewriter.

Why nothing is wrapped under a synthetic property
-------------------------------------------------
The obvious alternative for a genuinely non-object root — wrap it as
``{"type":"object","properties":{"input": <original>},"required":["input"]}`` and unwrap on the way
back — was considered and rejected. It would make the adapter reshape the model's *arguments* on the
return path, and those arguments are read by three other mechanisms that must agree on their shape:
`_idempotency.create_kind` (an ordinal that drifts is a message posted twice), the transcript's
replayability exception (`_session._replayable`, which re-posts an interrupted create *from* its
arguments), and the resume. A hidden reshaping under all three is a large new surface for a case
nobody has observed. **Refusing that one tool, loudly, is the honest answer** — and the caller's
per-tool drop is what keeps the cost of it bounded to the tool rather than the wake.

Fails safe in the direction that matters
----------------------------------------
Every operation here is best-effort over untrusted, server-authored data: a schema that is not a
`dict`, a combinator that is not a list, a branch that is not a `dict`. Nothing raises except the two
attested refusals above. A shape this module does not understand is passed through unchanged, and a
refusal costs exactly one tool.
"""

from __future__ import annotations

from typing import Any

#: The combinator keywords this module folds. ``allOf`` is conjunctive (every branch applies); the
#: other two are disjunctive (at most one does), which is why their branches contribute properties
#: but never `required` — and why only *they* carry the vendor's attested root refusal.
_COMBINATORS = ("allOf", "anyOf", "oneOf")
_DISJUNCTIVE = ("anyOf", "oneOf")

#: How each disjunctive keyword reads in the note the model is given. ``anyOf`` is "at least one";
#: ``oneOf`` is "exactly one" — a distinction the model can act on, so it is not flattened away.
_NOTE_LEAD = {"anyOf": "At least one of", "oneOf": "Exactly one of"}

#: Keywords that carry a *subschema* worth recursing into, and the shape they carry it in.
_CHILD_MAPS = ("properties", "patternProperties", "$defs", "definitions")
_CHILD_ONE = ("items", "additionalProperties", "contains", "not", "propertyNames")
_CHILD_LISTS = ("prefixItems",)

#: How deep this walks before it stops and hands the rest back untouched. The input is a *server's*
#: JSON, so its depth is not the harness's to choose, and a `RecursionError` raised out of here would
#: kill the wake — which is the one outcome this whole module exists to prevent. Far past any real
#: tool schema (`mcp-mail-server`'s deepest is 3), so nothing legitimate ever meets it.
_MAX_DEPTH = 64


class UnrepresentableSchema(Exception):
    """This schema's root is not an object and cannot be made one, so the tool cannot be offered.

    Raised only for the two shapes the vendor has itself named (see the module docstring), and
    carried to the caller so the *reason* reaches the log line beside the tool's name — a drop
    nobody can explain is a capability that vanished, which is the failure class this whole module
    exists inside.
    """


def normalize_object_root(schema: Any) -> tuple[dict[str, Any], list[str]]:
    """Rewrite `schema` so its root is a plain object type, and say what that cost.

    Returns ``(schema, notes)`` — a **new** schema (the input is never mutated; an MCP tool's
    ``parameters`` is shared with the live `Tool` object and with every *other* provider's offer) and
    the human-readable notes for any constraint the rewrite could not express, each already prefixed
    with the path it applied at. An unchanged schema comes back equal to its input with no notes,
    which is what lets a caller offer a well-formed schema byte-for-byte as it arrived.

    Raises `UnrepresentableSchema` for a root the vendor has said it will not take — the caller's cue
    to drop that one tool and carry on with the rest.
    """
    notes: list[str] = []
    result = _normalize(schema, path="", notes=notes, root=True, depth=0)
    return result, notes


def _normalize(node: Any, *, path: str, notes: list[str], root: bool, depth: int) -> Any:
    """Normalize one node, recursing into its subschemas. Returns a new node."""
    if depth > _MAX_DEPTH and not root:
        return node
    if not isinstance(node, dict):
        # `true`/`false` schemas and anything malformed. At the root there is nothing here to build
        # an object from; nested, it is not ours to touch.
        if root:
            raise UnrepresentableSchema(
                f"its root is {type(node).__name__}, not a JSON-Schema object"
            )
        return node

    if root:
        _refuse_if_not_an_object(node)
        if "$ref" in node:
            # A root that only points elsewhere. Resolving it would mean implementing `$ref`, remote
            # refs and `$id` bases for a case no server has produced — and the vendor has *not* said
            # it refuses one, so it goes to the wire exactly as written and xAI judges it.
            return dict(node)

    node = dict(node)
    if _should_fold(node, root=root):
        node = _fold(node, path=path, notes=notes)

    node = _recurse(node, path=path, notes=notes, depth=depth)

    if root:
        node = _state_object_root(node)
    return node


def _refuse_if_not_an_object(node: dict[str, Any]) -> None:
    """The two refusals, both quoted from the vendor's own message (issue #496).

    Nothing else is refused here — see the module docstring for why the set is this narrow. Called
    before anything is rewritten, so a refusal has changed nothing and published no notes.
    """
    declared = node.get("type")
    if declared is not None:
        stated = declared if isinstance(declared, list) else [declared]
        if "object" not in stated:
            raise UnrepresentableSchema(f"its root type is {declared!r}, not an object")
    for keyword in _DISJUNCTIVE:
        branches = node.get(keyword)
        if not isinstance(branches, list):
            continue
        for branch in branches:
            if _states_a_non_object(branch):
                raise UnrepresentableSchema(
                    f"its root {keyword} carries a non-object branch (type={branch.get('type')!r})"
                )


def _states_a_non_object(branch: Any) -> bool:
    """Does this branch *declare* a type that is not an object?

    Deliberately a claim about what the branch **says**, not about what it is. A branch with no
    ``type`` (the `mcp-mail-server` shape) or one that is only a ``$ref`` states nothing, so it is
    not this — the vendor said *non-object branch*, and inferring one from silence is exactly the
    over-eager refusal this module is calibrated against.
    """
    if not isinstance(branch, dict):
        return False
    declared = branch.get("type")
    if declared is None:
        return False
    stated = declared if isinstance(declared, list) else [declared]
    return "object" not in stated


def _should_fold(node: dict[str, Any], *, root: bool) -> bool:
    """Is this a node whose combinator should be folded in?

    The root's, because that is where the vendor's demand lives. A nested node's only when the node
    is *already* object-shaped, which is the one pathology this module repairs; a nested type union
    is ordinary schema and is left alone.
    """
    if not any(k in node for k in _COMBINATORS):
        return False
    return root or _is_object_shaped(node)


def _is_object_shaped(node: dict[str, Any]) -> bool:
    """Does this node already declare itself an object?"""
    declared = node.get("type")
    if declared == "object" or (isinstance(declared, list) and "object" in declared):
        return True
    return "properties" in node


def _fold(node: dict[str, Any], *, path: str, notes: list[str]) -> dict[str, Any]:
    """Fold this node's combinators into the node itself, noting what could not be expressed.

    **Per keyword, never all-or-nothing:** a combinator whose branches cannot all be merged is left
    exactly where it is and the others still fold. That is what keeps an unfoldable ``allOf`` — a
    root ``allOf`` of ``$ref``s, say — from costing a tool the vendor never objected to.
    """
    properties: dict[str, Any] = dict(node.get("properties") or {})
    required: list[str] = [r for r in (node.get("required") or []) if isinstance(r, str)]

    for keyword in _COMBINATORS:
        branches = node.get(keyword)
        if not isinstance(branches, list) or not all(_foldable(b) for b in branches):
            continue
        node.pop(keyword)
        alternatives: list[str] = []
        for branch in branches:
            for name, sub in (branch.get("properties") or {}).items():
                # First branch wins on a name two alternatives both declare: the merged schema can
                # only carry one shape, and the alternatives are named in the note either way.
                properties.setdefault(name, sub)
            names = [r for r in (branch.get("required") or []) if isinstance(r, str)]
            if keyword == "allOf":
                # Conjunctive: every branch applies, so its `required` genuinely is required.
                required.extend(n for n in names if n not in required)
            elif names:
                alternatives.append("(" + ", ".join(names) + ")")
        if alternatives:
            lead = _NOTE_LEAD.get(keyword, "One of")
            where = f"`{path}`: " if path else ""
            notes.append(f"{where}{lead}: {' or '.join(alternatives)}.")

    if properties:
        node["properties"] = properties
    if required:
        node["required"] = required
    return node


def _foldable(branch: Any) -> bool:
    """Can this branch be merged into an object — object-shaped, or pure constraint?

    A branch stating ``{"required": ["text"]}`` and nothing else constrains an object without
    declaring one; that is the `mcp-mail-server` shape and it merges cleanly. A branch that declares
    a *different* type cannot, and neither can one that only points elsewhere (`$ref`) or is not a
    schema object at all.
    """
    if not isinstance(branch, dict) or "$ref" in branch:
        return False
    declared = branch.get("type")
    if declared is None:
        return True
    if isinstance(declared, list):
        return "object" in declared
    return declared == "object"


def _recurse(node: dict[str, Any], *, path: str, notes: list[str], depth: int) -> dict[str, Any]:
    """Normalize every subschema this node carries, leaving its own keywords alone."""
    for key in _CHILD_MAPS:
        children = node.get(key)
        if isinstance(children, dict):
            node[key] = {
                name: _normalize(
                    sub, path=_join(path, key, name), notes=notes, root=False, depth=depth + 1
                )
                for name, sub in children.items()
            }
    for key in _CHILD_ONE:
        child = node.get(key)
        if isinstance(child, dict):
            node[key] = _normalize(
                child, path=_join(path, key), notes=notes, root=False, depth=depth + 1
            )
    for key in (*_CHILD_LISTS, *_COMBINATORS):
        # A combinator still present here was deliberately *not* folded (a nested type union, or one
        # whose branches could not be merged), so its branches are still live schemas the model
        # reads — normalize inside them, never past them.
        children = node.get(key)
        if isinstance(children, list):
            node[key] = [
                _normalize(
                    sub, path=_join(path, key, str(i)), notes=notes, root=False, depth=depth + 1
                )
                for i, sub in enumerate(children)
            ]
    return node


def _join(path: str, *parts: str) -> str:
    """The dotted path of a subschema, for the note that names where a constraint applied.

    Cosmetic — it exists so a note reads ``signature: At least one of …`` rather than floating free
    — so the structural keyword that carries no meaning for a reader (``properties``) is dropped.
    """
    kept = [p for p in parts if p != "properties"]
    return ".".join([p for p in (path, *kept) if p])


def _state_object_root(node: dict[str, Any]) -> dict[str, Any]:
    """Say ``type: "object"`` where that is plainly what the root is, and stay quiet otherwise.

    An untyped root carrying ``properties`` is the type every consumer already infers, and an empty
    root is the explicit no-arguments schema. A root still holding a combinator this module could not
    fold is left **untouched**: stamping ``type: "object"`` (and an empty ``properties``) onto a
    union would be a claim about arguments nobody made, and would take the tool's real parameters
    away from the model — a far worse outcome than the request the vendor may or may not reject.
    """
    if "type" in node:
        return node
    if "properties" in node or not any(k in node for k in _COMBINATORS):
        node["type"] = "object"
        node.setdefault("properties", {})
    return node
