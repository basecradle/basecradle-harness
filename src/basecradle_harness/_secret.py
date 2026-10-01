"""A credential held by a harness object, kept out of every representation of that object.

Issue #599 (class: basecradle#612): *a generic serializer or representation of an object holding a
credential emits the credential.* A key held as a plain attribute sits in the object's ``__dict__``,
and that one fact is what makes ``vars()``, a crash reporter's frame expansion (Sentry, ``rich``,
``cgitb`` render locals one level deep through ``__dict__``), ``json.dumps(vars(o), default=str)``
and ``json.dump(o, fp, default=vars)`` emit it — the last one *even when it then raises* on a
circular reference, because the bytes are already on the stream. It is also what lets
``__reduce__`` hand the key to ``pickle``. ``repr`` never needed the attribute to be public to leak:
a dataclass's generated ``__repr__`` prints every field.

`Secret` closes all of those at the value rather than at each holder. It has no ``__dict__`` (a
slot), so a walk that reaches it finds nothing to expand; its ``repr`` and ``str`` are a fixed
``[REDACTED]``; and it refuses ``__reduce__`` with a `TypeError` naming the risk, which is the error
``pickle`` already raises for an object it cannot serialize. A **copy** is allowed and returns the
same object: the value is immutable, a copy duplicates no bytes, and refusing it would break
``dataclasses.asdict`` and ``copy.deepcopy`` of a holder for no gain — the leak was never the copy,
it was the plain attribute the copy carried. `reveal` is the one way out, called on the line that
hands the key to a vendor.

What it does not do, stated: a credential a **vendor SDK** keeps on its own client object (the
``openai`` client's ``api_key``, say) is that vendor's attribute, reached only through our private
``_client``; reimplementing vendor clients is not this module's job.
"""

from __future__ import annotations

import hmac
from typing import Any

_REDACTED = "[REDACTED]"


class Secret:
    """A credential string that never appears in a representation of whatever holds it.

    >>> key = Secret("sk-example")
    >>> key
    Secret('[REDACTED]')
    >>> key.reveal()
    'sk-example'
    """

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def reveal(self) -> str:
        """The credential itself — for the call that sends it, and nothing else."""
        return self._value

    def __repr__(self) -> str:
        return f"Secret({_REDACTED!r})"

    __str__ = __repr__

    def __bool__(self) -> bool:
        return bool(self._value)

    def __eq__(self, other: object) -> bool:
        # Equality is what lets two configs holding the same key compare equal; compared in
        # constant time because it is a credential. No `__hash__`: nothing needs one, and a hash of
        # the value is one more thing derived from it.
        if not isinstance(other, Secret):
            return NotImplemented
        return hmac.compare_digest(self._value.encode(), other._value.encode())

    __hash__ = None  # type: ignore[assignment]

    def __reduce__(self) -> Any:
        raise TypeError(
            "A Secret cannot be serialized: it holds a credential, and pickling it would write "
            "the credential into the bytes. Rebuild the object holding it from its environment "
            "instead of serializing it."
        )

    def __copy__(self) -> Secret:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Secret:
        return self


def secret(value: str | None) -> Secret | None:
    """`value` held as a `Secret`, or ``None`` when there is no credential (``None`` or empty).

    The harness's credential arguments are optional — ``None`` means *read the environment at call
    time* — so a holder wraps through this rather than branching at every constructor.
    """
    return Secret(value) if value else None


def reveal(value: Secret | None) -> str | None:
    """The credential inside `value`, or ``None`` — the counterpart to `secret`."""
    return value.reveal() if value is not None else None
