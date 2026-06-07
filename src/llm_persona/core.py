"""llm_persona — named persona registry for LLM agents.

Define reusable :class:`Persona` objects with system prompts, tags, and
metadata, then look them up by name via :class:`PersonaRegistry`.
Personas can also carry request parameters (model, temperature, and arbitrary
extra keyword arguments) and produce ready-to-send request payloads via
:meth:`Persona.apply` / :meth:`PersonaRegistry.apply`.

Zero dependencies.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

# A chat message is a plain dict, e.g. ``{"role": "user", "content": "hi"}``.
Message = dict[str, Any]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PersonaNotFoundError(KeyError):
    """Raised when a persona name is not in the registry."""


# Backwards/alternate-friendly alias. Both names refer to the same class so
# ``except PersonaNotFound`` and ``except PersonaNotFoundError`` both work.
PersonaNotFound = PersonaNotFoundError


# ---------------------------------------------------------------------------
# Persona
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Persona:
    """An immutable named LLM persona.

    Args:
        name: Unique persona identifier. Must not be empty.
        system_prompt: The system prompt text. Must not be empty.
        description: Optional human-readable description.
        tags: Sequence of string tags (stored as a tuple).
        metadata: Arbitrary extra key/value pairs (informational only).
        model: Optional default model id for requests built from this persona.
        temperature: Optional default sampling temperature.
        extra: Optional extra request parameters merged into payloads built by
            :meth:`apply` (for example ``{"max_tokens": 1024}``).

    Raises:
        ValueError: If *name* or *system_prompt* is empty.
    """

    name: str
    system_prompt: str
    description: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)
    model: str | None = None
    temperature: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty")
        if not self.system_prompt:
            raise ValueError("system_prompt must not be empty")
        # Normalise tags to tuple regardless of input type
        object.__setattr__(self, "tags", tuple(self.tags))
        # Deep-copy mutable inputs so callers can't mutate our internals
        object.__setattr__(self, "metadata", copy.deepcopy(self.metadata))
        object.__setattr__(self, "extra", copy.deepcopy(self.extra))

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def as_message(self) -> dict[str, str]:
        """Return as an OpenAI-style system message dict.

        Returns:
            ``{"role": "system", "content": <system_prompt>}``
        """
        return {"role": "system", "content": self.system_prompt}

    # Alias for callers that prefer the more explicit name.
    as_system_message = as_message

    def inject_system(self, messages: list[Message]) -> list[Message]:
        """Return a *new* message list with this persona's system message first.

        The input list is never mutated.

        Args:
            messages: Existing chat messages (OpenAI-style role/content dicts).

        Returns:
            A new list: ``[system_message, *messages]``.
        """
        return [self.as_message(), *messages]

    def apply(
        self,
        messages: list[Message],
        *,
        include_model: bool = True,
    ) -> dict[str, Any]:
        """Build a request-kwargs dict for this persona.

        The returned dict is suitable for splatting into an LLM client call,
        e.g. ``client.messages.create(**persona.apply(messages))``. It uses the
        Anthropic-style ``system`` parameter (a top-level string) rather than a
        system message embedded in ``messages``.

        Keys are populated as follows:

        * ``system`` — always the persona's :attr:`system_prompt`.
        * ``messages`` — the *messages* argument, unchanged.
        * ``model`` — included only if set and *include_model* is true.
        * ``temperature`` — included only if set (not ``None``).
        * any keys from :attr:`extra` are merged in.

        Args:
            messages: The chat messages to send.
            include_model: When ``False``, omit ``model`` even if the persona
                defines one (useful when the caller pins the model elsewhere).

        Returns:
            A new dict of request keyword arguments.
        """
        payload: dict[str, Any] = dict(self.extra)
        payload["system"] = self.system_prompt
        payload["messages"] = messages
        if self.model is not None and include_model:
            payload["model"] = self.model
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        return payload

    # ------------------------------------------------------------------
    # Builders (return new Persona)
    # ------------------------------------------------------------------

    def with_metadata(self, **extra: Any) -> "Persona":
        """Return a new Persona with *extra* merged into metadata.

        Existing keys are preserved; *extra* values take precedence.

        Args:
            **extra: Additional metadata key/value pairs.

        Returns:
            A new :class:`Persona` with merged metadata.
        """
        merged = {**self.metadata, **extra}
        return Persona(
            name=self.name,
            system_prompt=self.system_prompt,
            description=self.description,
            tags=self.tags,
            metadata=merged,
            model=self.model,
            temperature=self.temperature,
            extra=self.extra,
        )

    def with_tags(self, *tags: str) -> "Persona":
        """Return a new Persona with *tags* added (duplicates removed).

        Args:
            *tags: Tags to add.

        Returns:
            A new :class:`Persona` with the combined, deduplicated tag set.
        """
        combined: list[str] = list(self.tags)
        seen = set(self.tags)
        for t in tags:
            if t not in seen:
                combined.append(t)
                seen.add(t)
        return Persona(
            name=self.name,
            system_prompt=self.system_prompt,
            description=self.description,
            tags=tuple(combined),
            metadata=self.metadata,
            model=self.model,
            temperature=self.temperature,
            extra=self.extra,
        )

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Persona(name={self.name!r}, "
            f"system_prompt={self.system_prompt!r})"
        )

    def __str__(self) -> str:
        return f"Persona<{self.name}>"


# ---------------------------------------------------------------------------
# PersonaRegistry
# ---------------------------------------------------------------------------


class PersonaRegistry:
    """A mutable registry of named :class:`Persona` objects.

    Usage::

        registry = PersonaRegistry()
        registry.register("helper", "You are a helpful assistant.")
        kwargs = registry.apply("helper", messages)
        # response = client.messages.create(**kwargs)
    """

    def __init__(self) -> None:
        self._store: dict[str, Persona] = {}

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def register(
        self,
        persona_or_name: "Persona | str",
        system_prompt: str = "",
        *,
        description: str = "",
        tags: list[str] | tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        **extra: Any,
    ) -> "PersonaRegistry":
        """Register a persona.

        Accepts either a :class:`Persona` object or a ``(name, system_prompt)``
        shorthand. With the shorthand form, any unknown keyword arguments are
        collected into the persona's :attr:`Persona.extra` request parameters
        (for example ``register("p", "X", max_tokens=512)``).

        Args:
            persona_or_name: A :class:`Persona` instance, or the persona name
                string when using the shorthand form.
            system_prompt: Required when *persona_or_name* is a string.
            description: Optional description (shorthand form only).
            tags: Optional tags (shorthand form only).
            metadata: Optional metadata dict (shorthand form only).
            model: Optional default model id (shorthand form only).
            temperature: Optional default temperature (shorthand form only).
            **extra: Extra request parameters (shorthand form only).

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: If a persona with the same name is already registered,
                or if using the shorthand form without a system_prompt.
        """
        if isinstance(persona_or_name, Persona):
            p = persona_or_name
        else:
            if not system_prompt:
                raise ValueError(
                    "system_prompt is required when registering by name"
                )
            p = Persona(
                name=persona_or_name,
                system_prompt=system_prompt,
                description=description,
                tags=tuple(tags),
                metadata=metadata or {},
                model=model,
                temperature=temperature,
                extra=extra,
            )
        if p.name in self._store:
            raise ValueError(
                f"Persona {p.name!r} already registered; use upsert() to replace"
            )
        self._store[p.name] = p
        return self

    def unregister(self, name: str) -> "PersonaRegistry":
        """Remove a persona by name.

        Args:
            name: Persona name to remove.

        Returns:
            ``self`` for chaining.

        Raises:
            PersonaNotFoundError: If *name* is not registered.
        """
        if name not in self._store:
            raise PersonaNotFoundError(name)
        del self._store[name]
        return self

    def remove(self, name: str) -> "PersonaRegistry":
        """Remove a persona by name if present; a no-op when it is not.

        Unlike :meth:`unregister`, this never raises for a missing name.

        Args:
            name: Persona name to remove.

        Returns:
            ``self`` for chaining.
        """
        self._store.pop(name, None)
        return self

    def update(self, persona: Persona) -> "PersonaRegistry":
        """Replace an existing persona (must already exist).

        Args:
            persona: Replacement :class:`Persona`.

        Returns:
            ``self`` for chaining.

        Raises:
            PersonaNotFoundError: If the persona is not already registered.
        """
        if persona.name not in self._store:
            raise PersonaNotFoundError(persona.name)
        self._store[persona.name] = persona
        return self

    def upsert(self, persona: Persona) -> "PersonaRegistry":
        """Insert or replace a persona unconditionally.

        Args:
            persona: :class:`Persona` to store.

        Returns:
            ``self`` for chaining.
        """
        self._store[persona.name] = persona
        return self

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(
        self,
        name: str,
        default: "Persona | None" = None,
    ) -> "Persona | None":
        """Return a persona by name, or *default* if not found.

        Args:
            name: Persona name.
            default: Value to return when not found. Defaults to ``None``.

        Returns:
            The :class:`Persona` or *default*.
        """
        return self._store.get(name, default)

    # ``get_or_none`` is an explicit alias of ``get`` (default ``None``).
    def get_or_none(self, name: str) -> "Persona | None":
        """Return a persona by name, or ``None`` if not found.

        Args:
            name: Persona name.

        Returns:
            The :class:`Persona` or ``None``.
        """
        return self._store.get(name)

    def require(self, name: str) -> Persona:
        """Return a persona by name, raising if not found.

        Args:
            name: Persona name.

        Returns:
            The registered :class:`Persona`.

        Raises:
            PersonaNotFoundError: If *name* is not registered.
        """
        p = self._store.get(name)
        if p is None:
            raise PersonaNotFoundError(name)
        return p

    def has(self, name: str) -> bool:
        """Return ``True`` if *name* is registered."""
        return name in self._store

    def names(self) -> list[str]:
        """Return all registered persona names, sorted."""
        return sorted(self._store)

    def by_tag(self, tag: str) -> list[str]:
        """Return names of all personas that have *tag*, sorted.

        Args:
            tag: Tag string to filter by.

        Returns:
            Sorted list of matching persona names.
        """
        return sorted(n for n, p in self._store.items() if tag in p.tags)

    def all_tags(self) -> list[str]:
        """Return a sorted, deduplicated list of all tags across all personas."""
        tags: set[str] = set()
        for p in self._store.values():
            tags.update(p.tags)
        return sorted(tags)

    def all_personas(self) -> list[Persona]:
        """Return all personas sorted by name."""
        return [self._store[n] for n in sorted(self._store)]

    def system_prompt(self, name: str) -> str:
        """Return the system prompt for *name*, raising if not found.

        Args:
            name: Persona name.

        Returns:
            The persona's system prompt string.

        Raises:
            PersonaNotFoundError: If *name* is not registered.
        """
        return self.require(name).system_prompt

    def apply(
        self,
        name: str,
        messages: list[Message],
        *,
        include_model: bool = True,
    ) -> dict[str, Any]:
        """Build request kwargs for the named persona.

        Equivalent to ``registry.require(name).apply(messages, ...)``.

        Args:
            name: Persona name.
            messages: The chat messages to send.
            include_model: When ``False``, omit ``model`` even if set.

        Returns:
            A new dict of request keyword arguments.

        Raises:
            PersonaNotFoundError: If *name* is not registered.
        """
        return self.require(name).apply(messages, include_model=include_model)

    # ------------------------------------------------------------------
    # Properties / dunders
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Number of registered personas."""
        return len(self._store)

    @property
    def is_empty(self) -> bool:
        """``True`` if no personas are registered."""
        return len(self._store) == 0

    def __len__(self) -> int:
        return self.count

    def __contains__(self, name: object) -> bool:
        return name in self._store

    def __getitem__(self, name: str) -> Persona:
        """Return a persona by name, raising :class:`PersonaNotFoundError`."""
        return self.require(name)

    def __iter__(self):
        """Iterate over personas in sorted-name order."""
        return iter(self.all_personas())

    def __repr__(self) -> str:
        return f"PersonaRegistry(count={self.count})"


# ---------------------------------------------------------------------------
# Module-level convenience API (a shared default registry)
# ---------------------------------------------------------------------------

#: A process-wide default registry used by the module-level helpers below.
default_registry = PersonaRegistry()


def register(
    name: str,
    system_prompt: str,
    **kwargs: Any,
) -> Persona:
    """Register a persona on the shared :data:`default_registry`.

    Args:
        name: Persona name.
        system_prompt: The system prompt text.
        **kwargs: Forwarded to :meth:`PersonaRegistry.register` (description,
            tags, metadata, model, temperature, and extra request params).

    Returns:
        The newly registered :class:`Persona`.
    """
    default_registry.register(name, system_prompt, **kwargs)
    return default_registry.require(name)


def get(name: str) -> Persona:
    """Return a persona from the shared :data:`default_registry`.

    Args:
        name: Persona name.

    Returns:
        The registered :class:`Persona`.

    Raises:
        PersonaNotFoundError: If *name* is not registered.
    """
    return default_registry.require(name)


def apply(
    name: str,
    messages: list[Message],
    *,
    include_model: bool = True,
) -> dict[str, Any]:
    """Build request kwargs from a persona in the shared registry.

    See :meth:`PersonaRegistry.apply`.
    """
    return default_registry.apply(name, messages, include_model=include_model)


def system_prompt(name: str) -> str:
    """Return the system prompt for a persona in the shared registry."""
    return default_registry.system_prompt(name)
