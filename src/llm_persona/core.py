"""llm_persona — named persona registry for LLM agents.

Define reusable :class:`Persona` objects with system prompts, tags, and
metadata, then look them up by name via :class:`PersonaRegistry`.
Zero dependencies.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PersonaNotFoundError(KeyError):
    """Raised when a persona name is not in the registry."""


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
        metadata: Arbitrary extra key/value pairs.

    Raises:
        ValueError: If *name* or *system_prompt* is empty.
    """

    name: str
    system_prompt: str
    description: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty")
        if not self.system_prompt:
            raise ValueError("system_prompt must not be empty")
        # Normalise tags to tuple regardless of input type
        object.__setattr__(self, "tags", tuple(self.tags))
        # Deep-copy metadata so callers can't mutate our internals
        object.__setattr__(self, "metadata", copy.deepcopy(self.metadata))

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def as_message(self) -> dict[str, str]:
        """Return as an OpenAI-style system message dict.

        Returns:
            ``{"role": "system", "content": <system_prompt>}``
        """
        return {"role": "system", "content": self.system_prompt}

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
            metadata=copy.deepcopy(self.metadata),
        )

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Persona(name={self.name!r}, "
            f"system_prompt={self.system_prompt!r})"
        )


# ---------------------------------------------------------------------------
# PersonaRegistry
# ---------------------------------------------------------------------------


class PersonaRegistry:
    """A mutable registry of named :class:`Persona` objects.

    Usage::

        registry = PersonaRegistry()
        registry.register("helper", "You are a helpful assistant.")
        p = registry.require("helper")
        kwargs = p.as_message()
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
    ) -> "PersonaRegistry":
        """Register a persona.

        Accepts either a :class:`Persona` object or a ``(name, system_prompt)``
        shorthand.

        Args:
            persona_or_name: A :class:`Persona` instance, or the persona name
                             string when using the shorthand form.
            system_prompt: Required when *persona_or_name* is a string.
            description: Optional description (shorthand form only).
            tags: Optional tags (shorthand form only).
            metadata: Optional metadata dict (shorthand form only).

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

    def __repr__(self) -> str:
        return f"PersonaRegistry(count={self.count})"
