"""
llm_persona — named persona registry for LLM agent loops.

Define reusable personas with system prompts, optional model/temperature
settings, and inject them into API calls. Zero dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class PersonaNotFound(KeyError):
    """Raised when a persona name is not registered."""


# ---------------------------------------------------------------------------
# Persona
# ---------------------------------------------------------------------------

@dataclass
class Persona:
    """
    A named LLM persona.

    Attributes:
        name: Unique persona identifier.
        system_prompt: The system prompt text.
        description: Optional human-readable description.
        model: Optional preferred model (informational; used by :meth:`as_kwargs`).
        temperature: Optional preferred temperature.
        extra: Additional kwargs to include in :meth:`as_kwargs`.
    """

    name: str
    system_prompt: str
    description: str = ""
    model: str | None = None
    temperature: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def apply(
        self,
        messages: list[dict[str, Any]],
        *,
        include_model: bool = True,
        include_temperature: bool = True,
    ) -> dict[str, Any]:
        """
        Build a request dict with this persona's system prompt injected.

        Returns a dict with ``"system"`` and ``"messages"`` keys, plus
        optional ``"model"`` and ``"temperature"`` if set.

        Args:
            messages: List of message dicts.
            include_model: Include ``model`` key if set on this persona.
            include_temperature: Include ``temperature`` key if set.

        Returns:
            Dict ready to pass to ``client.messages.create(**result)``.
        """
        result: dict[str, Any] = {
            "system": self.system_prompt,
            "messages": messages,
        }
        if include_model and self.model is not None:
            result["model"] = self.model
        if include_temperature and self.temperature is not None:
            result["temperature"] = self.temperature
        result.update(self.extra)
        return result

    def as_system_message(self) -> dict[str, str]:
        """Return as an OpenAI-style system message dict."""
        return {"role": "system", "content": self.system_prompt}

    def inject_system(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Prepend a system message to a message list (OpenAI style).

        Returns a new list; does not mutate the original.
        """
        return [self.as_system_message()] + list(messages)

    def __str__(self) -> str:
        return f"Persona({self.name!r})"

    def __repr__(self) -> str:
        desc = f", description={self.description!r}" if self.description else ""
        model = f", model={self.model!r}" if self.model else ""
        return f"Persona(name={self.name!r}{desc}{model})"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class PersonaRegistry:
    """
    Registry of named personas.

    Usage::

        registry = PersonaRegistry()

        registry.register(
            "researcher",
            "You are a meticulous research assistant. Cite sources.",
            description="Research persona",
        )

        kwargs = registry.apply("researcher", messages)
        response = client.messages.create(**kwargs)
    """

    def __init__(self) -> None:
        self._personas: dict[str, Persona] = {}

    def register(
        self,
        name: str,
        system_prompt: str,
        *,
        description: str = "",
        model: str | None = None,
        temperature: float | None = None,
        **extra: Any,
    ) -> Persona:
        """
        Register a persona.

        Args:
            name: Unique persona name.
            system_prompt: System prompt text.
            description: Optional description.
            model: Optional preferred model.
            temperature: Optional preferred temperature.
            **extra: Additional kwargs included in :meth:`Persona.apply`.

        Returns:
            The registered :class:`Persona`.
        """
        persona = Persona(
            name=name,
            system_prompt=system_prompt,
            description=description,
            model=model,
            temperature=temperature,
            extra=extra,
        )
        self._personas[name] = persona
        return persona

    def get(self, name: str) -> Persona:
        """
        Return a persona by name.

        Raises:
            :class:`PersonaNotFound` if not registered.
        """
        persona = self._personas.get(name)
        if persona is None:
            raise PersonaNotFound(f"No persona named {name!r}")
        return persona

    def get_or_none(self, name: str) -> Persona | None:
        """Return persona by name, or None if not found."""
        return self._personas.get(name)

    def apply(
        self,
        name: str,
        messages: list[dict[str, Any]],
        *,
        include_model: bool = True,
        include_temperature: bool = True,
    ) -> dict[str, Any]:
        """
        Apply a persona to a message list and return request kwargs.

        Args:
            name: Persona name.
            messages: Message list.
            include_model: Include model if set.
            include_temperature: Include temperature if set.

        Returns:
            Dict ready for ``client.messages.create(**result)``.
        """
        return self.get(name).apply(
            messages,
            include_model=include_model,
            include_temperature=include_temperature,
        )

    def system_prompt(self, name: str) -> str:
        """Return just the system prompt for a persona."""
        return self.get(name).system_prompt

    def remove(self, name: str) -> None:
        """Remove a persona by name."""
        self._personas.pop(name, None)

    def names(self) -> list[str]:
        """Return all registered persona names."""
        return list(self._personas.keys())

    def all_personas(self) -> list[Persona]:
        """Return all registered personas."""
        return list(self._personas.values())

    def __contains__(self, name: str) -> bool:
        return name in self._personas

    def __len__(self) -> int:
        return len(self._personas)

    def __getitem__(self, name: str) -> Persona:
        return self.get(name)


# ---------------------------------------------------------------------------
# Module-level default registry
# ---------------------------------------------------------------------------

_default = PersonaRegistry()


def register(
    name: str,
    system_prompt: str,
    *,
    description: str = "",
    model: str | None = None,
    temperature: float | None = None,
    **extra: Any,
) -> Persona:
    """Register a persona on the default registry."""
    return _default.register(
        name, system_prompt,
        description=description,
        model=model,
        temperature=temperature,
        **extra,
    )


def get(name: str) -> Persona:
    """Get a persona from the default registry."""
    return _default.get(name)


def apply(
    name: str,
    messages: list[dict[str, Any]],
    *,
    include_model: bool = True,
    include_temperature: bool = True,
) -> dict[str, Any]:
    """Apply a persona from the default registry."""
    return _default.apply(name, messages, include_model=include_model, include_temperature=include_temperature)


def system_prompt(name: str) -> str:
    """Get system prompt from the default registry."""
    return _default.system_prompt(name)
