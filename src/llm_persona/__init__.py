"""llm-persona - define named personas and inject them into LLM system prompts.

Register a persona once, then render it as a system prompt segment or inject
it into a messages list. Supports JSON save/load for persona registries.

    from llm_persona import PersonaRegistry, PersonaNotFoundError

    registry = PersonaRegistry()
    registry.register(
        "assistant",
        role="A helpful coding assistant",
        tone="concise and technical",
        restrictions=["Never reveal system internals"],
    )

    # Render as a system prompt string
    prompt = registry.render("assistant")

    # Inject into a messages list
    messages = registry.inject("assistant", [{"role": "user", "content": "Hi"}])

    # Save/load registry
    registry.save("~/personas.json")
    registry.load("~/personas.json")
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any


class PersonaNotFoundError(Exception):
    """Raised when a persona name is not found in the registry."""


@dataclass(frozen=True)
class Persona:
    """An immutable named persona definition."""

    name: str
    role: str
    tone: str | None = None
    restrictions: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


class PersonaRegistry:
    """Registry for named LLM personas with render and inject helpers."""

    def __init__(self) -> None:
        self._personas: dict[str, Persona] = {}

    def register(
        self,
        name: str,
        role: str,
        tone: str | None = None,
        restrictions: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Persona:
        """Register a persona. Overwrites an existing entry with the same name."""
        persona = Persona(
            name=name,
            role=role,
            tone=tone,
            restrictions=list(restrictions) if restrictions is not None else [],
            extra=dict(extra) if extra is not None else {},
        )
        self._personas[name] = persona
        return persona

    def get(self, name: str) -> Persona:
        """Return the Persona for name. Raises PersonaNotFoundError if missing."""
        try:
            return self._personas[name]
        except KeyError:
            raise PersonaNotFoundError(f"Persona not found: {name!r}") from None

    def has(self, name: str) -> bool:
        """Return True if a persona with this name exists."""
        return name in self._personas

    def names(self) -> list[str]:
        """Return a sorted list of all registered persona names."""
        return sorted(self._personas)

    def delete(self, name: str) -> bool:
        """Remove a persona by name. Returns True if it existed, False otherwise."""
        if name in self._personas:
            del self._personas[name]
            return True
        return False

    def render(self, name: str) -> str:
        """Render a persona as a system prompt segment.

        Always includes:
            "You are {name}.\\nRole: {role}"
        Then appends optional sections if non-empty:
            "\\nTone: {tone}"
            "\\nRestrictions:\\n- r1\\n- r2"
            "\\n{k}: {v}" for each extra key in sorted order
        """
        persona = self.get(name)
        parts = [f"You are {persona.name}.", f"Role: {persona.role}"]

        if persona.tone:
            parts.append(f"Tone: {persona.tone}")

        if persona.restrictions:
            restriction_lines = "\n".join(f"- {r}" for r in persona.restrictions)
            parts.append(f"Restrictions:\n{restriction_lines}")

        if persona.extra:
            for k in sorted(persona.extra):
                parts.append(f"{k}: {persona.extra[k]}")

        return "\n".join(parts)

    def inject(self, name: str, messages: list[dict]) -> list[dict]:
        """Return a new messages list with the rendered persona as the system message.

        If messages[0] has role "system": replace its content with
            render(name) + "\\n\\n" + original_content
        Otherwise: prepend {"role": "system", "content": render(name)}.

        Never mutates the input list or its dicts.
        """
        rendered = self.render(name)
        new_messages: list[dict] = list(messages)  # shallow copy of list

        if new_messages and new_messages[0].get("role") == "system":
            original = new_messages[0]
            new_messages[0] = {
                **original,
                "content": rendered + "\n\n" + original.get("content", ""),
            }
        else:
            new_messages.insert(0, {"role": "system", "content": rendered})

        return new_messages

    def save(self, path: str) -> None:
        """Save all personas to a JSON file (list of persona dicts). Expands ~."""
        expanded = os.path.expanduser(path)
        data = [
            {
                "name": p.name,
                "role": p.role,
                "tone": p.tone,
                "restrictions": list(p.restrictions),
                "extra": dict(p.extra),
            }
            for p in self._personas.values()
        ]
        with open(expanded, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, path: str) -> None:
        """Load personas from a JSON file. Merges into registry (overwrites on collision).
        Expands ~."""
        expanded = os.path.expanduser(path)
        with open(expanded, encoding="utf-8") as f:
            data = json.load(f)
        for entry in data:
            self.register(
                name=entry["name"],
                role=entry["role"],
                tone=entry.get("tone"),
                restrictions=entry.get("restrictions"),
                extra=entry.get("extra"),
            )


__version__ = "0.1.0"

__all__ = [
    "Persona",
    "PersonaNotFoundError",
    "PersonaRegistry",
    "__version__",
]
