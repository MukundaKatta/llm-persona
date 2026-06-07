"""llm_persona — named persona registry for LLM agents."""

from .core import (
    Persona,
    PersonaNotFound,
    PersonaNotFoundError,
    PersonaRegistry,
    apply,
    default_registry,
    get,
    register,
    system_prompt,
)

__all__ = [
    "Persona",
    "PersonaNotFound",
    "PersonaNotFoundError",
    "PersonaRegistry",
    "apply",
    "default_registry",
    "get",
    "register",
    "system_prompt",
]
