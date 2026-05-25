"""llm_persona — named persona registry for LLM agents."""

from .core import Persona, PersonaNotFoundError, PersonaRegistry

__all__ = [
    "Persona",
    "PersonaNotFoundError",
    "PersonaRegistry",
]
