"""Tests for the module-level convenience API and exception aliases.

Uses the standard library ``unittest`` only (no third-party deps).
"""

from __future__ import annotations

import os
import sys
import unittest

# Make ``src/`` importable so the tests run with a plain
# ``python3 -m unittest discover -s tests`` without an editable install.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import llm_persona  # noqa: E402
from llm_persona import (  # noqa: E402
    Persona,
    PersonaNotFound,
    PersonaNotFoundError,
    PersonaRegistry,
)


class TestExceptionAlias(unittest.TestCase):
    def test_persona_not_found_alias_is_same_class(self) -> None:
        # ``PersonaNotFound`` is an alias of ``PersonaNotFoundError`` so either
        # name can be used in an ``except`` clause.
        self.assertIs(PersonaNotFound, PersonaNotFoundError)

    def test_alias_catches_real_error(self) -> None:
        with self.assertRaises(PersonaNotFound):
            PersonaRegistry().require("nope")


class TestModuleLevelAPI(unittest.TestCase):
    def setUp(self) -> None:
        # Each test gets a clean shared registry so ordering does not matter.
        llm_persona.default_registry._store.clear()

    def tearDown(self) -> None:
        llm_persona.default_registry._store.clear()

    def test_register_returns_persona(self) -> None:
        p = llm_persona.register("a", "System prompt.")
        self.assertIsInstance(p, Persona)
        self.assertEqual(p.name, "a")
        self.assertEqual(p.system_prompt, "System prompt.")

    def test_register_and_get(self) -> None:
        llm_persona.register("greeter", "Say hi.")
        self.assertEqual(llm_persona.get("greeter").system_prompt, "Say hi.")

    def test_get_missing_raises(self) -> None:
        with self.assertRaises(PersonaNotFoundError):
            llm_persona.get("missing")

    def test_apply(self) -> None:
        llm_persona.register("a", "Apply this.", model="m")
        result = llm_persona.apply("a", [{"role": "user", "content": "hi"}])
        self.assertEqual(result["system"], "Apply this.")
        self.assertEqual(result["model"], "m")

    def test_system_prompt(self) -> None:
        llm_persona.register("a", "SP text.")
        self.assertEqual(llm_persona.system_prompt("a"), "SP text.")

    def test_register_with_extra_params(self) -> None:
        llm_persona.register("a", "X", temperature=0.2, max_tokens=64)
        result = llm_persona.apply("a", [])
        self.assertEqual(result["temperature"], 0.2)
        self.assertEqual(result["max_tokens"], 64)

    def test_default_registry_is_shared(self) -> None:
        llm_persona.register("shared", "X")
        self.assertIn("shared", llm_persona.default_registry)


if __name__ == "__main__":
    unittest.main()
