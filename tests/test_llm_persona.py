"""Tests for the core llm_persona API (stdlib unittest only)."""

from __future__ import annotations

import os
import sys
import unittest

# Make ``src/`` importable so the tests run with a plain
# ``python3 -m unittest discover -s tests`` without an editable install.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from llm_persona import Persona, PersonaNotFoundError, PersonaRegistry  # noqa: E402


# ---------------------------------------------------------------------------
# Persona dataclass
# ---------------------------------------------------------------------------


class TestPersona(unittest.TestCase):
    def test_persona_basic(self) -> None:
        p = Persona(name="bot", system_prompt="Be helpful.")
        self.assertEqual(p.name, "bot")
        self.assertEqual(p.system_prompt, "Be helpful.")
        self.assertEqual(p.description, "")
        self.assertEqual(p.tags, ())
        self.assertEqual(p.metadata, {})
        self.assertIsNone(p.model)
        self.assertIsNone(p.temperature)
        self.assertEqual(p.extra, {})

    def test_persona_init_with_tags(self) -> None:
        p = Persona(name="bot", system_prompt="...", tags=["a", "b"])
        self.assertEqual(p.tags, ("a", "b"))

    def test_persona_tags_from_list_is_tuple(self) -> None:
        p = Persona(name="bot", system_prompt="...", tags=["x"])
        self.assertIsInstance(p.tags, tuple)

    def test_persona_frozen(self) -> None:
        p = Persona(name="bot", system_prompt="...")
        with self.assertRaises((AttributeError, TypeError)):
            p.name = "other"  # type: ignore[misc]

    def test_persona_empty_name_raises(self) -> None:
        with self.assertRaises(ValueError):
            Persona(name="", system_prompt="...")

    def test_persona_empty_prompt_raises(self) -> None:
        with self.assertRaises(ValueError):
            Persona(name="bot", system_prompt="")

    def test_persona_repr(self) -> None:
        p = Persona(name="bot", system_prompt="...")
        self.assertIn("bot", repr(p))

    def test_persona_str(self) -> None:
        p = Persona(name="bot", system_prompt="...")
        self.assertIn("bot", str(p))

    def test_persona_as_message(self) -> None:
        p = Persona(name="bot", system_prompt="Be helpful.")
        self.assertEqual(
            p.as_message(), {"role": "system", "content": "Be helpful."}
        )

    def test_persona_as_system_message_alias(self) -> None:
        p = Persona(name="bot", system_prompt="Be helpful.")
        self.assertEqual(p.as_system_message(), p.as_message())

    def test_persona_with_metadata(self) -> None:
        p = Persona(name="bot", system_prompt="...", metadata={"t": 0.5})
        p2 = p.with_metadata(model="gpt-4o")
        self.assertEqual(p2.metadata, {"t": 0.5, "model": "gpt-4o"})
        self.assertNotIn("model", p.metadata)  # original unchanged

    def test_persona_metadata_is_deep_copied(self) -> None:
        source = {"nested": {"k": 1}}
        p = Persona(name="bot", system_prompt="...", metadata=source)
        source["nested"]["k"] = 999
        self.assertEqual(p.metadata["nested"]["k"], 1)

    def test_persona_with_tags(self) -> None:
        p = Persona(name="bot", system_prompt="...", tags=["a"])
        p2 = p.with_tags("b", "c")
        self.assertIn("a", p2.tags)
        self.assertIn("b", p2.tags)
        self.assertNotIn("b", p.tags)  # original unchanged

    def test_persona_with_tags_dedup(self) -> None:
        p = Persona(name="bot", system_prompt="...", tags=["a"])
        p2 = p.with_tags("a", "b")
        self.assertEqual(p2.tags.count("a"), 1)

    def test_persona_with_tags_preserves_params(self) -> None:
        p = Persona(
            name="bot",
            system_prompt="...",
            model="m",
            temperature=0.3,
            extra={"max_tokens": 8},
        )
        p2 = p.with_tags("x")
        self.assertEqual(p2.model, "m")
        self.assertEqual(p2.temperature, 0.3)
        self.assertEqual(p2.extra, {"max_tokens": 8})


# ---------------------------------------------------------------------------
# Persona.apply / inject_system
# ---------------------------------------------------------------------------


class TestPersonaApply(unittest.TestCase):
    def test_apply_includes_system_and_messages(self) -> None:
        p = Persona(name="r", system_prompt="Be helpful.")
        messages = [{"role": "user", "content": "hi"}]
        result = p.apply(messages)
        self.assertEqual(result["system"], "Be helpful.")
        self.assertEqual(result["messages"], messages)

    def test_apply_includes_model(self) -> None:
        p = Persona(name="r", system_prompt="X", model="claude-sonnet-4-5")
        self.assertEqual(p.apply([])["model"], "claude-sonnet-4-5")

    def test_apply_no_model_skipped(self) -> None:
        p = Persona(name="r", system_prompt="X")
        self.assertNotIn("model", p.apply([]))

    def test_apply_include_model_false(self) -> None:
        p = Persona(name="r", system_prompt="X", model="m")
        self.assertNotIn("model", p.apply([], include_model=False))

    def test_apply_includes_temperature(self) -> None:
        p = Persona(name="r", system_prompt="X", temperature=0.7)
        self.assertEqual(p.apply([])["temperature"], 0.7)

    def test_apply_temperature_zero_included(self) -> None:
        # 0.0 is falsy but a valid temperature: it must still be included.
        p = Persona(name="r", system_prompt="X", temperature=0.0)
        self.assertEqual(p.apply([])["temperature"], 0.0)

    def test_apply_no_temperature_skipped(self) -> None:
        p = Persona(name="r", system_prompt="X")
        self.assertNotIn("temperature", p.apply([]))

    def test_apply_extra_kwargs(self) -> None:
        p = Persona(name="r", system_prompt="X", extra={"max_tokens": 1024})
        self.assertEqual(p.apply([])["max_tokens"], 1024)

    def test_apply_does_not_mutate_extra(self) -> None:
        p = Persona(name="r", system_prompt="X", extra={"max_tokens": 1024})
        result = p.apply([])
        result["max_tokens"] = 2
        self.assertEqual(p.extra["max_tokens"], 1024)

    def test_inject_system(self) -> None:
        p = Persona(name="r", system_prompt="Be helpful.")
        messages = [{"role": "user", "content": "hi"}]
        result = p.inject_system(messages)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["role"], "system")
        self.assertEqual(result[1]["role"], "user")

    def test_inject_system_no_mutate(self) -> None:
        p = Persona(name="r", system_prompt="X")
        messages = [{"role": "user", "content": "hi"}]
        p.inject_system(messages)
        self.assertEqual(len(messages), 1)


# ---------------------------------------------------------------------------
# PersonaRegistry — constructor / repr
# ---------------------------------------------------------------------------


class TestRegistryBasics(unittest.TestCase):
    def test_registry_repr(self) -> None:
        self.assertIn("count=0", repr(PersonaRegistry()))

    def test_registry_initial_empty(self) -> None:
        r = PersonaRegistry()
        self.assertEqual(r.count, 0)
        self.assertTrue(r.is_empty)


# ---------------------------------------------------------------------------
# register
# ---------------------------------------------------------------------------


class TestRegister(unittest.TestCase):
    def test_register_persona_returns_self(self) -> None:
        r = PersonaRegistry()
        self.assertIs(r.register(Persona(name="bot", system_prompt="...")), r)

    def test_register_persona_adds(self) -> None:
        r = PersonaRegistry()
        r.register(Persona(name="bot", system_prompt="..."))
        self.assertTrue(r.has("bot"))

    def test_register_duplicate_raises(self) -> None:
        r = PersonaRegistry()
        r.register(Persona(name="bot", system_prompt="..."))
        with self.assertRaisesRegex(ValueError, "already registered"):
            r.register(Persona(name="bot", system_prompt="..."))

    def test_register_by_name(self) -> None:
        r = PersonaRegistry()
        r.register("bot", "Be helpful.", tags=["general"])
        p = r.require("bot")
        self.assertEqual(p.name, "bot")
        self.assertEqual(p.system_prompt, "Be helpful.")
        self.assertIn("general", p.tags)

    def test_register_by_name_no_prompt_raises(self) -> None:
        with self.assertRaises(ValueError):
            PersonaRegistry().register("bot")

    def test_register_by_name_with_params(self) -> None:
        r = PersonaRegistry()
        r.register(
            "p", "X", description="d", model="m", temperature=0.5, max_tokens=512
        )
        p = r.require("p")
        self.assertEqual(p.description, "d")
        self.assertEqual(p.model, "m")
        self.assertEqual(p.temperature, 0.5)
        self.assertEqual(p.extra, {"max_tokens": 512})


# ---------------------------------------------------------------------------
# unregister / remove / update / upsert
# ---------------------------------------------------------------------------


class TestMutations(unittest.TestCase):
    def test_unregister(self) -> None:
        r = PersonaRegistry()
        r.register(Persona(name="bot", system_prompt="..."))
        r.unregister("bot")
        self.assertFalse(r.has("bot"))

    def test_unregister_missing_raises(self) -> None:
        with self.assertRaises(PersonaNotFoundError):
            PersonaRegistry().unregister("nope")

    def test_unregister_returns_self(self) -> None:
        r = PersonaRegistry()
        r.register(Persona(name="bot", system_prompt="..."))
        self.assertIs(r.unregister("bot"), r)

    def test_remove_present(self) -> None:
        r = PersonaRegistry()
        r.register("p", "X")
        r.remove("p")
        self.assertNotIn("p", r)

    def test_remove_missing_no_raise(self) -> None:
        # Must not raise.
        self.assertIs(PersonaRegistry().remove("nope").__class__, PersonaRegistry)

    def test_update_replaces(self) -> None:
        r = PersonaRegistry()
        r.register(Persona(name="bot", system_prompt="old"))
        r.update(Persona(name="bot", system_prompt="new"))
        self.assertEqual(r.require("bot").system_prompt, "new")

    def test_update_missing_raises(self) -> None:
        with self.assertRaises(PersonaNotFoundError):
            PersonaRegistry().update(Persona(name="bot", system_prompt="..."))

    def test_upsert_creates(self) -> None:
        r = PersonaRegistry()
        r.upsert(Persona(name="bot", system_prompt="..."))
        self.assertTrue(r.has("bot"))

    def test_upsert_replaces(self) -> None:
        r = PersonaRegistry()
        r.register(Persona(name="bot", system_prompt="old"))
        r.upsert(Persona(name="bot", system_prompt="new"))
        self.assertEqual(r.require("bot").system_prompt, "new")


# ---------------------------------------------------------------------------
# get / get_or_none / require / has / __getitem__
# ---------------------------------------------------------------------------


class TestLookup(unittest.TestCase):
    def test_get_present(self) -> None:
        r = PersonaRegistry()
        p = Persona(name="bot", system_prompt="...")
        r.register(p)
        self.assertEqual(r.get("bot"), p)

    def test_get_missing_default(self) -> None:
        self.assertIsNone(PersonaRegistry().get("nope"))

    def test_get_custom_default(self) -> None:
        fallback = Persona(name="fallback", system_prompt="...")
        self.assertIs(PersonaRegistry().get("nope", fallback), fallback)

    def test_get_or_none_missing(self) -> None:
        self.assertIsNone(PersonaRegistry().get_or_none("nope"))

    def test_require_present(self) -> None:
        r = PersonaRegistry()
        p = Persona(name="bot", system_prompt="...")
        r.register(p)
        self.assertEqual(r.require("bot"), p)

    def test_require_missing_raises(self) -> None:
        with self.assertRaises(PersonaNotFoundError):
            PersonaRegistry().require("nope")

    def test_getitem_present(self) -> None:
        r = PersonaRegistry()
        r.register("p", "X")
        self.assertEqual(r["p"].name, "p")

    def test_getitem_missing_raises(self) -> None:
        with self.assertRaises(PersonaNotFoundError):
            _ = PersonaRegistry()["missing"]

    def test_persona_not_found_is_key_error(self) -> None:
        self.assertTrue(issubclass(PersonaNotFoundError, KeyError))


# ---------------------------------------------------------------------------
# names / by_tag / all_tags / all_personas / system_prompt / apply
# ---------------------------------------------------------------------------


class TestQueries(unittest.TestCase):
    def test_names_sorted(self) -> None:
        r = PersonaRegistry()
        r.register(Persona(name="z", system_prompt="..."))
        r.register(Persona(name="a", system_prompt="..."))
        self.assertEqual(r.names(), ["a", "z"])

    def test_by_tag(self) -> None:
        r = PersonaRegistry()
        r.register(Persona(name="c", system_prompt="...", tags=["py"]))
        r.register(Persona(name="a", system_prompt="...", tags=["py", "async"]))
        r.register(Persona(name="b", system_prompt="...", tags=["go"]))
        self.assertEqual(r.by_tag("py"), ["a", "c"])

    def test_all_tags_sorted(self) -> None:
        r = PersonaRegistry()
        r.register(Persona(name="a", system_prompt="...", tags=["z", "a"]))
        r.register(Persona(name="b", system_prompt="...", tags=["m"]))
        self.assertEqual(r.all_tags(), ["a", "m", "z"])

    def test_all_personas_sorted(self) -> None:
        r = PersonaRegistry()
        r.register(Persona(name="z", system_prompt="..."))
        r.register(Persona(name="a", system_prompt="..."))
        self.assertEqual([p.name for p in r.all_personas()], ["a", "z"])

    def test_system_prompt(self) -> None:
        r = PersonaRegistry()
        r.register("p", "Be thorough.")
        self.assertEqual(r.system_prompt("p"), "Be thorough.")

    def test_system_prompt_missing_raises(self) -> None:
        with self.assertRaises(PersonaNotFoundError):
            PersonaRegistry().system_prompt("nope")

    def test_registry_apply(self) -> None:
        r = PersonaRegistry()
        r.register("p", "Be helpful.", model="m", max_tokens=512)
        messages = [{"role": "user", "content": "hi"}]
        result = r.apply("p", messages)
        self.assertEqual(result["system"], "Be helpful.")
        self.assertEqual(result["messages"], messages)
        self.assertEqual(result["model"], "m")
        self.assertEqual(result["max_tokens"], 512)

    def test_registry_apply_missing_raises(self) -> None:
        with self.assertRaises(PersonaNotFoundError):
            PersonaRegistry().apply("nope", [])


# ---------------------------------------------------------------------------
# count / is_empty / len / contains / iter
# ---------------------------------------------------------------------------


class TestDunders(unittest.TestCase):
    def test_count(self) -> None:
        r = PersonaRegistry()
        r.register(Persona(name="a", system_prompt="..."))
        self.assertEqual(r.count, 1)

    def test_len(self) -> None:
        r = PersonaRegistry()
        r.register(Persona(name="a", system_prompt="..."))
        self.assertEqual(len(r), 1)

    def test_contains(self) -> None:
        r = PersonaRegistry()
        r.register(Persona(name="a", system_prompt="..."))
        self.assertIn("a", r)
        self.assertNotIn("b", r)

    def test_iter(self) -> None:
        r = PersonaRegistry()
        r.register(Persona(name="z", system_prompt="..."))
        r.register(Persona(name="a", system_prompt="..."))
        self.assertEqual([p.name for p in r], ["a", "z"])


if __name__ == "__main__":
    unittest.main()
