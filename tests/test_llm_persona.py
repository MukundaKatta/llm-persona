"""Tests for llm_persona."""

from __future__ import annotations

import pytest

from llm_persona import Persona, PersonaNotFoundError, PersonaRegistry

# ---------------------------------------------------------------------------
# Persona dataclass
# ---------------------------------------------------------------------------


def test_persona_basic():
    p = Persona(name="bot", system_prompt="Be helpful.")
    assert p.name == "bot"
    assert p.system_prompt == "Be helpful."
    assert p.description == ""
    assert p.tags == ()
    assert p.metadata == {}


def test_persona_init_with_tags():
    p = Persona(name="bot", system_prompt="...", tags=["a", "b"])
    assert p.tags == ("a", "b")


def test_persona_tags_from_list():
    p = Persona(name="bot", system_prompt="...", tags=["x"])
    assert isinstance(p.tags, tuple)


def test_persona_frozen():
    p = Persona(name="bot", system_prompt="...")
    with pytest.raises((AttributeError, TypeError)):
        p.name = "other"  # type: ignore[misc]


def test_persona_empty_name_raises():
    with pytest.raises(ValueError):
        Persona(name="", system_prompt="...")


def test_persona_empty_prompt_raises():
    with pytest.raises(ValueError):
        Persona(name="bot", system_prompt="")


def test_persona_repr():
    p = Persona(name="bot", system_prompt="...")
    assert "bot" in repr(p)


def test_persona_as_message():
    p = Persona(name="bot", system_prompt="Be helpful.")
    msg = p.as_message()
    assert msg == {"role": "system", "content": "Be helpful."}


def test_persona_with_metadata():
    p = Persona(name="bot", system_prompt="...", metadata={"t": 0.5})
    p2 = p.with_metadata(model="gpt-4o")
    assert p2.metadata == {"t": 0.5, "model": "gpt-4o"}
    # original unchanged
    assert "model" not in p.metadata


def test_persona_with_tags():
    p = Persona(name="bot", system_prompt="...", tags=["a"])
    p2 = p.with_tags("b", "c")
    assert "a" in p2.tags
    assert "b" in p2.tags
    # original unchanged
    assert "b" not in p.tags


def test_persona_with_tags_dedup():
    p = Persona(name="bot", system_prompt="...", tags=["a"])
    p2 = p.with_tags("a", "b")
    assert p2.tags.count("a") == 1


# ---------------------------------------------------------------------------
# PersonaRegistry — constructor / repr
# ---------------------------------------------------------------------------


def test_registry_repr():
    r = PersonaRegistry()
    assert "count=0" in repr(r)


def test_registry_initial_empty():
    r = PersonaRegistry()
    assert r.count == 0
    assert r.is_empty is True


# ---------------------------------------------------------------------------
# register — Persona object
# ---------------------------------------------------------------------------


def test_register_persona_returns_self():
    r = PersonaRegistry()
    assert r.register(Persona(name="bot", system_prompt="...")) is r


def test_register_persona_adds():
    r = PersonaRegistry()
    r.register(Persona(name="bot", system_prompt="..."))
    assert r.has("bot") is True


def test_register_duplicate_raises():
    r = PersonaRegistry()
    r.register(Persona(name="bot", system_prompt="..."))
    with pytest.raises(ValueError, match="already registered"):
        r.register(Persona(name="bot", system_prompt="..."))


# ---------------------------------------------------------------------------
# register — by name
# ---------------------------------------------------------------------------


def test_register_by_name():
    r = PersonaRegistry()
    r.register("bot", "Be helpful.", tags=["general"])
    p = r.require("bot")
    assert p.name == "bot"
    assert p.system_prompt == "Be helpful."
    assert "general" in p.tags


def test_register_by_name_no_prompt_raises():
    r = PersonaRegistry()
    with pytest.raises(ValueError):
        r.register("bot")


# ---------------------------------------------------------------------------
# unregister
# ---------------------------------------------------------------------------


def test_unregister():
    r = PersonaRegistry()
    r.register(Persona(name="bot", system_prompt="..."))
    r.unregister("bot")
    assert r.has("bot") is False


def test_unregister_missing_raises():
    with pytest.raises(PersonaNotFoundError):
        PersonaRegistry().unregister("nope")


def test_unregister_returns_self():
    r = PersonaRegistry()
    r.register(Persona(name="bot", system_prompt="..."))
    assert r.unregister("bot") is r


# ---------------------------------------------------------------------------
# update / upsert
# ---------------------------------------------------------------------------


def test_update_replaces():
    r = PersonaRegistry()
    r.register(Persona(name="bot", system_prompt="old"))
    r.update(Persona(name="bot", system_prompt="new"))
    assert r.require("bot").system_prompt == "new"


def test_update_missing_raises():
    r = PersonaRegistry()
    with pytest.raises(PersonaNotFoundError):
        r.update(Persona(name="bot", system_prompt="..."))


def test_upsert_creates():
    r = PersonaRegistry()
    r.upsert(Persona(name="bot", system_prompt="..."))
    assert r.has("bot") is True


def test_upsert_replaces():
    r = PersonaRegistry()
    r.register(Persona(name="bot", system_prompt="old"))
    r.upsert(Persona(name="bot", system_prompt="new"))
    assert r.require("bot").system_prompt == "new"


# ---------------------------------------------------------------------------
# get / require / has
# ---------------------------------------------------------------------------


def test_get_present():
    r = PersonaRegistry()
    p = Persona(name="bot", system_prompt="...")
    r.register(p)
    assert r.get("bot") == p


def test_get_missing_default():
    assert PersonaRegistry().get("nope") is None


def test_get_custom_default():
    fallback = Persona(name="fallback", system_prompt="...")
    assert PersonaRegistry().get("nope", fallback) is fallback


def test_require_present():
    r = PersonaRegistry()
    p = Persona(name="bot", system_prompt="...")
    r.register(p)
    assert r.require("bot") == p


def test_require_missing_raises():
    with pytest.raises(PersonaNotFoundError):
        PersonaRegistry().require("nope")


def test_persona_not_found_is_key_error():
    assert issubclass(PersonaNotFoundError, KeyError)


# ---------------------------------------------------------------------------
# names / by_tag / all_tags / all_personas
# ---------------------------------------------------------------------------


def test_names_sorted():
    r = PersonaRegistry()
    r.register(Persona(name="z", system_prompt="..."))
    r.register(Persona(name="a", system_prompt="..."))
    assert r.names() == ["a", "z"]


def test_by_tag():
    r = PersonaRegistry()
    r.register(Persona(name="c", system_prompt="...", tags=["py"]))
    r.register(Persona(name="a", system_prompt="...", tags=["py", "async"]))
    r.register(Persona(name="b", system_prompt="...", tags=["go"]))
    assert r.by_tag("py") == ["a", "c"]


def test_all_tags_sorted():
    r = PersonaRegistry()
    r.register(Persona(name="a", system_prompt="...", tags=["z", "a"]))
    r.register(Persona(name="b", system_prompt="...", tags=["m"]))
    assert r.all_tags() == ["a", "m", "z"]


def test_all_personas_sorted():
    r = PersonaRegistry()
    r.register(Persona(name="z", system_prompt="..."))
    r.register(Persona(name="a", system_prompt="..."))
    names = [p.name for p in r.all_personas()]
    assert names == ["a", "z"]


# ---------------------------------------------------------------------------
# count / is_empty / len / contains
# ---------------------------------------------------------------------------


def test_count():
    r = PersonaRegistry()
    r.register(Persona(name="a", system_prompt="..."))
    assert r.count == 1


def test_len():
    r = PersonaRegistry()
    r.register(Persona(name="a", system_prompt="..."))
    assert len(r) == 1


def test_contains():
    r = PersonaRegistry()
    r.register(Persona(name="a", system_prompt="..."))
    assert "a" in r
    assert "b" not in r
