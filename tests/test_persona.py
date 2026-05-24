"""Tests for llm_persona.PersonaRegistry and Persona."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from llm_persona import Persona, PersonaNotFoundError, PersonaRegistry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def reg() -> PersonaRegistry:
    return PersonaRegistry()


# ---------------------------------------------------------------------------
# register()
# ---------------------------------------------------------------------------


def test_register_creates_persona_with_correct_fields(reg: PersonaRegistry) -> None:
    p = reg.register("alice", role="A helpful assistant", tone="friendly")
    assert p.name == "alice"
    assert p.role == "A helpful assistant"
    assert p.tone == "friendly"


def test_register_returns_persona_instance(reg: PersonaRegistry) -> None:
    p = reg.register("bob", role="A coding helper")
    assert isinstance(p, Persona)


def test_register_default_tone_is_none(reg: PersonaRegistry) -> None:
    p = reg.register("carol", role="Analyst")
    assert p.tone is None


def test_register_default_restrictions_is_empty_list(reg: PersonaRegistry) -> None:
    p = reg.register("dave", role="Writer")
    assert p.restrictions == []


def test_register_default_extra_is_empty_dict(reg: PersonaRegistry) -> None:
    p = reg.register("eve", role="Reviewer")
    assert p.extra == {}


def test_register_stores_restrictions(reg: PersonaRegistry) -> None:
    p = reg.register("frank", role="Guard", restrictions=["no PII", "no secrets"])
    assert p.restrictions == ["no PII", "no secrets"]


def test_register_stores_extra(reg: PersonaRegistry) -> None:
    p = reg.register("grace", role="Analyst", extra={"language": "en", "style": "terse"})
    assert p.extra == {"language": "en", "style": "terse"}


def test_register_overwrites_existing_name(reg: PersonaRegistry) -> None:
    reg.register("heidi", role="First role", tone="warm")
    reg.register("heidi", role="Second role", tone="cool")
    p = reg.get("heidi")
    assert p.role == "Second role"
    assert p.tone == "cool"


# ---------------------------------------------------------------------------
# get()
# ---------------------------------------------------------------------------


def test_get_returns_correct_persona(reg: PersonaRegistry) -> None:
    reg.register("ivan", role="Summarizer")
    p = reg.get("ivan")
    assert p.name == "ivan"
    assert p.role == "Summarizer"


def test_get_raises_persona_not_found_error_for_missing(reg: PersonaRegistry) -> None:
    with pytest.raises(PersonaNotFoundError):
        reg.get("nobody")


# ---------------------------------------------------------------------------
# has()
# ---------------------------------------------------------------------------


def test_has_returns_true_for_registered(reg: PersonaRegistry) -> None:
    reg.register("judy", role="Tester")
    assert reg.has("judy") is True


def test_has_returns_false_for_missing(reg: PersonaRegistry) -> None:
    assert reg.has("ghost") is False


# ---------------------------------------------------------------------------
# names()
# ---------------------------------------------------------------------------


def test_names_returns_sorted_list(reg: PersonaRegistry) -> None:
    reg.register("zara", role="z")
    reg.register("anna", role="a")
    reg.register("mike", role="m")
    assert reg.names() == ["anna", "mike", "zara"]


def test_names_empty_registry(reg: PersonaRegistry) -> None:
    assert reg.names() == []


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------


def test_delete_returns_true_if_existed(reg: PersonaRegistry) -> None:
    reg.register("kim", role="Agent")
    assert reg.delete("kim") is True


def test_delete_returns_false_if_not_existed(reg: PersonaRegistry) -> None:
    assert reg.delete("nobody") is False


def test_delete_removes_persona(reg: PersonaRegistry) -> None:
    reg.register("leo", role="Agent")
    reg.delete("leo")
    assert not reg.has("leo")


# ---------------------------------------------------------------------------
# render()
# ---------------------------------------------------------------------------


def test_render_includes_name_and_role(reg: PersonaRegistry) -> None:
    reg.register("mia", role="A data analyst")
    result = reg.render("mia")
    assert "You are mia." in result
    assert "Role: A data analyst" in result


def test_render_includes_tone_when_set(reg: PersonaRegistry) -> None:
    reg.register("ned", role="Helper", tone="concise")
    result = reg.render("ned")
    assert "Tone: concise" in result


def test_render_skips_tone_when_none(reg: PersonaRegistry) -> None:
    reg.register("ola", role="Helper")
    result = reg.render("ola")
    assert "Tone" not in result


def test_render_includes_restrictions_when_set(reg: PersonaRegistry) -> None:
    reg.register("pat", role="Guard", restrictions=["no PII", "no links"])
    result = reg.render("pat")
    assert "Restrictions:" in result
    assert "- no PII" in result
    assert "- no links" in result


def test_render_skips_restrictions_when_empty(reg: PersonaRegistry) -> None:
    reg.register("quinn", role="Helper")
    result = reg.render("quinn")
    assert "Restrictions" not in result


def test_render_includes_extra_in_sorted_key_order(reg: PersonaRegistry) -> None:
    reg.register("rex", role="Bot", extra={"zzz": "last", "aaa": "first"})
    result = reg.render("rex")
    aaa_pos = result.index("aaa: first")
    zzz_pos = result.index("zzz: last")
    assert aaa_pos < zzz_pos


def test_render_skips_extra_when_empty(reg: PersonaRegistry) -> None:
    reg.register("sara", role="Helper")
    result = reg.render("sara")
    # No extra lines beyond the standard ones
    lines = result.splitlines()
    assert all(not line.startswith("extra") for line in lines)


def test_render_raises_for_missing_persona(reg: PersonaRegistry) -> None:
    with pytest.raises(PersonaNotFoundError):
        reg.render("nobody")


# ---------------------------------------------------------------------------
# inject()
# ---------------------------------------------------------------------------


def test_inject_no_system_msg_prepends_system(reg: PersonaRegistry) -> None:
    reg.register("tia", role="Helper")
    messages = [{"role": "user", "content": "Hello"}]
    result = reg.inject("tia", messages)
    assert result[0]["role"] == "system"
    assert "You are tia." in result[0]["content"]


def test_inject_existing_system_msg_prepends_persona(reg: PersonaRegistry) -> None:
    reg.register("uma", role="Guide")
    messages = [
        {"role": "system", "content": "Existing instructions."},
        {"role": "user", "content": "Hi"},
    ]
    result = reg.inject("uma", messages)
    assert result[0]["role"] == "system"
    assert "You are uma." in result[0]["content"]
    assert "Existing instructions." in result[0]["content"]
    # Persona should come before the original content
    persona_pos = result[0]["content"].index("You are uma.")
    existing_pos = result[0]["content"].index("Existing instructions.")
    assert persona_pos < existing_pos


def test_inject_does_not_mutate_input_list(reg: PersonaRegistry) -> None:
    reg.register("vic", role="Agent")
    messages: list[dict] = [{"role": "user", "content": "Hi"}]
    original_length = len(messages)
    reg.inject("vic", messages)
    assert len(messages) == original_length


def test_inject_does_not_mutate_existing_system_dict(reg: PersonaRegistry) -> None:
    reg.register("wren", role="Guide")
    sys_msg: dict = {"role": "system", "content": "Original."}
    messages = [sys_msg]
    reg.inject("wren", messages)
    # The original dict must be unchanged
    assert sys_msg["content"] == "Original."


def test_inject_preserves_other_messages(reg: PersonaRegistry) -> None:
    reg.register("xena", role="Warrior")
    messages = [
        {"role": "user", "content": "Question 1"},
        {"role": "assistant", "content": "Answer 1"},
    ]
    result = reg.inject("xena", messages)
    assert result[1]["role"] == "user"
    assert result[2]["role"] == "assistant"


# ---------------------------------------------------------------------------
# save() + load()
# ---------------------------------------------------------------------------


def test_save_and_load_round_trip(reg: PersonaRegistry) -> None:
    reg.register("yara", role="Poet", tone="lyrical", restrictions=["rhyme always"])
    reg.register("zack", role="Coder", extra={"lang": "python"})
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        reg.save(path)
        reg2 = PersonaRegistry()
        reg2.load(path)
        assert reg2.has("yara")
        assert reg2.has("zack")
        yara = reg2.get("yara")
        assert yara.tone == "lyrical"
        assert yara.restrictions == ["rhyme always"]
        zack = reg2.get("zack")
        assert zack.extra == {"lang": "python"}
    finally:
        os.unlink(path)


def test_load_merges_without_wiping_existing(reg: PersonaRegistry) -> None:
    reg.register("alpha", role="First")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        # Save a registry with a different persona
        reg2 = PersonaRegistry()
        reg2.register("beta", role="Second")
        reg2.save(path)
        # Load into reg — alpha should still be there
        reg.load(path)
        assert reg.has("alpha")
        assert reg.has("beta")
    finally:
        os.unlink(path)


def test_save_produces_valid_json(reg: PersonaRegistry) -> None:
    reg.register("gamma", role="Tester")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        reg.save(path)
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert data[0]["name"] == "gamma"
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Multiple personas / isolation
# ---------------------------------------------------------------------------


def test_multiple_personas_are_independent(reg: PersonaRegistry) -> None:
    reg.register("p1", role="Role One", tone="warm")
    reg.register("p2", role="Role Two", tone="cold")
    assert reg.get("p1").tone == "warm"
    assert reg.get("p2").tone == "cold"


def test_persona_is_frozen(reg: PersonaRegistry) -> None:
    p = reg.register("frozen", role="Immutable")
    with pytest.raises((AttributeError, TypeError)):
        p.role = "Changed"  # type: ignore[misc]
