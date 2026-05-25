import pytest
from llm_persona import Persona, PersonaRegistry, PersonaNotFound


# ---------------------------------------------------------------------------
# Persona dataclass
# ---------------------------------------------------------------------------

def test_persona_str():
    p = Persona(name="researcher", system_prompt="Be thorough.")
    assert "researcher" in str(p)

def test_persona_repr():
    p = Persona(name="researcher", system_prompt="Be thorough.")
    assert "researcher" in repr(p)

def test_persona_apply_includes_system():
    p = Persona(name="r", system_prompt="Be helpful.")
    messages = [{"role": "user", "content": "hi"}]
    result = p.apply(messages)
    assert result["system"] == "Be helpful."
    assert result["messages"] == messages

def test_persona_apply_includes_model():
    p = Persona(name="r", system_prompt="X", model="claude-sonnet-4-5")
    result = p.apply([])
    assert result["model"] == "claude-sonnet-4-5"

def test_persona_apply_no_model_skipped():
    p = Persona(name="r", system_prompt="X")
    result = p.apply([])
    assert "model" not in result

def test_persona_apply_include_model_false():
    p = Persona(name="r", system_prompt="X", model="claude-sonnet-4-5")
    result = p.apply([], include_model=False)
    assert "model" not in result

def test_persona_apply_includes_temperature():
    p = Persona(name="r", system_prompt="X", temperature=0.7)
    result = p.apply([])
    assert result["temperature"] == 0.7

def test_persona_apply_no_temperature_skipped():
    p = Persona(name="r", system_prompt="X")
    result = p.apply([])
    assert "temperature" not in result

def test_persona_apply_extra_kwargs():
    p = Persona(name="r", system_prompt="X", extra={"max_tokens": 1024})
    result = p.apply([])
    assert result["max_tokens"] == 1024

def test_persona_as_system_message():
    p = Persona(name="r", system_prompt="Be helpful.")
    msg = p.as_system_message()
    assert msg == {"role": "system", "content": "Be helpful."}

def test_persona_inject_system():
    p = Persona(name="r", system_prompt="Be helpful.")
    messages = [{"role": "user", "content": "hi"}]
    result = p.inject_system(messages)
    assert len(result) == 2
    assert result[0]["role"] == "system"
    assert result[1]["role"] == "user"

def test_persona_inject_system_no_mutate():
    p = Persona(name="r", system_prompt="X")
    messages = [{"role": "user", "content": "hi"}]
    p.inject_system(messages)
    assert len(messages) == 1


# ---------------------------------------------------------------------------
# PersonaRegistry
# ---------------------------------------------------------------------------

def test_register_and_get():
    r = PersonaRegistry()
    r.register("researcher", "Be thorough.")
    p = r.get("researcher")
    assert p.name == "researcher"
    assert p.system_prompt == "Be thorough."

def test_get_missing_raises():
    r = PersonaRegistry()
    with pytest.raises(PersonaNotFound):
        r.get("missing")

def test_get_or_none_missing():
    r = PersonaRegistry()
    assert r.get_or_none("missing") is None

def test_register_description():
    r = PersonaRegistry()
    r.register("p", "X", description="A persona")
    assert r.get("p").description == "A persona"

def test_register_model():
    r = PersonaRegistry()
    r.register("p", "X", model="claude-sonnet-4-5")
    assert r.get("p").model == "claude-sonnet-4-5"

def test_register_temperature():
    r = PersonaRegistry()
    r.register("p", "X", temperature=0.5)
    assert r.get("p").temperature == 0.5

def test_register_extra_kwargs():
    r = PersonaRegistry()
    r.register("p", "X", max_tokens=512)
    result = r.apply("p", [])
    assert result["max_tokens"] == 512

def test_apply():
    r = PersonaRegistry()
    r.register("p", "Be helpful.", model="m")
    messages = [{"role": "user", "content": "hi"}]
    result = r.apply("p", messages)
    assert result["system"] == "Be helpful."
    assert result["messages"] == messages
    assert result["model"] == "m"

def test_system_prompt():
    r = PersonaRegistry()
    r.register("p", "Be thorough.")
    assert r.system_prompt("p") == "Be thorough."

def test_remove():
    r = PersonaRegistry()
    r.register("p", "X")
    r.remove("p")
    assert "p" not in r

def test_remove_missing_no_raise():
    r = PersonaRegistry()
    r.remove("nonexistent")

def test_names():
    r = PersonaRegistry()
    r.register("a", "X")
    r.register("b", "Y")
    assert set(r.names()) == {"a", "b"}

def test_all_personas():
    r = PersonaRegistry()
    r.register("a", "X")
    r.register("b", "Y")
    assert len(r.all_personas()) == 2

def test_contains():
    r = PersonaRegistry()
    r.register("p", "X")
    assert "p" in r
    assert "missing" not in r

def test_len():
    r = PersonaRegistry()
    r.register("a", "X")
    r.register("b", "Y")
    assert len(r) == 2

def test_getitem():
    r = PersonaRegistry()
    r.register("p", "X")
    assert r["p"].name == "p"

def test_getitem_missing_raises():
    r = PersonaRegistry()
    with pytest.raises(PersonaNotFound):
        _ = r["missing"]


# ---------------------------------------------------------------------------
# Module-level API
# ---------------------------------------------------------------------------

def test_module_register_and_get():
    from llm_persona import register, get
    register("test_mod_persona", "Test system prompt.")
    p = get("test_mod_persona")
    assert p.system_prompt == "Test system prompt."

def test_module_apply():
    from llm_persona import register, apply
    register("test_mod_apply", "Apply this.", model="m")
    result = apply("test_mod_apply", [{"role": "user", "content": "hi"}])
    assert result["system"] == "Apply this."

def test_module_system_prompt():
    from llm_persona import register, system_prompt
    register("test_mod_sp", "SP text.")
    assert system_prompt("test_mod_sp") == "SP text."
