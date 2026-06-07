# llm-persona

[![CI](https://github.com/MukundaKatta/llm-persona/actions/workflows/ci.yml/badge.svg)](https://github.com/MukundaKatta/llm-persona/actions/workflows/ci.yml)

A tiny, **zero-dependency** named-persona registry for LLM agents.

Define reusable system prompts (plus optional default model, temperature, and
extra request parameters) once, give each one a name, then look them up and turn
them into ready-to-send request payloads. Works with any chat-completion style
client (OpenAI, Anthropic, etc.) because it only produces plain `dict`s — it
never imports a provider SDK.

## Install

```bash
pip install llm-persona
```

Requires Python 3.10+.

## Quick start

```python
from llm_persona import PersonaRegistry

registry = PersonaRegistry()
registry.register(
    "researcher",
    "You are a meticulous research assistant. Cite sources.",
    model="claude-sonnet-4-5",
    temperature=0.2,
)
registry.register(
    "coder",
    "You are an expert Python developer. Write clean, typed code.",
    tags=["engineering"],
)

messages = [{"role": "user", "content": "Summarize the CAP theorem."}]

# Build request kwargs and splat them straight into your client call.
kwargs = registry.apply("researcher", messages)
# kwargs == {
#     "system": "You are a meticulous research assistant. Cite sources.",
#     "messages": [{"role": "user", "content": "Summarize the CAP theorem."}],
#     "model": "claude-sonnet-4-5",
#     "temperature": 0.2,
# }

# e.g. with the Anthropic SDK:
#   response = client.messages.create(max_tokens=1024, **kwargs)
```

### Module-level shortcut

For quick scripts there is a process-wide default registry and matching helpers:

```python
import llm_persona

llm_persona.register("helper", "You are a concise, friendly assistant.")
kwargs = llm_persona.apply("helper", [{"role": "user", "content": "hi"}])
print(llm_persona.system_prompt("helper"))  # "You are a concise, friendly assistant."
```

### Injecting a system message (OpenAI chat style)

If you prefer the system prompt as the first message rather than a top-level
`system` field:

```python
persona = registry.require("coder")
chat = persona.inject_system([{"role": "user", "content": "Refactor this."}])
# chat[0] == {"role": "system", "content": "You are an expert Python developer..."}
# the original list passed in is never mutated
```

## API

### `Persona` (frozen dataclass)

| Field | Type | Description |
| --- | --- | --- |
| `name` | `str` | Unique identifier (required, non-empty). |
| `system_prompt` | `str` | System prompt text (required, non-empty). |
| `description` | `str` | Optional human-readable description. |
| `tags` | `tuple[str, ...]` | Tags (any iterable is normalised to a tuple). |
| `metadata` | `dict[str, Any]` | Arbitrary informational key/values (deep-copied). |
| `model` | `str \| None` | Optional default model id. |
| `temperature` | `float \| None` | Optional default sampling temperature. |
| `extra` | `dict[str, Any]` | Extra request params merged into `apply()` output. |

Methods:

- `apply(messages, *, include_model=True) -> dict` — build request kwargs
  (`system`, `messages`, and any of `model` / `temperature` / `extra` that are
  set). Pass `include_model=False` to omit the model.
- `as_message()` / `as_system_message()` — return `{"role": "system", "content": ...}`.
- `inject_system(messages) -> list` — return a new list with the system message prepended.
- `with_tags(*tags) -> Persona` — copy with extra tags (deduplicated).
- `with_metadata(**extra) -> Persona` — copy with merged metadata.

Personas are immutable; the `with_*` builders return new instances.

### `PersonaRegistry`

Mutations (all return `self` for chaining):

- `register(persona_or_name, system_prompt="", *, description="", tags=(), metadata=None, model=None, temperature=None, **extra)`
  — register a `Persona` or a `(name, system_prompt)` shorthand. Unknown
  keyword arguments are collected into `extra`. Raises `ValueError` on a
  duplicate name.
- `upsert(persona)` — insert or replace unconditionally.
- `update(persona)` — replace an existing persona (raises `PersonaNotFoundError` if absent).
- `unregister(name)` — remove (raises `PersonaNotFoundError` if absent).
- `remove(name)` — remove if present (never raises).

Lookups:

- `get(name, default=None)` / `get_or_none(name)` — return the persona or a default / `None`.
- `require(name)` / `registry[name]` — return the persona or raise `PersonaNotFoundError`.
- `has(name)` / `name in registry` — membership test.
- `system_prompt(name)` — the persona's system prompt (raises if absent).
- `apply(name, messages, *, include_model=True)` — `require(name).apply(...)`.

Collections:

- `names()` — sorted names.
- `by_tag(tag)` — sorted names carrying `tag`.
- `all_tags()` — sorted, deduplicated tags across all personas.
- `all_personas()` / `iter(registry)` — personas sorted by name.
- `count` / `len(registry)` / `is_empty`.

### Module-level helpers

`register`, `get`, `apply`, and `system_prompt` operate on the shared
`llm_persona.default_registry`.

### Exceptions

`PersonaNotFoundError` (a subclass of `KeyError`). `PersonaNotFound` is a
convenience alias for the same class.

## Development

The library has no runtime dependencies and the tests use only the standard
library, so you can run them without installing anything:

```bash
python3 -m unittest discover -s tests
```

## License

MIT — see [LICENSE](LICENSE).
