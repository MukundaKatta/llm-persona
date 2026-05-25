# llm-persona

Named persona registry for LLM agents. Zero dependencies.

```python
from llm_persona import PersonaRegistry

registry = PersonaRegistry()
registry.register("researcher", "You are a meticulous research assistant. Cite sources.")
registry.register("coder", "You are an expert Python developer. Write clean, typed code.")

# Apply to a request
kwargs = registry.apply("researcher", messages)
response = client.messages.create(**kwargs)
```

## Install

```bash
pip install llm-persona
```
