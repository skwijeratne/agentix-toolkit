# Prompt versioning

Your **prompt** — the instructions you give the model — is one of the most important parts of an agent. Small wording changes can make quality better *or* worse. The `PromptRegistry` helps you manage prompts like you manage code: keep versions, and roll back instantly if a change makes things worse.

## Keeping versions

Register a prompt by name. Each time you register new text, it becomes a new version and the active one:

```
from agentix import PromptRegistry

prompts = PromptRegistry()
prompts.register("assistant", "You are a helpful assistant.")          # version 1
prompts.register("assistant", "You are a helpful, concise assistant.") # version 2 (now active)

prompts.get("assistant")              # the active text
agent = Agent(model=m, system_prompt=prompts.get("assistant"), tools=[...])
```

## Rolling back

Shipped a prompt change that made things worse? Roll back to a known-good version in one line — no need to remember the old wording:

```
prompts.rollback("assistant", 1)      # go back to version 1
```

## Filling in blanks

`render` fills placeholders so you can reuse a template:

```
prompts.register("greeting", "Hello, {name}. How can I help?")
prompts.render("greeting", name="Sanjaya")     # "Hello, Sanjaya. How can I help?"
```

You can also save the whole registry to a file and load it back, so your prompt history travels with your project.

→ Runnable example: [`examples/20_prompts.py`](https://github.com/skwijeratne/agentix-toolkit/blob/main/examples/20_prompts.py)
