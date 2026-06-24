# Models & providers

A **provider** is the company or program that runs the AI model — Anthropic
(Claude), OpenAI (GPT), Google (Gemini), and so on. agentix talks to all of them
through small **adapters**, so the rest of your code never changes. Switching
providers is a one-line edit.

## Pick a model

Each adapter lives in `agentix.providers` and needs its matching extra installed.

```python
from agentix.providers.anthropic import AnthropicModel
model = AnthropicModel(model="claude-opus-4-8")     # reads ANTHROPIC_API_KEY

from agentix.providers.openai import OpenAIModel
model = OpenAIModel(model="gpt-4o")                 # reads OPENAI_API_KEY
```

Whatever you choose, the agent is identical:

```python
agent = Agent(model=model, system_prompt="...", tools=[...])
```

## What's available

| Adapter | Install | Notes |
|---|---|---|
| `AnthropicModel` | `agentix-toolkit[anthropic]` | Claude models |
| `OpenAIModel` | `agentix-toolkit[openai]` | GPT models; also works with any "OpenAI-compatible" server |
| `GeminiModel` | `agentix-toolkit[gemini]` | Google Gemini |
| `BedrockModel` | `agentix-toolkit[bedrock]` | Models hosted on AWS Bedrock |
| `OllamaModel` | `agentix-toolkit[ollama]` | Models running **locally** on your machine |
| `LiteLLMModel` | `agentix-toolkit[litellm]` | One bridge to 100+ providers |

→ Runnable gallery:
[`examples/21_providers.py`](https://github.com/skwijeratne/agentix-toolkit/blob/main/examples/21_providers.py)

## Running a model locally

Want to run a model on your own computer with no API key and no cloud? Use
[Ollama](https://ollama.com): install it, start it (`ollama serve`), pull a
model, and point `OllamaModel` at it.

```python
from agentix.providers.ollama import OllamaModel
model = OllamaModel(model="llama3.1")     # runs on your machine, free
```

## Testing without a real model

For tests and learning, `MockModel` returns answers you write in advance — no
network, no key, no cost. See **[Getting started](../getting-started.md)**. To
record real responses once and replay them in tests, see **[Reliability →
cassettes](reliability.md#record-and-replay)**.
