"""20 — Prompt registry & versioning (+ provider reasoning knobs).

`PromptRegistry` keeps named prompts under version control so you can roll back a
change that regressed. Combine it with the eval harness to compare versions.

Also shown: the typed reasoning/cost knobs on the Anthropic adapter
(`thinking` / `effort` / `task_budget`) — configured below but not called (no API
key needed for this demo).

Run:
    PYTHONPATH=src python examples/20_prompts.py
"""

from __future__ import annotations

from agentix import PromptRegistry


def main() -> None:
    prompts = PromptRegistry()

    v1 = prompts.register("assistant", "You are a helpful assistant.")
    v2 = prompts.register("assistant", "You are a terse, snarky assistant.")  # regressed
    print(f"registered v{v1} and v{v2}; active = v{prompts.active('assistant')}")
    print("active prompt:", prompts.get("assistant"))

    # The v2 change tanked your eval pass-rate — roll back.
    prompts.rollback("assistant", v1)
    print(f"\nrolled back; active = v{prompts.active('assistant')}")
    print("active prompt:", prompts.get("assistant"))

    # Templating + persistence.
    prompts.register("greet", "Hello {name}, welcome to {product}.")
    print("\nrendered:", prompts.render("greet", name="Sam", product="agentix"))
    blob = prompts.to_dict()                 # persist via a Store / JSON
    restored = PromptRegistry.from_dict(blob)
    print("restored active 'assistant':", restored.get("assistant"))

    # --- reasoning / cost knobs on the Anthropic adapter (config only) ---
    print("\nAnthropic reasoning/cost knobs (typed, no opaque extra):")
    print("  AnthropicModel(thinking='summarized', effort='low')   # cheaper, shows reasoning")
    print("  AnthropicModel(effort='xhigh')                        # max quality for hard tasks")
    print("  AnthropicModel(task_budget=50_000)                    # self-moderated loop budget")


if __name__ == "__main__":
    main()
