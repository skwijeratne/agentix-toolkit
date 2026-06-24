# Reliability

Real services have hiccups: a request times out, a provider has a blip, you hit a rate limit, or the model returns something malformed. These tools keep your agent working through all of that.

## Retry on temporary errors

Wrap your model in `RetryModel` and it automatically retries when a call fails:

```
from agentix import RetryModel

model = RetryModel(my_model, retries=3)
```

It's **rate-limit aware**. When a provider says "you're going too fast, wait 5 seconds" (a *rate limit*), `RetryModel` waits exactly that long instead of guessing — and falls back to gradually increasing waits for other kinds of errors.

→ Runnable example: [`examples/28_rate_limit.py`](https://github.com/skwijeratne/agentix-toolkit/blob/main/examples/28_rate_limit.py)

## Fall back to another model

If one model (or provider) is down, automatically try the next one:

```
from agentix import FallbackModel

model = FallbackModel([primary_model, backup_model])
```

Useful for surviving an outage, or for "try the cheap model first, fall back to the big one."

## Validate the output

Make sure the answer is usable before your code relies on it. If it isn't, the agent re-asks the model:

```
from agentix import json_output

agent = Agent(model=m, system_prompt="Reply with JSON.",
              output_validator=json_output, max_output_retries=2)
outcome = await agent.run("...")
outcome.parsed     # the validated value — safe to use
```

For typed data with one setting, see **[Structured output](https://skwijeratne.github.io/agentix-toolkit/guides/structured-output/index.md)**.

→ Runnable example: [`examples/16_reliability.py`](https://github.com/skwijeratne/agentix-toolkit/blob/main/examples/16_reliability.py)

## Record and replay

Testing against a real model is slow, costs money, and gives different answers each time. `CassetteModel` records real responses **once** to a file, then **replays** them in later test runs — fast, free, and identical every time. (The name comes from recording onto a cassette tape.)

```
from agentix import CassetteModel

# First run records to the file; later runs replay from it. "auto" does the right
# thing based on whether the file already exists.
model = CassetteModel("tests/cassettes/weather.json", model=AnthropicModel())
# ... run the agent ...
model.save()
```

→ Runnable example: [`examples/29_cassettes.py`](https://github.com/skwijeratne/agentix-toolkit/blob/main/examples/29_cassettes.py)
