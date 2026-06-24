# Structured output

Often you don't want a paragraph of text back — you want **data** your program can
use: a name and an age, a list of items, a yes/no with a reason. Structured output
makes the model return clean, predictable data instead of prose.

## The one knob: `response_model`

Tell the agent the *shape* you want, and it handles everything:

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

agent = Agent(model=m, system_prompt="Extract the person.", response_model=Person)
outcome = await agent.run("Ada Lovelace, 36 years old.")

person = outcome.parsed          # a validated Person(name="Ada", age=36)
print(person.name, person.age)
```

`outcome.parsed` is the ready-to-use object. (`outcome.answer` is still the raw
text the model produced.)

## What it does for you

Setting `response_model` wires up three things at once:

1. **Checks the answer.** If the model's reply doesn't match the shape, the agent
   automatically **asks it to try again** (a few times) instead of handing you
   broken data.
2. **Tells the model the shape up front**, in plain instructions, so any model —
   even a basic one — knows what to produce.
3. **Turns on the provider's built-in enforcement** when available (Anthropic,
   OpenAI, Gemini, and others can guarantee valid output at their end), for the
   most reliable results.

You don't have to use [Pydantic](https://docs.pydantic.dev/) — you can pass a
plain description of the shape (a JSON Schema dictionary) instead, and
`outcome.parsed` will be a regular dictionary.

```python
schema = {"type": "object",
          "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
          "required": ["name", "age"]}
agent = Agent(model=m, system_prompt="...", response_model=schema)
```

!!! tip "Jargon, briefly"
    A **schema** is just a description of the shape of some data — which fields
    exist and what type each one is. **Validation** means checking that real data
    actually matches that shape.

→ Runnable example:
[`examples/27_structured_output.py`](https://github.com/skwijeratne/agentix-toolkit/blob/main/examples/27_structured_output.py)
