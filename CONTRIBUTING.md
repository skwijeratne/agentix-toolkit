# Contributing to agentix

Thanks for your interest in improving agentix! This guide gets you set up and
explains what we look for in a contribution.

> The distribution is **`agentix-toolkit`** on PyPI; you import it as **`agentix`**.

## Development setup

This project uses [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/skwijeratne/agentix-toolkit
cd agentix-toolkit
uv sync --all-extras        # create the venv, install deps + dev tools + extras
```

## The checks (all three must pass)

CI runs these on every PR across Python 3.10–3.13, and they are **blocking**:

```bash
uv run pytest                # tests
uv run ruff check src tests  # lint
uv run mypy                  # type-check (strict)
```

Run them locally before pushing. Optionally enable the pre-commit hooks so
lint runs automatically:

```bash
uv run pre-commit install
```

## Making a change

1. **Open an issue first** for anything non-trivial, so we can agree on the
   approach before you invest time.
2. Branch off `main`.
3. Keep the change focused. Match the surrounding style — small, shared core;
   load-bearing behavior is injected and configurable, not baked into the loop.
4. **Add tests.** New behavior needs coverage; bug fixes need a regression test.
   Tests are plain `def` / `async def test_*` functions (pytest, `asyncio_mode`
   is `auto`).
5. Update docs where relevant: docstrings, the README, an `examples/` script,
   and a `CHANGELOG.md` entry under `[Unreleased]`.
6. Make sure all three checks pass.
7. Open a PR using the template; describe the change and link the issue.

## Design principles

- **Provider-agnostic core.** Don't couple the loop to a specific model
  provider; provider code lives behind adapters (`providers/`).
- **Inject, don't bake in.** New capabilities should be opt-in and composable
  (a guard, a tool, a strategy, an executor), not hard-coded into `agent.py`.
- **Security defaults are conservative.** When a guard is ambiguous, fail
  closed. See `SECURITY.md`.
- **Typed and tested.** Public APIs are typed (`mypy --strict`) and exercised by
  tests.

## Reporting bugs / requesting features

Use the issue templates. For **security vulnerabilities, do not open a public
issue** — see [`SECURITY.md`](./SECURITY.md).

By contributing, you agree that your contributions are licensed under the
project's [MIT License](./LICENSE).
