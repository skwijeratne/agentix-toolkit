# Releasing agentix

## One-time setup: PyPI Trusted Publishing

The release workflow publishes via OIDC (no stored token).

1. Create the project on PyPI (and optionally TestPyPI) if it doesn't exist.
2. On PyPI → the project → **Publishing** → add a *Trusted Publisher*:
   - Owner: `skwijeratne` · Repository: `agentix`
   - Workflow: `release.yml` · Environment: `pypi`
3. In the GitHub repo, create an **Environment** named `pypi`
   (Settings → Environments). Add reviewers if you want a manual approval gate.

## Cutting a release

1. Ensure CI is green on `main`.
2. Bump the version in `pyproject.toml` and move the `CHANGELOG.md`
   "Unreleased" notes under the new version.
3. Commit, then tag and push:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```
4. The `Release` workflow builds, runs `twine check`, and publishes to PyPI.

## Dry run (recommended first time)

Test the build locally before tagging (this project uses [uv](https://docs.astral.sh/uv/)):
```bash
uv build
uvx twine check dist/*
```
To rehearse the upload end-to-end, publish to **TestPyPI** first
(`uvx twine upload --repository testpypi dist/*`) and install from there.
