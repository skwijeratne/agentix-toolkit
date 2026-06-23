# Releasing agentix-toolkit

> Distribution name: **`agentix-toolkit`** (PyPI). Import name: **`agentix`**.

## One-time setup: PyPI Trusted Publishing

The release workflow publishes via OIDC (no stored token). Since the project
doesn't exist on PyPI yet, add a *pending* publisher (the first publish creates
the project).

1. PyPI → **https://pypi.org/manage/account/publishing/** → add a pending publisher:
   - PyPI Project Name: `agentix-toolkit`
   - Owner: `skwijeratne` · Repository: `agentix-toolkit`
   - Workflow: `release.yml` · Environment: `pypi`
2. In the GitHub repo, create an **Environment** named `pypi`
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
