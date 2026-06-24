"""Generate one API-reference page per public agentix module.

Run automatically by the mkdocs-gen-files plugin at build time: it walks the
source tree, writes a virtual ``reference/<module>.md`` containing a mkdocstrings
``:::`` directive for each module, and builds a ``SUMMARY.md`` that the
literate-nav plugin turns into the "API reference" nav section. Private modules
(leading underscore) are skipped.
"""

from __future__ import annotations

from pathlib import Path

import mkdocs_gen_files

SRC = Path("src")
PACKAGE = "agentix"

nav = mkdocs_gen_files.Nav()

for path in sorted((SRC / PACKAGE).rglob("*.py")):
    module_path = path.relative_to(SRC).with_suffix("")
    parts = list(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
    # Skip the top-level package page (just re-exports) and any private module.
    if len(parts) < 2 or any(p.startswith("_") for p in parts[1:]):
        continue

    doc_path = Path(*parts[1:]).with_suffix(".md")  # drop the "agentix/" prefix
    full_doc_path = Path("reference", doc_path)
    identifier = ".".join(parts)

    nav[tuple(parts[1:])] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(f"# `{identifier}`\n\n::: {identifier}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
