"""Generate API reference pages automatically with improved formatting."""

from __future__ import annotations

from pathlib import Path

import mkdocs_gen_files

src = Path(__file__).parent.parent / "src"

root_dir = Path(__file__).parent.parent

# Copy CONTRIBUTING.md to docs directory
contributing_src = root_dir / "CONTRIBUTING.md"
contributing_dest = Path("CONTRIBUTING.md")
if contributing_src.exists():
    with open(contributing_src, encoding="utf-8") as src_fd:
        content = src_fd.read()
    with mkdocs_gen_files.open("CONTRIBUTING.md", "w") as fd:
        fd.write(content)
    mkdocs_gen_files.set_edit_path(Path("CONTRIBUTING.md"), contributing_src)

# Copy SECURITY.md to docs directory
security_src = root_dir / "SECURITY.md"
security_dest = Path("SECURITY.md")
if security_src.exists():
    with open(security_src, encoding="utf-8") as src_fd:
        content = src_fd.read()
    with mkdocs_gen_files.open("SECURITY.md", "w") as fd:
        fd.write(content)
    mkdocs_gen_files.set_edit_path(Path("SECURITY.md"), security_src)

# Generate the index page
with mkdocs_gen_files.open("reference/index.md", "w") as fd:
    fd.write("# API Reference\n\n")
    fd.write("Complete API documentation for the gwsim package.\n\n")
    fd.write("## Modules\n\n")

    # Collect unique subpackages (immediate children of gwsim)
    modules = set()
    gwsim_dir = src / "gwsim"
    if gwsim_dir.exists():
        for item in gwsim_dir.iterdir():
            if item.is_dir() and (item / "__init__.py").exists() and not item.name.startswith("_"):
                modules.add(item.name)

    # Generate module list
    for module in sorted(modules):
        fd.write(f"- [`gwsim.{module}`](gwsim/{module}/index.md)\n")

# Process all Python modules
for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    # Skip private modules and test files
    if any(part.startswith("_") and part != "__init__" for part in parts):
        continue
    if parts and parts[-1].startswith("test_"):
        continue

    # Skip empty parts (from root __init__)
    if not parts:
        continue

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        indent = ".".join(parts)

        # Generate improved page content
        fd.write(f"# `{indent}`\n\n")
        fd.write("::: " + indent + "\n")
        fd.write("    options:\n")
        fd.write("      docstring_style: google\n")
        fd.write("      show_source: true\n")
        fd.write("      show_root_heading: true\n")
        fd.write("      show_object_full_path: true\n")
        fd.write("      members_order: source\n")
        fd.write("      filters:\n")
        fd.write("        - '!^_'\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)
