"""Export source-only GitHub content, preserving local notebook outputs and widget defaults.

Usage: python scripts/export_source.py --output .cache/github_new_repo
The destination must not already contain files. No data, runs, weights or environments are copied.
"""

import argparse
import json
from pathlib import Path
import shutil


def export_source(destination):
    root = Path(__file__).resolve().parents[1]
    destination = Path(destination).resolve()
    if destination == root or destination in root.parents:
        raise ValueError("Export into a separate directory, not the source project or its parents")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Export destination must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in (".gitignore", "README.md", "pyproject.toml", "requirements-lock.txt"):
        shutil.copy2(root / name, destination / name)
        copied.append(name)
    for folder in ("src", "scripts", "tests", "notebooks"):
        suffix = ".ipynb" if folder == "notebooks" else ".py"
        for source in sorted((root / folder).rglob("*" + suffix)):
            if "__pycache__" in source.parts or ".ipynb_checkpoints" in source.parts:
                continue
            relative = source.relative_to(root)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if suffix == ".ipynb":
                notebook = json.loads(source.read_text(encoding="utf-8"))
                original_sources = [cell["source"] for cell in notebook["cells"]]
                notebook["metadata"] = {key: value for key, value in notebook.get("metadata", {}).items()
                                        if key in ("kernelspec", "language_info")}
                for cell in notebook["cells"]:
                    cell["metadata"] = {"tags": cell["metadata"]["tags"]} if "tags" in cell.get("metadata", {}) else {}
                    cell.pop("attachments", None)
                    if cell["cell_type"] == "code":
                        cell["outputs"] = []
                        cell["execution_count"] = None
                assert [cell["source"] for cell in notebook["cells"]] == original_sources
                target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            else:
                shutil.copy2(source, target)
            copied.append(relative.as_posix())
    print(f"Exported {len(copied)} source/documentation files to {destination}")
    return copied


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    export_source(parser.parse_args().output)


if __name__ == "__main__":
    main()
