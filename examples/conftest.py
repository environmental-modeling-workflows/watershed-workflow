"""Apply pytest markers to notebooks based on cell tags in the first cell."""
import json
import pytest


def pytest_collect_file(parent, file_path):
    if file_path.suffix == ".ipynb":
        _apply_notebook_markers(parent, file_path)


def _apply_notebook_markers(parent, file_path):
    try:
        nb = json.loads(file_path.read_text())
    except Exception:
        return
    tags = nb.get("cells", [{}])[0].get("metadata", {}).get("tags", [])
    for tag in tags:
        if tag in ("network", "integration"):
            # Store tags on the file path for use in pytest_collection_modifyitems
            if not hasattr(parent.config, "_notebook_tags"):
                parent.config._notebook_tags = {}
            parent.config._notebook_tags[str(file_path)] = tags


def pytest_collection_modifyitems(config, items):
    notebook_tags = getattr(config, "_notebook_tags", {})
    for item in items:
        path = str(item.fspath)
        tags = notebook_tags.get(path, [])
        for tag in tags:
            if tag in ("network", "integration"):
                item.add_marker(getattr(pytest.mark, tag))
