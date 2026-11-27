"""Runtime environment policy for the MetaX Docker image."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from pathlib import Path


def check_archived_environment(expected: dict[str, str], expected_python: tuple[int, int] = (3, 11)) -> dict[str, str | None]:
    """Record reference differences and continue to real import/inference checks."""
    actual: dict[str, str | None] = {}
    missing: list[str] = []
    for name in expected:
        try:
            actual[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual[name] = None
            missing.append(name)

    mismatches = {name: (expected[name], actual[name]) for name in expected if actual[name] != expected[name]}
    if sys.version_info[:2] != expected_python or mismatches:
        allowlist = Path(__file__).with_name("allowed_metax_environment.json")
        allowlisted_python = None
        if allowlist.is_file():
            try:
                allowlisted_python = json.loads(allowlist.read_text()).get("python")
            except (OSError, json.JSONDecodeError):
                allowlisted_python = None
        print(
            "WARNING: MetaX allowlisted environment differs from archived reference "
            f"(python={platform.python_version()}, allowlisted_python={allowlisted_python}, "
            f"missing={missing}, mismatches={mismatches})",
            file=sys.stderr,
        )
    return actual
