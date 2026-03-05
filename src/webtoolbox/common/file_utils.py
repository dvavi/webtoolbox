from __future__ import annotations

import re
from pathlib import Path, PurePath

SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9._ -]+$")


class InvalidFilenameError(ValueError):
    """Raised when a user-provided filename is invalid."""


def validate_filename(filename: str) -> str:
    """Validate that filename is a simple, safe file name."""

    if not filename or filename in {".", ".."}:
        raise InvalidFilenameError("Filename is empty or invalid")

    pure = PurePath(filename)
    if pure.name != filename:
        raise InvalidFilenameError("Directory separators are not allowed")

    if not SAFE_FILENAME_RE.fullmatch(filename):
        raise InvalidFilenameError("Filename contains unsupported characters")

    return filename


def ensure_within_dir(path: Path, base_dir: Path) -> Path:
    """Ensure resolved path remains within the expected base directory."""

    resolved = path.resolve()
    resolved_base = base_dir.resolve()
    if resolved_base not in resolved.parents and resolved != resolved_base:
        raise InvalidFilenameError("Path escapes allowed directory")
    return resolved
