from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from webtoolbox.common.file_utils import InvalidFilenameError, ensure_within_dir, validate_filename

logger = logging.getLogger(__name__)


@dataclass
class BaseStore:
    base_dir: Path
    allowed_extensions: set[str]

    def __post_init__(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _safe_path(self, filename: str) -> Path:
        validated = validate_filename(filename)
        ext = Path(validated).suffix.lower().lstrip(".")
        if ext not in self.allowed_extensions:
            raise InvalidFilenameError("File extension is not allowed")
        return ensure_within_dir(self.base_dir / validated, self.base_dir)

    def list_files(self) -> list[str]:
        files = [
            item.name
            for item in self.base_dir.iterdir()
            if item.is_file() and item.suffix.lower().lstrip(".") in self.allowed_extensions
        ]
        files.sort()
        return files

    def save_bytes(self, filename: str, data: bytes) -> str:
        target = self._safe_path(filename)
        target.write_bytes(data)
        logger.info(
            "file_saved",
            extra={"action": "upload", "file": target.name, "directory": str(self.base_dir)},
        )
        return target.name

    def delete(self, filename: str) -> None:
        target = self._safe_path(filename)
        if target.exists():
            target.unlink()
            logger.info(
                "file_deleted",
                extra={"action": "delete", "file": target.name, "directory": str(self.base_dir)},
            )

    def rename(self, old_name: str, new_name: str) -> str:
        old_path = self._safe_path(old_name)
        new_path = self._safe_path(new_name)
        if not old_path.exists():
            raise FileNotFoundError(old_name)
        if new_path.exists():
            raise FileExistsError(new_name)
        old_path.rename(new_path)
        logger.info(
            "file_renamed",
            extra={
                "action": "rename",
                "old_file": old_path.name,
                "new_file": new_path.name,
                "directory": str(self.base_dir),
            },
        )
        return new_path.name

    def get_path(self, filename: str) -> Path:
        target = self._safe_path(filename)
        if not target.exists():
            raise FileNotFoundError(filename)
        logger.info(
            "file_accessed",
            extra={"action": "download", "file": target.name, "directory": str(self.base_dir)},
        )
        return target


class AudioStore(BaseStore):
    pass


class TextStore(BaseStore):
    pass
