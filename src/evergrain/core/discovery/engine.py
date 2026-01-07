from __future__ import annotations

import platform
from pathlib import Path
from typing import Optional

import psutil

from evergrain.core.exceptions.discovery import DiscoveryError


class Discovery:
    """
    OS-agnostic removable-volume locator for Evergrain sources.
    """

    def __init__(
        self,
        marker: str = ".evergrain",
        sub_path: str | Path = "EPSCAN/001",
    ) -> None:
        self.marker = marker
        self.sub_path = Path(sub_path)

    def find(self, override: Optional[str | Path] = None) -> Path:
        """Return resolved Path to the sub-directory; raise DiscoveryError."""
        if override:
            override = Path(override)
            root = override.resolve()
            if not root.is_dir():
                raise DiscoveryError(f"Override path is not a directory: {root}")
            drives = [root]
        else:
            drives = self._removable_volumes()

        hits: list[Path] = []

        for vol in drives:
            marker_file = vol / self.marker
            if not marker_file.is_file():
                continue
            if not self._valid_marker(marker_file):
                continue

            sub = (vol / self.sub_path).resolve()
            if not sub.is_dir():
                continue
            hits.append(sub)

        if not hits:
            raise DiscoveryError("No removable source found.")
        if len(hits) > 1:
            raise DiscoveryError(f"Multiple sources found: {hits}")
        return hits[0]

    @staticmethod
    def _valid_marker(path: Path) -> bool:
        """First line must be b'evergrain-source v1'."""
        try:
            with path.open("rb") as fh:
                head = fh.readline().rstrip(b"\r\n")
            return head == b"evergrain-source v1"
        except OSError:
            return False

    @staticmethod
    def _removable_volumes() -> list[Path]:
        """Return list of root Paths that are removable media."""
        removable: list[Path] = []
        for part in psutil.disk_partitions(all=False):
            # Windows -> "removable",  macOS/Linux -> "msdos"/"exfat" without "noauto"
            if part.opts and "removable" in part.opts:
                removable.append(Path(part.mountpoint))
            elif platform.system() != "Windows":
                # Linux/macOS: treat anything mounted under /media or /Volumes as removable
                if part.mountpoint.startswith(("/media/", "/Volumes/")):
                    removable.append(Path(part.mountpoint))
        return removable
