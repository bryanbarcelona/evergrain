import os
import tempfile
from pathlib import Path
from typing import Callable


def atomic_replace(target: Path, saver: Callable[[Path], None]) -> None:
    """
    Atomically replace `target` with content produced by `saver`.

    Parameters
    ----------
    target : Path
        File to be replaced.
    saver : Callable[[Path], None]
        Receives a temporary Path; must write the desired content to it.
        On success the temp file replaces `target`.
        On any exception the temp file is removed and `target` is untouched.
    """
    target = target.resolve()
    tmp_dir = target.parent
    fd, tmp_path = tempfile.mkstemp(dir=tmp_dir, prefix=target.stem + ".", suffix=target.suffix)
    os.close(fd)
    try:
        saver(Path(tmp_path))
        os.replace(tmp_path, target)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise
