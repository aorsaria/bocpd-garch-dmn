"""Small deterministic archive writers used only by the test suite."""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

import pandas as pd


def frame_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serialise a canonical seven-column CLC frame for a test fixture.

    Args:
        frame: Data frame indexed by observation date, with OHLC, volume, and
            open-interest columns.

    Returns:
        ASCII bytes in the headerless format accepted by the experiment
        archive readers.
    """
    ordered = frame.copy().sort_index()
    ordered.index = pd.DatetimeIndex(ordered.index)
    ordered.index.name = "date"
    table = ordered.reset_index()
    table["date"] = table["date"].dt.strftime("%m/%d/%Y")
    return table.to_csv(index=False, header=False, lineterminator="\n").encode("ascii")


def deterministic_zip(path: Path, members: dict[str, bytes]) -> None:
    """Write sorted test members with fixed ZIP metadata.

    Args:
        path: Destination ZIP path.
        members: Mapping from archive member names to uncompressed bytes.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for name in sorted(members):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                info.create_system = 3
                archive.writestr(info, members[name])
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
