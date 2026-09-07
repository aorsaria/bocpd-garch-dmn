"""Atomic writers for deterministic experiment artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import pandas as pd


def clean_changepoint_artifacts(project_root: Path) -> tuple[Path, Path]:
    """Remove only generated changepoint build and output trees.

    Args:
        project_root: Project directory containing ``build`` and ``outputs``.

    Returns:
        Resolved paths of the two scoped trees, whether or not they existed.
    """
    root = Path(project_root).resolve()
    build = (root / "build" / "changepoint").resolve()
    output = (root / "outputs" / "changepoint").resolve()
    for target in (build, output):
        if target.name != "changepoint" or target.parent.parent != root:
            raise RuntimeError(f"Refusing to clean unexpected path: {target}")
        if target.exists():
            shutil.rmtree(target)
    return build, output


def _temporary_path(path: Path) -> Path:
    """Create an empty sibling temporary file for an atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    return Path(name)


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically replace a file with UTF-8 text.

    Args:
        path: Destination file.
        text: Text to write.
    """
    temporary = _temporary_path(path)
    try:
        temporary.write_text(text)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: Any) -> None:
    """Atomically write an indented JSON document.

    Args:
        path: Destination file.
        payload: JSON-serialisable value.
    """
    atomic_write_text(path, json.dumps(payload, indent=2) + "\n")


def atomic_write_csv(
    frame: pd.DataFrame, path: Path, *, index: bool = False, **kwargs: Any
) -> None:
    """Atomically serialise a data frame as CSV.

    Args:
        frame: Data frame to serialise.
        path: Destination file.
        index: Whether to include the pandas index.
        **kwargs: Additional arguments forwarded to ``DataFrame.to_csv``.
    """
    temporary = _temporary_path(path)
    try:
        frame.to_csv(temporary, index=index, **kwargs)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Atomically serialise a data frame as Parquet without its index.

    Args:
        frame: Data frame to serialise.
        path: Destination file.
    """
    temporary = _temporary_path(path)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_copy(source: Path, destination: Path) -> None:
    """Atomically copy a file.

    Args:
        source: Existing source file.
        destination: Replacement destination.
    """
    temporary = _temporary_path(destination)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_npz(path: Path, arrays: dict[str, Any]) -> None:
    """Atomically write named arrays in compressed NPZ format.

    Args:
        path: Destination file.
        arrays: Mapping of array names to NumPy-compatible values.
    """
    temporary = _temporary_path(path)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
