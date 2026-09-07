"""Hashing, atomic writes, deterministic seeds, and cache metadata."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import zlib

import pandas as pd
import numpy as np


def file_sha256(path: Path) -> str:
    """Return a file's hexadecimal SHA-256 digest.

    Args:
        path: File to hash.

    Returns:
        Lower-case hexadecimal digest.
    """
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha256(value: object) -> str:
    """Hash an object through a canonical JSON representation.

    Args:
        value: JSON-compatible object; unsupported scalar values use ``str``.

    Returns:
        Lower-case hexadecimal digest.
    """
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(payload.encode()).hexdigest()


def stable_seed(base: int, *parts: object) -> int:
    """Derive a stable positive seed.

    Args:
        base: Root integer seed.
        *parts: Label components identifying a distinct stochastic operation.

    Returns:
        Deterministic integer in NumPy's accepted positive range.
    """
    suffix = "|".join(str(part) for part in parts).encode()
    return int((base + zlib.crc32(suffix)) % (2**31 - 1))


def atomic_json(path: Path, value: object) -> None:
    """Atomically write an indented JSON document.

    Args:
        path: Destination file.
        value: JSON-compatible value.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, default=str)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    """Atomically serialise a data frame as Parquet.

    Args:
        path: Destination file.
        frame: Data frame to store without its index.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".parquet", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    """Atomically serialise a data frame as CSV.

    Args:
        path: Destination file.
        frame: Data frame to store without its index.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", suffix=".csv", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_npz(path: Path, **arrays: object) -> None:
    """Atomically write named arrays in compressed NPZ format.

    Args:
        path: Destination file.
        **arrays: Named NumPy-compatible values.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".npz", delete=False
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def assert_cache_fingerprint(metadata_path: Path, fingerprint: str) -> bool:
    """Return ``False`` for a missing cache and reject a stale cache.

    Args:
        metadata_path: JSON cache-metadata file.
        fingerprint: Expected deterministic recipe hash.

    Returns:
        ``True`` when existing metadata contains the expected fingerprint.
    """
    if not metadata_path.exists():
        return False
    metadata = json.loads(metadata_path.read_text())
    actual = metadata.get("fingerprint")
    if actual != fingerprint:
        raise RuntimeError(
            f"stale cache at {metadata_path}: expected fingerprint {fingerprint}, "
            f"found {actual}; move the cache aside or use a new output profile"
        )
    return True
